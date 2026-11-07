#!/usr/bin/env python3
"""Finite-population intervals of the detector experiment, written to results_finite_population.jsonl.

The exact enumeration checks coverage at the requested confidence, rather than
tuning confidence against simulations.  Optional stored detector flags replay
the earlier study's 500 audit draws and balanced-stratum draws with its original
random-seed convention.  Truth is used to simulate audits and score coverage;
the interval API receives only audited counts and full-population stratum sizes.

Example (the output directory must not already exist)::

    python analysis/validate_detector_intervals.py --output-dir /tmp/interval-v1
    python analysis/validate_detector_intervals.py --flags-dir /path/flags \
        --previous-results results/results_detectors.jsonl \
        --tables smartfactory --detectors D1 --units record,cell \
        --output-dir /tmp/interval-v1-replay
"""

import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import sys
import time
import zlib

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hqd_assess.finite_population import (  # noqa: E402
    METHOD, hypergeometric_count_interval, interval_cache_info,
    stratified_error_interval,
)


def seeded(*parts):
    """Exactly the seed convention of study/run_detectors.py."""
    return np.random.default_rng(zlib.crc32(
        json.dumps(parts, sort_keys=True).encode()) % (2 ** 32))


def probabilities(population, errors, n):
    """Exact finite-population sampling probabilities, without scipy tails."""
    denominator = math.comb(population, n)
    return [(k, Fraction(math.comb(errors, k) *
                         math.comb(population - errors, n - k), denominator))
            for k in range(max(0, n - population + errors), min(n, errors) + 1)]


def enumerate_coverage(confidence=0.95, single_max=14, strata_max=9):
    """Exhaust all finite populations/truth counts/audit sizes in a small grid."""
    target = Fraction(str(confidence))
    result = {}
    worst, worst_case, cases = Fraction(1), None, 0
    for size in range(single_max + 1):
        for n in range(size + 1):
            intervals = {k: hypergeometric_count_interval(k, n, size, confidence)
                         for k in range(n + 1)}
            for errors in range(size + 1):
                coverage = sum((p for k, p in probabilities(size, errors, n)
                                if intervals[k][0] <= errors <= intervals[k][1]),
                               Fraction())
                cases += 1
                if coverage < worst:
                    worst, worst_case = coverage, dict(N=size, K=errors, n=n)
    result["single_stratum"] = dict(cases=cases, minimum_coverage=float(worst),
                                    exact_minimum=str(worst), worst_case=worst_case,
                                    passed=worst >= target)
    worst, worst_case, cases = Fraction(1), None, 0
    for sizes in itertools.product(range(strata_max + 1), repeat=2):
        if sum(sizes) == 0:
            continue
        for samples in itertools.product(*(range(size + 1) for size in sizes)):
            intervals = {}
            for counts in itertools.product(*(range(n + 1) for n in samples)):
                r = stratified_error_interval(counts, samples, sizes, confidence)
                intervals[counts] = (sum(x[0] for x in r.count_intervals),
                                     sum(x[1] for x in r.count_intervals))
            for errors in itertools.product(*(range(size + 1) for size in sizes)):
                covered = Fraction()
                p1, p0 = (probabilities(size, error, n)
                          for size, error, n in zip(sizes, errors, samples))
                for (k1, a), (k0, b) in itertools.product(p1, p0):
                    low, high = intervals[k1, k0]
                    if low <= sum(errors) <= high:
                        covered += a * b
                cases += 1
                if covered < worst:
                    worst, worst_case = covered, dict(N=sizes, K=errors, n=samples)
    result["two_strata_conditional"] = dict(
        cases=cases, minimum_coverage=float(worst), exact_minimum=str(worst),
        worst_case=worst_case, passed=worst >= target)
    # Explicitly integrate the random stratum allocation of a simple random
    # sample, as well as checking all conditional allocations above.
    sizes = (strata_max, strata_max + 1)
    worst, worst_case, cases = Fraction(1), None, 0
    for n in range(sum(sizes) + 1):
        allocations = probabilities(sum(sizes), sizes[0], n)
        for errors in itertools.product(*(range(size + 1) for size in sizes)):
            covered = Fraction()
            for n1, allocation_probability in allocations:
                samples = (n1, n - n1)
                p1, p0 = (probabilities(size, error, nh)
                          for size, error, nh in zip(sizes, errors, samples))
                for (k1, a), (k0, b) in itertools.product(p1, p0):
                    r = stratified_error_interval((k1, k0), samples, sizes, confidence)
                    if sum(v[0] for v in r.count_intervals) <= sum(errors) <= sum(
                            v[1] for v in r.count_intervals):
                        covered += allocation_probability * a * b
            cases += 1
            if covered < worst:
                worst, worst_case = covered, dict(N=sizes, K=errors, n=n)
    result["srs_random_allocation"] = dict(
        cases=cases, minimum_coverage=float(worst), exact_minimum=str(worst),
        worst_case=worst_case, passed=worst >= target)
    result["confidence"] = confidence
    result["passed"] = all(v["passed"] for v in result.values() if isinstance(v, dict))
    return result


def source_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def vector_unit(data, unit, names):
    truth = data["E"].astype(bool)
    if unit == "cell":
        return truth.ravel(), {d: data[d].astype(bool).ravel() for d in names
                               if data[d].ndim == 2}
    truth = truth.any(1)
    if unit == "record_label":
        truth = truth | data["yE"].astype(bool)
    return truth, {d: data[d].astype(bool).any(1) if data[d].ndim == 2
                   else data[d].astype(bool) for d in names}


def summarize_intervals(count1, n1, count0, n0, sizes, confidence, truth):
    arrays = np.broadcast_arrays(count1, n1, count0, n0)
    values = [stratified_error_interval((int(k1), int(k0)), (int(a), int(b)),
                                       sizes, confidence)
              for k1, a, k0, b in zip(*(x.ravel() for x in arrays))]
    estimates = np.array([v.estimate for v in values], dtype=float)
    lows, highs = np.array([v.low for v in values]), np.array([v.high for v in values])
    return dict(mean_est=float(estimates.mean()),
                mean_abs_error=float(np.abs(estimates - truth).mean()),
                coverage=float(((lows <= truth + 1e-12) & (highs >= truth - 1e-12)).mean()),
                mean_half_width=float(((highs - lows) / 2).mean()),
                minimum_width=float((highs - lows).min()),
                share_point_fallback=float(np.mean([bool(v.unsampled_strata) for v in values])),
                maximum_missed_stratum_population_share=float(max(
                    sum(v.population_sizes[i] for i in v.unsampled_strata) /
                    sum(v.population_sizes) for v in values)))


def selected(text):
    return set(x for x in text.split(",") if x)


def replay_flags(args):
    """Replay draws, producing new rows; existing flags/results remain read-only."""
    tables, detectors, units = selected(args.tables), selected(args.detectors), selected(args.units)
    previous = {}
    if args.previous_results:
        with args.previous_results.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                r = json.loads(line)
                key = r["ds"], r["unit"], r["detector"], r["n"], r["method"]
                if key in previous:
                    raise ValueError("duplicate key in previous results: {}".format(key))
                previous[key] = r
    rows, reproduction, sources = [], [], {}
    paths = sorted(p for p in args.flags_dir.glob("*.npz")
                   if not p.name.endswith(".raha.npz") and (not tables or p.stem in tables))
    if not paths:
        raise ValueError("no matching stored detector flag files")
    for path in paths:
        sources[str(path.resolve())] = source_hash(path)
        with np.load(path, allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
        raha_path = path.with_name(path.stem + ".raha.npz")
        if raha_path.exists():
            sources[str(raha_path.resolve())] = source_hash(raha_path)
            with np.load(raha_path, allow_pickle=False) as z:
                data["D6"] = z["D6"]
        names = sorted(k for k in data if k not in ("E", "yE") and
                       (not detectors or k in detectors))
        available_units = ["record", "cell"] + (["record_label"] if data["yE"].any() else [])
        for unit in available_units:
            if unit not in units:
                continue
            t, flags = vector_unit(data, unit, names)
            size, truth = len(t), float(t.mean())
            for n in args.sizes:
                if n > size:
                    continue
                rng = seeded(path.stem, unit, n, args.seed, "srs")
                index = np.stack([rng.choice(size, n, replace=False) for _ in range(args.repeats)])
                sample_truth = t[index]
                for detector, f in flags.items():
                    if len(f) != size:
                        raise ValueError("flag/truth length mismatch")
                    positive = int(f.sum())
                    sizes = positive, size - positive
                    sf = f[index]
                    tp, fp = (sf & sample_truth).sum(1), (sf & ~sample_truth).sum(1)
                    fn, tn = (~sf & sample_truth).sum(1), (~sf & ~sample_truth).sum(1)
                    results = [("finite_poststratified", "poststratified", "srs_conditioned_on_stratum_counts",
                                summarize_intervals(tp, tp + fp, fn, fn + tn, sizes,
                                                    args.confidence, truth))]
                    n1 = min(n // 2, sizes[0])
                    n0 = min(n - n1, sizes[1])
                    n1 = min(n - n0, sizes[0])
                    rng = seeded(path.stem, unit, n, args.seed, detector, "strat")
                    k1, k0 = int((f & t).sum()), int((~f & t).sum())
                    draw1 = rng.hypergeometric(k1, sizes[0] - k1, n1, args.repeats) if n1 else np.zeros(args.repeats, int)
                    draw0 = rng.hypergeometric(k0, sizes[1] - k0, n0, args.repeats) if n0 else np.zeros(args.repeats, int)
                    results.append(("finite_stratified", "stratified", "balanced_strata_srs_without_replacement",
                                    summarize_intervals(draw1, n1, draw0, n0, sizes,
                                                        args.confidence, truth)))
                    for method, old_method, design, stats in results:
                        row = dict(ds=path.stem, detector=detector, unit=unit, n=n,
                                   N=size, population_positive=positive, true_rate=truth,
                                   positive_rate=positive / size, repeats=args.repeats,
                                   seed=args.seed, method=method, interval_method=METHOD,
                                   confidence=args.confidence, sampling_design=design, **stats)
                        baseline = {m: previous.get((path.stem, unit, detector, n, m))
                                    for m in ("sample", "corrected", old_method)}
                        row["previous"] = {m: {key: r.get(key) for key in
                                               ("repeats", "mean_est", "mean_abs_error", "coverage", "mean_half_width")}
                                           for m, r in baseline.items() if r is not None}
                        old = baseline[old_method]
                        if args.previous_results:
                            same_repeats = old is not None and old.get("repeats") == args.repeats
                            differences = {key: abs(stats[key] - old[key]) for key in
                                           ("mean_est", "mean_abs_error")} if same_repeats else {}
                            reproduction.append(dict(ds=path.stem, unit=unit, detector=detector,
                                                     n=n, method=method, baseline_present=old is not None,
                                                     same_repeats=same_repeats, differences=differences,
                                                     passed=bool(differences) and max(differences.values()) <= 1.1e-6))
                        rows.append(row)
                print(json.dumps(dict(ds=path.stem, unit=unit, n=n, rows=len(rows))), flush=True)
    if not rows:
        raise ValueError("no selected detector/unit/sample-size combinations")
    return rows, reproduction, sources


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--flags-dir", type=Path)
    ap.add_argument("--previous-results", type=Path)
    ap.add_argument("--tables", default="")
    ap.add_argument("--detectors", default="")
    ap.add_argument("--units", default="record,cell,record_label")
    ap.add_argument("--sizes", default="50,100,200,400,800")
    ap.add_argument("--repeats", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument("--single-max", type=int, default=14)
    ap.add_argument("--strata-max", type=int, default=9)
    ap.add_argument("--skip-enumeration", action="store_true")
    args = ap.parse_args()
    args.sizes = [int(x) for x in args.sizes.split(",")]
    if not 0 < args.confidence < 1 or args.repeats < 1 or min(args.sizes) < 1:
        ap.error("invalid confidence, repeats, or sample sizes")
    if args.single_max < 1 or args.strata_max < 1:
        ap.error("enumeration population limits must be positive")
    if args.previous_results and not args.flags_dir:
        ap.error("--previous-results requires --flags-dir")
    if not selected(args.units) <= {"record", "cell", "record_label"}:
        ap.error("unknown unit")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    start = time.time()
    enumeration = None if args.skip_enumeration else enumerate_coverage(
        args.confidence, args.single_max, args.strata_max)
    rows, reproduction, sources = replay_flags(args) if args.flags_dir else ([], [], {})
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["unit"], r["n"], r["method"]].append(r)
    summary = [dict(unit=key[0], n=key[1], method=key[2], pairs=len(value),
                    mean_coverage=float(np.mean([r["coverage"] for r in value])),
                    minimum_coverage=min(r["coverage"] for r in value),
                    mean_abs_error=float(np.mean([r["mean_abs_error"] for r in value])),
                    mean_half_width=float(np.mean([r["mean_half_width"] for r in value])))
               for key, value in sorted(grouped.items())]
    report = dict(method=METHOD, confidence=args.confidence, enumeration=enumeration,
                  real_conditions=len(rows), replay_reproduction=reproduction,
                  summary=summary, interval_cache=interval_cache_info(),
                  seconds=round(time.time() - start, 3),
                  passed=(enumeration is None or enumeration["passed"]) and
                         all(r["passed"] for r in reproduction),
                  note="Monte Carlo coverage is reported, never used to tune confidence or as a per-case pass threshold.")
    (args.output_dir / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (args.output_dir / "results_finite_population.jsonl").open("x") as stream:
        for r in rows:
            stream.write(json.dumps(r, allow_nan=False) + "\n")
    if args.previous_results:
        sources[str(args.previous_results.resolve())] = source_hash(args.previous_results)
    sources[str(Path(__file__).resolve())] = source_hash(__file__)
    module = ROOT / "hqd_assess/finite_population.py"
    sources[str(module)] = source_hash(module)
    manifest = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                    arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    sources=sources, preserved_previous_results=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(output=str(args.output_dir), passed=report["passed"],
                          enumeration=enumeration, real_conditions=len(rows), seconds=report["seconds"])))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
