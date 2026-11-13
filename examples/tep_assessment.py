#!/usr/bin/env python3
"""Assessment of the classic Tennessee Eastman benchmark, the running example of the survey.

The script scans the 44 files, reads the label definition, measures the overlap of random
window splits, runs a small task test under two splits (Algorithm 2) and prints quality claims
as six-tuples (Definition 4). The full experiment with five model classes and 20 seeds is
case_tep/run_case.py.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from hqd_assess import (AssessmentLevel, Estimate, QualityClaim, UsageContext, bootstrap_metric_interval, check_split_independence,
                        dimension_score, expected_performance_band, macro_f1, model_application_score, model_fit_threshold,
                        model_fit_value, passes, sliding_window_starts, window_row_overlap)

ONSET, VARIABLES, FILES = 160, 52, 22


def read_files(folder):
    """Read d00 to d21 and their test files, and record which files have the documented shape."""
    data, shapes = {}, {}
    for k in range(FILES):
        for suffix in ("", "_te"):
            name = f"d{k:02d}{suffix}.dat"
            a = np.loadtxt(os.path.join(folder, name))
            shapes[name] = a.shape
            data[name] = a.T if a.shape[1] != VARIABLES and a.shape[0] == VARIABLES else a
    return data, shapes


def windows(view, length):
    """All windows of the runs in view, flattened, with class, run and start row."""
    xs, ys, starts = [], [], []
    for k, a in enumerate(view):
        for s in sliding_window_starts(len(a), length):
            xs.append(a[s:s + length].ravel())
            ys.append(k)
            starts.append(s)
    return np.asarray(xs, np.float32), np.asarray(ys), np.asarray(starts)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "tep"), help="folder with the 44 .dat files")
    ap.add_argument("--input", choices=("rows", "windows"), default="rows", help="model input for the task test")
    ap.add_argument("--window", type=int, default=20, help="window length in rows")
    ap.add_argument("--trees", type=int, default=30, help="trees of the random forest in the task test")
    ap.add_argument("--expected", type=float, default=0.80, help="expected performance a* of the model fit value")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-task", action="store_true", help="skip the model training")
    args = ap.parse_args()

    data, shapes = read_files(args.data)
    train = [data[f"d{k:02d}.dat"] for k in range(FILES)]
    test = [data[f"d{k:02d}_te.dat"] for k in range(FILES)]
    records = np.vstack(list(data.values()))

    print("Static scan")
    nonconforming = [name for name, shape in shapes.items() if shape[1] != VARIABLES]
    duplicates = records.shape[0] - np.unique(records.round(8), axis=0).shape[0]
    constant = [j for j in range(records.shape[1]) if np.ptp(records[:, j]) == 0]
    missing = int(np.isnan(records).sum())
    non_finite = int((~np.isfinite(records)).sum())
    for label, value in (("files", len(shapes)), ("files with the documented shape", len(shapes) - len(nonconforming)),
                         ("files with rows and columns swapped", ", ".join(nonconforming) or "none"),
                         ("records", records.shape[0]), ("variables", records.shape[1]), ("missing values", missing),
                         ("non-finite values", non_finite), ("duplicate records", duplicates), ("constant variables", len(constant))):
        print(f"  {label:38s} {value}")

    file_conformity = 1 - len(nonconforming) / len(shapes)
    completeness = 1 - missing / records.size
    uniqueness = 1 - duplicates / records.shape[0]
    print(f"  {'score of the three ratios':38s} {dimension_score([file_conformity, completeness, uniqueness]):.1f}")

    print("\nLabel check")
    pre_onset_rows = (FILES - 1) * ONSET
    test_rows = sum(len(a) for a in test)
    label_accuracy = 1 - pre_onset_rows / test_rows
    print(f"  {'pre-onset share of a test fault file':38s} {ONSET / len(test[1]):.3f}")
    print(f"  {'file-name label accuracy':38s} {label_accuracy:.3f}")

    print("\nWindow overlap under a random split")
    rng = np.random.default_rng(args.seed + 7)
    post = [test[0]] + [a[ONSET:] for a in test[1:]]
    runs, starts = [], []
    for run, a in enumerate(train + post):
        s = sliding_window_starts(len(a), args.window)
        runs.append(np.full(len(s), run))
        starts.append(s)
    runs, starts = np.concatenate(runs), np.concatenate(starts)
    order = rng.permutation(len(runs))
    cut = int(0.7 * len(order))
    overlap = window_row_overlap(runs[order[:cut]], starts[order[:cut]], runs[order[cut:]], starts[order[cut:]], args.window)
    print(f"  {'test windows sharing a training row':38s} {overlap['windows_with_any_overlap']:.3f}")
    print(f"  {'test window rows found in training':38s} {overlap['rows_also_in_train']:.3f}")

    usage = UsageContext(task="fault diagnosis", metric="Macro-F1", reference="split by run", model_class=("random forest",),
                         scene="base operating mode", budget="full scan")
    claims = [
        QualityClaim("TEP classic, all files", "format conformity", "share of files with the documented shape",
                     Estimate.full_scan(file_conformity), AssessmentLevel.L3, usage),
        QualityClaim("TEP classic, all files", "structural completeness", "share of non-missing values",
                     Estimate.full_scan(completeness), AssessmentLevel.L3, usage),
        QualityClaim("TEP classic, test files", "annotation accuracy", "file-name label accuracy",
                     Estimate.full_scan(label_accuracy), AssessmentLevel.L3, usage),
        QualityClaim("TEP classic, random window split", "independence of the split", "test windows sharing a training row",
                     Estimate.full_scan(overlap["windows_with_any_overlap"]), AssessmentLevel.L3, usage),
    ]

    if not args.no_task:
        from sklearn.ensemble import RandomForestClassifier
        print(f"\nTask test, random forest with {args.trees} trees, input {args.input}")
        if args.input == "rows":
            xa, ya = np.vstack(train), np.concatenate([np.full(len(a), k) for k, a in enumerate(train)])
            xb, yb = np.vstack(post), np.concatenate([np.full(len(a), k) for k, a in enumerate(post)])
            limit_train, limit_test = len(ya), len(yb)
        else:
            xa, ya, _ = windows(train, args.window)
            xb, yb, _ = windows(post, args.window)
            limit_train, limit_test = 8000, 6000
        x, y = np.vstack([xa, xb]), np.r_[ya, yb]
        run_key = np.r_[ya, yb + 100]
        order = rng.permutation(len(y))
        cut = int(0.7 * len(y))
        random_split = (order[:cut][:limit_train], order[cut:][:limit_test])
        official = (rng.permutation(len(ya))[:limit_train], len(ya) + rng.permutation(len(yb))[:limit_test])

        def evaluate(tr, te):
            model = RandomForestClassifier(n_estimators=args.trees, random_state=args.seed, n_jobs=-1).fit(x[tr], y[tr])
            evaluate.last = (y[te], model.predict(x[te]))
            return macro_f1(*evaluate.last)

        result = check_split_independence(*random_split, {"run": run_key}, evaluate, resplit={"run": official})["run"]
        low, high = bootstrap_metric_interval(*evaluate.last, macro_f1, seed=args.seed)
        print(f"  {'overlap of runs under the random split':38s} {result.overlap:.3f}")
        print(f"  {'Macro-F1, random split':38s} {result.score_given:.3f}")
        print(f"  {'Macro-F1, split by run':38s} {result.score_grouped:.3f}")
        print(f"  {'gap':38s} {result.gap:.3f}")
        s_run = model_fit_value(result.score_grouped, args.expected)
        s_random = model_fit_value(result.score_given, args.expected)
        for label, s_m in (("split by run", s_run), ("random split", s_random)):
            score = model_application_score(s_m)
            print(f"  {'model application score, ' + label:38s} {score:.1f}, s_m = {s_m:.3f}, passes {passes(score)}")
        band = expected_performance_band(result.score_grouped, result.score_given)
        print(f"  {'fails below s_m':38s} {model_fit_threshold():.3f}")
        print(f"  {'expected values where the split decides':38s} {band[0]:.3f} to {band[1]:.3f}")
        claims.append(QualityClaim("TEP classic, split by run", "model fit", "Macro-F1",
                                   Estimate(result.score_grouped, low, high, "bootstrap over test records"), AssessmentLevel.L4, usage))

    print("\nQuality claims")
    for claim in claims:
        print(f"  {claim}")


if __name__ == "__main__":
    main()
