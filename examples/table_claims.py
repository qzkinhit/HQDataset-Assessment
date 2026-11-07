#!/usr/bin/env python3
"""Quality claims on one table with a dirty version and a clean reference.

The script computes the record and cell error rates and the score of the standard, audits a sample
with intervals, calibrates a rule detector (Algorithm 1), chooses among three versions of the
training data (Algorithm 3), and judges the claims (Algorithm 5). The full experiments on all
tables are study/run_real_tables.py and study/run_decision.py.
"""
import argparse
import json
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "study")]
warnings.filterwarnings("ignore")
from hqd_assess import (AssessmentLevel, Estimate, QualityClaim, Requirement, UsageContext, calibrate_error_rate, data_quality_score,
                        hindsight_best, jeffreys_interval, judge, macro_f1, passes, proportion_sample_size, r2, select_version,
                        split_holdout, static_rule, wilson_interval)
from hqd_assess.sampling import stratified_audit
from run_real_tables import NA, load, models

MIN_RECORDS = 30


def read_rules(path):
    """Read the DOMAIN and FD sections of a rule file. Other sections are left out of this example."""
    domain, fds, section = {}, [], None
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("["):
            section = line.strip("[]")
        elif section == "DOMAIN":
            column, _, rule = line.partition(":")
            match = re.match(r"(ENUM)\s*\{(.*)\}|(INT|FLOAT)\s*\[(.*),(.*)\]", rule.strip())
            if match and match.group(1):
                domain[column.strip()] = {v.strip() for v in match.group(2).split(",")}
            elif match:
                domain[column.strip()] = (float(match.group(4)), float(match.group(5)))
        elif section == "FD" and "=>" in line:
            left, right = line.split("=>")
            fds.append((left.strip(), right.strip()))
    return domain, fds


def rule_detector(table, domain, fds):
    """Flag a record with a missing token, a domain violation or a value outside the majority of its FD group.

    A column that is mostly empty in the table carries no missing-value signal and is not read for missing tokens.
    """
    flags = np.zeros(len(table), bool)
    text = table.astype(str).apply(lambda s: s.str.strip())
    for column in text.columns:
        missing = text[column].isin(NA).values
        if missing.mean() < 0.5:
            flags |= missing
    for column, rule in domain.items():
        if column not in text.columns:
            continue
        if isinstance(rule, set):
            flags |= ~text[column].isin(rule).values
        else:
            value = pd.to_numeric(text[column], errors="coerce").values
            flags |= ~((value >= rule[0]) & (value <= rule[1]))
    for left, right in fds:
        if left in text.columns and right in text.columns:
            majority = text.groupby(left)[right].transform(lambda s: s.mode().iloc[0])
            flags |= (text[right] != majority).values
    return flags


def version_scores(data, seed, model_name):
    """Train the model on the raw, repaired and deleted versions and predict the clean test records."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    n = len(data["a"])
    order = np.random.default_rng(seed + 500).permutation(n)
    cut = int(0.7 * n)
    train, test = order[:cut], order[cut:]
    wrong = data["E"][train].any(axis=1) | data["yE"][train]
    frames = dict(raw=(data["a"], np.ones(len(train), bool)), repair=(data["b"], np.ones(len(train), bool)))
    if (~wrong).sum() >= MIN_RECORDS:
        frames["delete"] = (data["a"], ~wrong)
    x_test = data["b"].iloc[test][data["feat"]].reset_index(drop=True)
    y_test = data["b"].iloc[test][data["y"]].values

    def prepare(frame):
        frame = frame.copy()
        for c in data["num"]:
            frame[c] = pd.to_numeric(frame[c], errors="coerce")
        for c in data["cats"]:
            frame[c] = frame[c].astype(str).str.strip().replace(list(NA), np.nan)
        return frame

    if data["task"] == "reg":
        y_test = pd.to_numeric(pd.Series(y_test), errors="coerce").values
    else:
        y_test = np.array([str(v) for v in y_test])

    def fit_predict(version):
        source, keep = frames[version]
        part = source.iloc[train].reset_index(drop=True)[keep]
        x_train, y_train = prepare(part[data["feat"]]), part[data["y"]].values
        steps = [("n", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), data["num"])] if data["num"] else []
        if data["cats"]:
            steps.append(("c", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                             OneHotEncoder(handle_unknown="ignore", max_categories=30, sparse_output=False)), data["cats"]))
        encoder = ColumnTransformer(steps)
        encoded = encoder.fit_transform(x_train)
        model = models(data["task"], seed)[model_name]
        if data["task"] == "reg":
            y_train = pd.to_numeric(pd.Series(y_train), errors="coerce")
            ok = y_train.notna().values
            model.fit(encoded[ok], y_train[ok])
        else:
            model.fit(encoded, np.array([str(v) for v in y_train]))
        return model.predict(encoder.transform(prepare(x_test)))

    def state(version):
        keep = frames[version][1]
        kinds = data["T"][train][keep]
        labels = data["yE"][train][keep]
        if version == "repair":
            kinds, labels = np.zeros_like(kinds), np.zeros_like(labels)
        return dict(rate_missing=float((kinds == 1).any(axis=1).mean()), rate_domain=float((kinds == 2).any(axis=1).mean()),
                    rate_value=float((kinds == 3).any(axis=1).mean()), rate_label=float(labels.mean()), n_train=int(keep.sum()))

    return fit_predict, {v: state(v) for v in frames}, y_test


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "tables"), help="folder with one subfolder per table")
    ap.add_argument("--meta", default=os.path.join(ROOT, "data", "tables", "table_meta.json"))
    ap.add_argument("--table", default="hospitals")
    ap.add_argument("--audit", type=int, default=200, help="records in the audit sample")
    ap.add_argument("--validation", type=int, default=100, help="validation records of Algorithm 3")
    ap.add_argument("--model", default="knn", choices=("linear", "rf", "gb", "knn", "mlp"))
    ap.add_argument("--threshold", type=float, default=0.9, help="threshold of each claim in the judgement")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-task", action="store_true", help="skip the model training")
    args = ap.parse_args()

    meta = json.load(open(args.meta))[args.table]
    data = load(args.root, args.table, meta)
    wrong = data["E"].any(axis=1) | data["yE"]
    kinds = data["T"]
    rate_missing, rate_domain, rate_value = [float((kinds == t).any(axis=1).mean()) for t in (1, 2, 3)]
    score = data_quality_score(rate_missing, rate_domain, rate_value)

    print(f"Table {args.table}, {len(wrong)} records, {len(data['feat'])} features, task {data['task']}, target {data['y']}")
    print(f"  {'wrong cells':38s} {100 * float((kinds > 0).mean()):.1f}%")
    print(f"  {'records with an error':38s} {100 * float(wrong.mean()):.1f}%")
    print(f"  {'records with a missing value':38s} {100 * rate_missing:.1f}%")
    print(f"  {'records with an out-of-domain value':38s} {100 * rate_domain:.1f}%")
    print(f"  {'records with a wrong in-domain value':38s} {100 * rate_value:.1f}%")
    print(f"  {'records with a wrong label':38s} {100 * float(data['yE'].mean()):.1f}%")
    print(f"  {'data quality score of the standard':38s} {score:.1f}, passes {passes(score)}")

    audit = stratified_audit(len(wrong), min(args.audit, len(wrong)), seed=args.seed)
    k = int(wrong[audit].sum())
    wilson, jeffreys = wilson_interval(k, len(audit)), jeffreys_interval(k, len(audit))
    print(f"\nAudit of {len(audit)} records with the clean reference as truth")
    print(f"  {'error rate':38s} {k / len(audit):.3f}")
    print(f"  {'Wilson interval':38s} {wilson[0]:.3f} to {wilson[1]:.3f}")
    print(f"  {'Jeffreys interval':38s} {jeffreys[0]:.3f} to {jeffreys[1]:.3f}")
    print(f"  {'records for a half-width of 0.02':38s} {proportion_sample_size(max(k / len(audit), 0.01), 0.02)}")

    domain, fds = read_rules(os.path.join(args.root, args.table, "rules.txt"))
    frame = data["a"].drop(columns=[c for c in ("index",) if c in data["a"].columns])
    flags = rule_detector(frame, domain, fds)
    calibrated = calibrate_error_rate(flags, lambda i: wrong[i], min(args.audit, len(wrong)), strata=flags, seed=args.seed)
    print("\nRule detector calibrated by Algorithm 1")
    print(f"  {'flagged share':38s} {calibrated.p_hat:.3f}")
    print(f"  {'sensitivity, specificity on the audit':38s} {calibrated.sensitivity:.3f}, {calibrated.specificity:.3f}")
    print(f"  {'corrected error rate':38s} {calibrated.estimate:.3f}, {calibrated.low:.3f} to {calibrated.high:.3f}")
    print(f"  {'method':38s} {calibrated.method}")
    print(f"  {'error rate from the full reference':38s} {float(wrong.mean()):.3f}")

    metric, metric_name = (macro_f1, "Macro-F1") if data["task"] == "cls" else (r2, "R2")
    usage = UsageContext(task=f"predict {data['y']}", metric=metric_name, reference="raw table", model_class=(args.model,),
                         scene="clean test records", budget=f"audit of {len(audit)} records")
    claims = [
        QualityClaim(args.table, "record correctness", "share of records without an error", Estimate.full_scan(1 - float(wrong.mean())), AssessmentLevel.L3, usage),
        QualityClaim(args.table, "structural completeness", "share of records without a missing value", Estimate.full_scan(1 - rate_missing), AssessmentLevel.L3, usage),
        QualityClaim(args.table, "format conformity", "share of records without an out-of-domain value", Estimate.full_scan(1 - rate_domain), AssessmentLevel.L3, usage),
        QualityClaim(args.table, "content authenticity", "share of records without a wrong in-domain value", Estimate.full_scan(1 - rate_value), AssessmentLevel.L3, usage),
        QualityClaim(args.table, "record correctness", "corrected share from the rule detector", Estimate(1 - calibrated.estimate, 1 - calibrated.high, 1 - calibrated.low, "Algorithm 1"), AssessmentLevel.L3, usage),
    ]

    if not args.no_task:
        fit_predict, states, y_test = version_scores(data, args.seed, args.model)
        half = len(y_test) // 2
        result = select_version(list(states), fit_predict, y_test, metric, "raw", min(args.validation, half), seed=args.seed + 900)
        print(f"\nChoice of a training version by Algorithm 3, model {args.model}, {metric_name} on clean test records")
        for version in states:
            print(f"  {version:38s} validation {result.validation_scores[version]:.3f}, test {result.test_scores[version]:.3f}")
        print(f"  {'task test selects':38s} {result.selected}, gain {result.gain:+.3f} [{result.gain_low:+.3f}, {result.gain_high:+.3f}], precision {result.precision:.3f} ({result.precision_method})")
        chosen = static_rule(states)
        best = hindsight_best(result.test_scores)
        print(f"  {'static rule selects':38s} {chosen}, gain {result.test_scores[chosen] - result.test_scores['raw']:+.3f}")
        print(f"  {'best in hindsight':38s} {best}, gain {result.test_scores[best] - result.test_scores['raw']:+.3f}")
        claims.append(QualityClaim(args.table, "model fit", f"{metric_name} of the selected version", Estimate(result.test_scores[result.selected], method="test part"), AssessmentLevel.L4, usage))
        if result.selected != "raw":
            claims.append(QualityClaim(args.table, "data change", f"gain of {result.selected} over the raw table",
                                       Estimate(result.gain, result.gain_low, result.gain_high, "paired bootstrap"), AssessmentLevel.L5, usage))

    print("\nQuality claims")
    for claim in claims:
        print(f"  {claim}")

    admission = [Requirement(claims[1], AssessmentLevel.L3, args.threshold)]
    others = [Requirement(claims[2], AssessmentLevel.L3, args.threshold), Requirement(claims[3], AssessmentLevel.L3, args.threshold), Requirement(claims[0], AssessmentLevel.L3, args.threshold)]
    verdict = judge(admission, others)
    print(f"\nJudgement by Algorithm 5 with threshold {args.threshold}")
    print(f"  {'passes':38s} {verdict.passed}")
    if verdict.failed_admission is not None:
        print(f"  {'failed admission condition':38s} {verdict.failed_admission.claim.measure}")
    else:
        print(f"  {'score of the three other claims':38s} {verdict.score:.1f}, range {verdict.score_low:.1f} to {verdict.score_high:.1f}")
        for record in verdict.records:
            if not record.ok:
                print(f"  {'below threshold':38s} {record.requirement.claim.measure}")


if __name__ == "__main__":
    main()
