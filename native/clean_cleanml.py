#!/usr/bin/env python3
"""Generates the cleaned versions of the CleanML benchmark (Li et al. 2021) with its own clean.py, training is not run. The tables keep the split of the archive, a table whose archive has no split gets one from CleanML's init.py with seed 1. The schema of the repository lists KDD only with injected mislabels, its Readme names the native errors, missing values and outliers, these are cleaned here without the HoloClean arm, whose output is not in the archive. Mislabel variants are injected and left out. Run from the repository root, with data/ beside the code."""
import argparse, os, sys, time
import numpy as np
sys.path.insert(0, os.getcwd())
KDD = dict(data_dir="KDD", error_types=["missing_values", "outliers"], label="is_exciting_20", ml_task="classification", class_imbalance=True, categorical_variables=["is_exciting_20"])
NAMES = ("Airbnb", "Citation", "Company", "Credit", "EEG", "KDD", "Marketing", "Movie", "Restaurant", "Sensor", "Titanic", "University", "USCensus")
def table(name):
    import utils
    return KDD if name == "KDD" else utils.get_dataset(name)
def methods(ds, err):
    import utils
    return ["dirty"] + [m for m in utils.get_error(err)["clean_methods"] if not (ds["data_dir"] == "KDD" and "holoclean" in m)]
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset"); A = ap.parse_args()
    import clean, init, utils
    ds = table(A.dataset); t0 = time.time(); np.random.seed(1)    # Isolation Forest of the benchmark draws from the global generator
    if not (os.path.exists(utils.get_dir(ds, "raw", "dirty_test.csv")) and os.path.exists(utils.get_dir(ds, "raw", "idx_test.csv"))): init.init(ds, seed=1)
    if A.dataset == "KDD":
        get = utils.get_error; utils.get_error = lambda n: dict(name=n, clean_methods={k: v for k, v in get(n)["clean_methods"].items() if "holoclean" not in k})
    clean.clean(ds); print(A.dataset, "finished", round(time.time() - t0), flush=True)
