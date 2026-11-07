#!/usr/bin/env python3
"""Signal traces of Figure 1. Standardizes one process variable of the normal test file and of the test files of faults 1 and 3.
The variable is the one with the largest mean shift under fault 1, measured after row 300. The script also records the largest shift of any variable under fault 3.
Usage: make_traces.py DATA_DIR OUT_JSON"""
import json, os, sys
import numpy as np
D = sys.argv[1]
def L(fn):
    a = np.loadtxt(os.path.join(D, fn))
    return a.T if a.shape[1] != 52 and a.shape[0] == 52 else a
n = L("d00_te.dat"); f1 = L("d01_te.dat"); f3 = L("d03_te.dat")
mu, sd = n.mean(0), n.std(0) + 1e-9
shift = np.abs((f1[300:].mean(0) - mu) / sd); v = int(np.argmax(shift)); print("max shift var", v, shift[v])
shift3 = np.abs((f3[300:].mean(0) - mu) / sd); print("fault 3 largest shift", shift3.max(), int(np.argmax(shift3)))
out = dict(var=v, normal=((n[:, v] - mu[v]) / sd[v]).round(3).tolist(), fault1=((f1[:, v] - mu[v]) / sd[v]).round(3).tolist(), fault3=((f3[:, v] - mu[v]) / sd[v]).round(3).tolist(), fault3_max_shift=float(shift3.max()), fault1_max_shift=float(shift[v]))
json.dump(out, open(sys.argv[2], "w"))
