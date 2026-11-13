#!/usr/bin/env python3
"""Sets the widths of the single panels of each themed result figure in part3.tex, so that the four panels of a row share one height and fill the text width."""
import re, subprocess, sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); PLOTS = os.path.join(HERE, "plots"); TEX = sys.argv[1]
def size(stem):
    out = subprocess.run(["pdfinfo", os.path.join(PLOTS, stem + ".pdf")], capture_output=True, text=True).stdout
    w, h = map(float, re.search(r"Page size:\s+([\d.]+) x ([\d.]+)", out).groups()); return w, h
t = open(TEX, encoding="utf-8").read()
for g in ("version", "protocol", "time", "newdata"):
    S = [size(f"res_{g}_{l}") for l in "abcd"]; H = max(h for _, h in S)
    ws = [w * H / h for w, h in S]; tot = sum(ws)
    for l, w in zip("abcd", ws):
        t, n = re.subn(r"\\includegraphics\[width=[\d.]+\\textwidth\]\{figures/res_%s_%s\.pdf\}" % (g, l), lambda m: "\\includegraphics[width=%.4f\\textwidth]{figures/res_%s_%s.pdf}" % (w / tot - .0005, g, l), t)
        assert n == 1, (g, l)
open(TEX, "w", encoding="utf-8").write(t); print("rows fitted")
