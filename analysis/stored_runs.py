#!/usr/bin/env python3
"""Number of scored runs in each result file, as a Markdown table. A row with an error field is not a scored run, for example a deletion that leaves fewer than 30 records.
With --update FILE the script rewrites the table between the markers STORED RUNS BEGIN and STORED RUNS END of that file.
Usage: stored_runs.py [--results DIR] [--update FILE]"""
import argparse, json, os, re
HERE = os.path.dirname(os.path.abspath(__file__))

def rows_of(path):
    """The runs of a result file, from a JSONL file or from the key runs of a JSON file."""
    if path.endswith(".jsonl"):
        return [json.loads(line) for line in open(path) if line.strip()]
    data = json.load(open(path))
    if "runs" in data:
        return data["runs"]
    return [r for part in data.values() if isinstance(part, dict) for r in part.get("runs", [])]

def ranges(values):
    """Seeds as text, 0 to 19 for a closed range."""
    values = sorted(values)
    if not values:
        return ""
    if values == list(range(values[0], values[-1] + 1)) and len(values) > 2:
        return f"{values[0]} to {values[-1]}"
    return ", ".join(map(str, values))

def table(results):
    lines = ["| File | Scored runs | Seeds |", "|---|---|---|"]
    other = {"results_sizes_native.json", "results_fidelity.json", "results_audioset_labels.json", "results_llm_cells.json", "results_detectors.jsonl"}     # sizes and summaries, one row is not one run
    names = sorted(n for n in os.listdir(results) if n.startswith(("results_", "case_tep")) and n.endswith((".json", ".jsonl")) and n not in other)
    for name in names:
        rows = [r for r in rows_of(os.path.join(results, name)) if "error" not in r]
        seeds = {r["seed"] for r in rows if isinstance(r.get("seed"), int)}
        if rows:
            lines.append(f"| `{name}` | {len(rows)} | {ranges(seeds)} |")
    return "\n".join(lines)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--results", default=os.path.join(HERE, "..", "results")); ap.add_argument("--update"); a = ap.parse_args()
    text = table(a.results); print(text)
    if a.update:
        doc = open(a.update, encoding="utf-8").read()
        doc, n = re.subn(r"(<!-- STORED RUNS BEGIN -->\n).*?(<!-- STORED RUNS END -->)", lambda m: m.group(1) + text + "\n" + m.group(2), doc, flags=re.S)
        if n != 1:
            raise SystemExit(f"{a.update} has no markers")
        open(a.update, "w", encoding="utf-8").write(doc)
