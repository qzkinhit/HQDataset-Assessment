#!/usr/bin/env python3
"""Work units of the table repair baseline on the GPU side. The model reads a dirty row and replies with the corrected values as JSON, the replies are stored per block of rows. The prompts come from prep_tables.py, this side holds no clean table."""
import json, os, re
from qwen import chat, generate, ids, turn
SYSTEM = "You repair errors in the rows of the data table {ds}. You receive one row as a JSON object with the column names as keys. Reply with a JSON object that holds only the cells whose value is wrong, each mapped to its corrected value. Reply with {{}} when the row has no error. Reply with JSON only."
ROWS_PER_UNIT = 128; ORDER = ("flights", "beers", "hospitals", "nasa", "breast_cancer", "soccer_full", "adult", "bike", "smartfactory", "soilmoisture"); LAST = ("mercedes",)
_plans = {}
def plan(root, ds):
    if ds not in _plans: _plans[ds] = json.load(open(f"{root}/prompts/{ds}.json"))
    return _plans[ds]
def units(root, tables, test=None):
    for ds in tables:
        if os.path.exists(f"{root}/prompts/{ds}.json"):
            for b in range(0, min(test or 10 ** 9, len(plan(root, ds)["rows"])), ROWS_PER_UNIT): yield ds, b
def dump(d): return json.dumps(d, ensure_ascii=False)
def parse(text, cols):
    """The JSON object of a reply, a reply that was cut off keeps its complete pairs. Returns the cells and whether the reply parsed as a whole."""
    s = text[text.find("{"):] if "{" in text else ""
    try: d = json.loads(s[:s.rfind("}") + 1]); ok = isinstance(d, dict); d = d if ok else {}
    except ValueError: d = dict(re.findall(r'"([^"]+)"\s*:\s*"([^"]*)"', s)); ok = False
    return {k: str(v).strip() for k, v in d.items() if k in cols and v is not None and not isinstance(v, (dict, list))}, ok
def run(model, tok, root, ds, start, size=ROWS_PER_UNIT):
    """Replies for the rows of one unit, one line per row with the cells to change and the raw text."""
    P = plan(root, ds); pos = {c: j for j, c in enumerate(P["columns"])}; rows, cells = P["rows"][start:start + size], P["cells"][start:start + size]; out = [dict(row=r, fix={}, bad=0, raw=[]) for r in rows]
    for block in P["blocks"]:
        part = lambda v: {c: v[pos[c]] for c in block}; turns = [(dump(part(s["cells"])), dump({c: x for c, x in s["fix"].items() if c in block})) for s in P["shots"]]
        prefix = ids(tok, chat(SYSTEM.format(ds=ds), turns)); sufs = [ids(tok, turn(dump(part(v)))) for v in cells]; replies = generate(model, tok, prefix, sufs, min(800, max(map(len, sufs))))
        for o, r in zip(out, replies):
            text = tok.decode(r, skip_special_tokens=True); fix, ok = parse(text, set(block)); o["fix"].update(fix); o["bad"] += not ok; o["raw"].append(text)
    return out
