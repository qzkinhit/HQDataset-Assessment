#!/usr/bin/env python3
"""Assessment levels of the requirement items of the assessment documents, from data/requirements/items.jsonl.
Computes the counts of Table 4, the other numbers of Section 3.2 and the agreement of the two coding rounds. Prints plain text, or JSON with --json.
Columns of Table 4: Items, L1 to L5 (items per given level), Sampling, Threshold and Scored (items that set a sampling design, give a pass condition
and enter a composite score), N>G (items whose needed level is above the given level). docs/standards.md describes the items and the coding rules.
Usage: requirement_levels.py [--json]"""
import argparse, collections, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
ITEMS = os.path.join(HERE, "..", "data", "requirements", "items.jsonl")
LEVELS = ["L1", "L2", "L3", "L4", "L5"]
CODES = ("given", "needed", "form")
COLUMNS = ["Items"] + LEVELS + ["Sampling", "Threshold", "Scored", "N>G"]
DUPLICATES = {"T04": "G48562", "T03": "G48561", "T02": "G48563"}     # technical document and the national standard draft with nearly the same content
INTERNATIONAL = ("I25012", "I25024", "NAI100", "NAI600")
ROWS = [("G48562 (T04)", ["G48562"]), ("G48560", ["G48560"]), ("G48561 (T03)", ["G48561"]), ("G48563 (T02)", ["G48563"]), ("DPQ", ["DPQ"]), ("T01", ["T01"]),
        ("ISO/IEC 25012, 25024", ["I25012", "I25024"]), ("NIST AI 100-1", ["NAI100"]), ("NIST AI 600-1", ["NAI600"])]
join = ", ".join


def rank(level):
    """1 to 5 for L1 to L5, 0 for an item that makes no judgment."""
    return LEVELS.index(level) + 1 if level in LEVELS else 0


def row(items):
    """One row of Table 4."""
    return ([len(items)] + [sum(r["given"] == l for r in items) for l in LEVELS] + [sum(r[k] for r in items) for k in ("sampling", "threshold", "aggregation")]
            + [sum(rank(r["needed"]) > rank(r["given"]) > 0 for r in items)])


def agreement(pairs):
    """Share of items with the same code in both rounds, Cohen's kappa and the number of items with different codes."""
    n = len(pairs); observed = sum(a == b for a, b in pairs) / n
    first, second = collections.Counter(a for a, _ in pairs), collections.Counter(b for _, b in pairs)
    expected = sum(first[k] * second[k] for k in first) / n ** 2
    return dict(agreement=observed, kappa=(observed - expected) / (1 - expected) if expected < 1 else 1.0, disagreements=sum(a != b for a, b in pairs))


def compute(path=ITEMS):
    """Table 4, the numbers of Section 3.2 and the agreement of the two rounds, as a dictionary."""
    items = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    by = collections.defaultdict(list)
    for r in items:
        by[r["file"]].append(r)
    files = [f for f in by if f not in DUPLICATES]                      # the 10 documents of Table 4
    assert sorted(f for _, fs in ROWS for f in fs) == sorted(files)
    kept = [r for f in files for r in by[f]]

    def where(test):
        """Documents of Table 4 with at least one item that passes the test."""
        return [f for f in files if any(test(r) for r in by[f])]
    table = {name: row([r for f in fs for r in by[f]]) for name, fs in ROWS}
    table["Total"] = row(kept)
    table["Share (%)"] = [round(100 * v / len(kept), 1) for v in table["Total"]]
    pairs = {}
    for tech, draft in DUPLICATES.items():
        a, b = ({r["name"]: r for r in by[f]} for f in (draft, tech))                # items of the two versions matched by name
        assert len(a) == len(by[draft]) and len(b) == len(by[tech])
        pairs[f"{draft}, {tech}"] = dict(items=[len(a), len(b)], only_in_draft=[a[k]["id"] for k in a if k not in b], only_in_technical_document=[b[k]["id"] for k in b if k not in a],
                                         coded_differently=[a[k]["id"] for k in a if k in b and any(a[k][c] != b[k][c] for c in CODES)])
    l1 = [r for r in kept if r["given"] == "L1"]
    l2 = [r for r in kept if r["given"] == "L2"]
    with_l2 = where(lambda r: r["given"] == "L2")
    numbers = dict(
        documents=len(by), domestic=sum(f not in INTERNATIONAL for f in by), international=sum(f in INTERNATIONAL for f in by), items=len(items),
        documents_counted=len(files), items_counted=len(kept),
        files_with_given_L2=with_l2, files_with_given_L2_without_sampling=[f for f in with_l2 if not any(r["sampling"] for r in by[f])],
        given_L2=len(l2), given_L2_without_sampling=sum(not r["sampling"] for r in l2), items_with_sampling=[r["id"] for r in kept if r["sampling"]],
        scored=sum(r["aggregation"] for r in kept), scored_with_sampling=sum(r["aggregation"] and r["sampling"] for r in kept),
        files_with_scored=where(lambda r: r["aggregation"]), threshold_by_file={f: sum(r["threshold"] for r in by[f]) for f in where(lambda r: r["threshold"])},
        files_with_given_L1=where(lambda r: r["given"] == "L1"), given_L1=len(l1), given_L1_by_needed={l: sum(r["needed"] == l for r in l1) for l in LEVELS},
        given_L1_needed_higher=sum(rank(r["needed"]) > 1 for r in l1),
        files_where_most_L1_need_higher=[f for f in where(lambda r: r["given"] == "L1")
                                         if 2 * sum(r["given"] == "L1" and rank(r["needed"]) > 1 for r in by[f]) > sum(r["given"] == "L1" for r in by[f])],
        given_L4=[r["id"] for r in kept if r["given"] == "L4"], given_L5=sum(r["given"] == "L5" for r in kept),
        reading_a_version_with_one_change=sum("controlled_variant" in r["inputs"] for r in kept),
        needed_L4_or_L5=sum(rank(r["needed"]) >= 4 for r in kept), files_with_needed_L4_or_L5=where(lambda r: rank(r["needed"]) >= 4))
    groups = dict(all=items, domestic=[r for r in items if r["file"] not in INTERNATIONAL], international=[r for r in items if r["file"] in INTERNATIONAL])
    rounds = {g: dict(items=len(rs), **{c: agreement([(r["round1"][c], r["round2"][c]) for r in rs]) for c in CODES}) for g, rs in groups.items()}
    rounds["items_with_disagreement"] = sum(r["round1"] != r["round2"] for r in items)
    # items of given level L1 whose needed level is L1 in one round and L2 in the other, the most frequent disagreement of the needed level
    rounds["needed_L1_against_L2_on_given_L1"] = sum(r["given"] == "L1" and {r["round1"]["needed"], r["round2"]["needed"]} == {"L1", "L2"} for r in items)
    return dict(items_per_document={f: len(rs) for f, rs in by.items()}, pairs=pairs, table4=dict(columns=COLUMNS, rows=table), section_3_2=numbers, agreement=rounds)


def report(s):
    x, a = s["section_3_2"], s["agreement"]
    width = [max(len(c), 5) + 2 for c in COLUMNS]
    lines = [f"{x['items']} items in {x['documents']} documents ({x['domestic']} domestic, {x['international']} international), "
             f"{x['items_counted']} items in the {x['documents_counted']} documents without {join(DUPLICATES)}",
             "Items per document: " + join(f"{f} {n}" for f, n in s["items_per_document"].items())]
    for pair, p in s["pairs"].items():
        lines.append(f"{pair}: {p['items'][0]} and {p['items'][1]} items, only in the draft: {join(p['only_in_draft']) or 'none'}, "
                     f"only in the technical document: {join(p['only_in_technical_document']) or 'none'}, coded differently: {join(p['coded_differently']) or 'none'}")
    lines += ["", "Table 4", f"{'Document':22s}" + "".join(f"{c:>{w}s}" for c, w in zip(COLUMNS, width))]
    for name, values in s["table4"]["rows"].items():
        lines.append(f"{name:22s}" + "".join(f"{v:>{w}.1f}" if isinstance(v, float) else f"{v:>{w}d}" for v, w in zip(values, width)))
    needed = x["given_L1_by_needed"]
    lines += ["", f"Section 3.2, over the {x['documents_counted']} documents of Table 4",
              f"Documents with items of given level L2: {len(x['files_with_given_L2'])}, without any item that sets sampling: "
              f"{len(x['files_with_given_L2_without_sampling'])} ({join(x['files_with_given_L2_without_sampling'])})",
              f"Items of given level L2 without a sampling plan, sample size or interval: {x['given_L2_without_sampling']} of {x['given_L2']}",
              f"Items that set sampling: {join(x['items_with_sampling'])}",
              f"Scored items: {x['scored']} in {join(x['files_with_scored'])}, with sampling: {x['scored_with_sampling']}",
              "Items with a threshold: " + join(f"{f} {n}" for f, n in x["threshold_by_file"].items()),
              f"Documents with items of given level L1: {len(x['files_with_given_L1'])}, items of given level L1: {x['given_L1']}",
              f"Items of given level L1 with needed level L1: {needed['L1']}, L2: {needed['L2']}, L3 to L5: {needed['L3'] + needed['L4'] + needed['L5']}",
              f"Items of given level L1 with a higher needed level: {x['given_L1_needed_higher']}, the majority of the L1 items in "
              f"{len(x['files_where_most_L1_need_higher'])} documents ({join(x['files_where_most_L1_need_higher'])})",
              f"Items of given level L5: {x['given_L5']}, items that read a version with one aspect of the data changed: {x['reading_a_version_with_one_change']}",
              f"Items of given level L4: {join(x['given_L4'])}",
              f"Items of needed level L4 or L5: {x['needed_L4_or_L5']} in {len(x['files_with_needed_L4_or_L5'])} documents ({join(x['files_with_needed_L4_or_L5'])})",
              "", "Agreement of the two coding rounds", f"{'Items':22s}{'Code':>8s}{'Agreement (%)':>15s}{'Kappa':>8s}{'Disagreements':>15s}"]
    for g in ("all", "domestic", "international"):
        for c in CODES:
            v = a[g][c]
            lines.append(f"{g + ' (' + str(a[g]['items']) + ')':22s}{c:>8s}{100 * v['agreement']:>15.1f}{v['kappa']:>8.2f}{v['disagreements']:>15d}")
    lines += [f"Items with a disagreement in at least one code: {a['items_with_disagreement']}",
              f"Needed level: {a['needed_L1_against_L2_on_given_L1']} of the {a['all']['needed']['disagreements']} disagreements are L1 against L2 on items of given level L1"]
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); ap.add_argument("--json", action="store_true", help="print JSON")
    s = compute()
    print(json.dumps(s, indent=1) if ap.parse_args().json else report(s))
