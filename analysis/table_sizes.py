#!/usr/bin/env python3
"""Number of records of each table. Usage: table_sizes.py TABLES_DIR OUT_JSON"""
import json, os, sys
root = sys.argv[1]; meta = json.load(open(os.path.join(root, "table_meta.json")))
sizes = {name: sum(1 for _ in open(os.path.join(root, name, "dirty_index.csv"), encoding="utf-8")) - 1 for name in meta}
json.dump(sizes, open(sys.argv[2], "w")); print(sizes)
