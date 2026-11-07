# Tables with a dirty version and a clean reference

Each folder holds `dirty_index.csv`, `clean_index.csv` and the rule library `rules.txt`. `table_meta.json` gives the task, the target and the categorical columns of each table. The tables come from the REIN benchmark (Abdelaal et al., 2023) and the UniClean benchmark (Ding et al., 2025), and [docs/datasets.md](../../docs/datasets.md) lists their sources and errors. A rule library has the sections DOMAIN, REGEX, FD, CFD and DC. A line that starts with `#` is a comment.

The two CSV files of `mercedes` are not in the repository, because the competition rules limit the use of the data to the competition. They come from the REIN benchmark.
