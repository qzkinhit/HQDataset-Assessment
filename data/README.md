# Data in the repository

This folder holds the small files of the experiments. [docs/datasets.md](../docs/datasets.md) gives the official source, the license as stated by the source, the use and the download command of every dataset, and [docs/release_assets.md](../docs/release_assets.md) describes the release archives with the large files.

| Folder | Content | Size | Source |
|---|---|---|---|
| `tables/` | 11 tables with a rule library and `table_meta.json`, the dirty version and the clean reference of 10 of them | 63 MB | REIN and UniClean benchmarks |
| `tep/` | 44 files of classic TEP and the license text of the distribution | 25 MB | Braatz group |
| `cifair/` | `duplicates_cifar10.csv` and `duplicates_cifar100.csv`, near duplicates of the CIFAR test images | 40 KB | Barz and Denzler |
| `label_errors/` | Reviewer votes `NAME_mturk.json` of ten test sets, `imdb_index.json` and `imagenet_index.json` | 8 MB | Northcutt et al. |
| `requirements/` | `items.jsonl`, the 479 quality requirement items of 13 assessment documents with their assessment levels and the codes of two coding rounds, described in [docs/standards.md](../docs/standards.md) | 0.3 MB | Section 6.1 of the survey |

The scripts read the two last folders from the download folder. `scripts/download_data.sh DIR cifair labelerrors` writes the same files there, together with the label arrays that are too large for the repository.

The competition rules of Mercedes limit the use of the data to the competition, so `tables/mercedes/` holds only `rules.txt`. Its `dirty_index.csv` and `clean_index.csv` come from the REIN benchmark and go into that folder.
