# Release assets

Files that are too large for the repository are attached to the GitHub Release of this repository as archives of at most 1.8 GiB each. `scripts/package_release.py` builds them on the machine that holds the files.

## What each archive holds

Paths inside an archive follow the layout of `scripts/download_data.sh`, so an archive unpacked into `downloads/` gives the files that the scripts read. The sizes are those of the packed files. A group larger than the limit comes as numbered parts.

| Archive | Content | Size | Origin |
|---|---|---|---|
| `hqd-data-rieth` | `rieth/TEP_*.RData`, the four files of the large TEP | 1.3 GiB | Downloaded |
| `hqd-data-taxi` | `tlc/yellow_tripdata_*.parquet`, 72 months | 3.8 GiB | Downloaded |
| `hqd-data-cifar` | `cifar-10-python.tar.gz`, `cifar_n/CIFAR-10_human.pt`, `cifar_n/CIFAR-100_human.pt`, the arrays `cifar10_real.npz` and `cifar100_real.npz`, the features `cifar10_clip.npz` | 0.7 GiB | Downloaded and written by `study/prep_vision.py`, `study/embed_cifar.py` |
| `hqd-data-coco` | `coco/annotations.zip`, `coco/val2017.zip`, the features `coco_clip.npz` | 1.0 GiB | Downloaded and written by `study/embed_coco.py` |
| `hqd-data-corpus` | `corpus/c4-00000.json.gz` to `c4-00002.json.gz`, `corpus/wiki-00001.parquet`, `corpus/tok/` | 1.2 GiB | Downloaded |
| `hqd-data-corpus-arrays` | `corpus/c4.npz`, `c4_group.npy`, `c4_ppl.npy`, `c4_llm_score.npy`, `wiki.npz`, `dclm_pool.npz`, `dclm_baseline.npz` | 1.1 GiB | Written by `corpus/prep_corpus.py`, `corpus/run_corpus.py`, `llm/judge_corpus.py`, `native/prep_dclm.py` |
| `hqd-data-textpool` | `processed/qpool_train.jsonl`, `processed/qpool_heldout.jsonl`, `processed/pool_manifest.json`, `text_ppl.npz` | 20 MiB | Written by `study/build_text_pool.py`, `study/run_text.py` |
| `hqd-data-synthetic` | `edm_200k.npz`, `ddpm_head.bin`, `ddpm_tail.bin`, `cifake_train.zip`, `cifar10.1_v6_data.npy`, `cifar10.1_v6_labels.npy`, the arrays `synth_edm.npz`, `synth_ddpm.npz`, `synth_cifake.npz`, `cifar10_1.npz` | 1.7 GiB | Downloaded and written by `native/prep_native.py` |
| `hqd-data-labelerrors` | `label_errors/NAME_cv_pred.npy`, `NAME_orig.npy`, `NAME_mturk.json`, `imdb_index.json`, `imagenet_index.json`, `class_labels_indices.csv` | 0.2 GiB | Downloaded |
| `hqd-data-dclm` | `dclm_pool/CC_shard_*.jsonl.zst`, `dclm_base/shard_*.jsonl.zst`, three shards each | 1.0 GiB | Downloaded |
| `hqd-data-cleanml` | `cleanml/CleanML-datasets-2020.zip` | 0.2 GiB | Downloaded |
| `hqd-data-cleanml-versions` | `cleanml/repo/data/`, the cleaned versions of the 13 tables | 3.2 GiB | Written by `native/clean_cleanml.py` |
| `hqd-data-datacomp` | `datacomp/0020f0cbd157d470aa56bea08e304b90.parquet`, `datacomp/b32_part.bin` | 1.1 GiB | Downloaded |
| `hqd-data-mirflickr` | `mirflickr25k.zip`, `mirflickr25k_annotations_v080.zip`, the features `mirflickr_clip.npz` | 3.0 GiB | Downloaded and written by `native/embed_mirflickr.py` |
| `hqd-data-wilds` | `wilds/civilcomments_raw.parquet`, the Camelyon17 subset `camelyon_x.npy` and `camelyon_meta.npz` | 2.7 GiB | Downloaded and written by `native/prep_camelyon.py` |
| `hqd-data-llmtexts` | `llm_texts/imdb.tgz`, `20news-bydate.tar.gz`, `amazon5core.tar.gz-partaa`, `amazon5core.tar.gz-partab` | 2.9 GiB | Downloaded |
| `hqd-data-agnews` | `agnews/train.csv`, `agnews/test.csv` | 30 MiB | Downloaded |
| `hqd-data-predictions` | `preds/*.npy`, saved test predictions of the image runs | 10 MiB | Written by `study/run_vision.py`, `study/run_synth.py` |
| `hqd-logs-LABEL` | `logs/TRACK/JOB.log`, the rewritten logs of the runs, per track, and the logs of the job pools and the preparation steps | 40 MiB packed | Written by the runs |

[datasets.md](datasets.md) gives the official source and the license as stated by the source for each dataset. The terms of a source also hold for its copy in an archive. The 11 tables, classic TEP, the ciFAIR lists and the reviewer votes are in `data/` and in no archive, apart from the copy of the votes in `hqd-data-labelerrors`.

The logs folder has one subfolder per track, `vision`, `text`, `multimodal`, `corpus`, `synthetic`, `camelyon`, `civilcomments`, `cleanml` and `llm`, named by the job id of `study/make_jobs.py` and its siblings, plus `pools` and `other`. A job that ran more than once has numbered logs in the order of the runs, `JOB.1.log` and `JOB.2.log`. Logs without content are left out.

## Next to the archives

| File | Content |
|---|---|
| `SHA256SUMS-LABEL.txt` | SHA-256 of every asset of the batch |
| `MANIFEST-LABEL.json` | For each archive its parts, its members with size and SHA-256, and the command that unpacks it |
| `scrub_report-LABEL.json` | Number of replacements per rewriting rule, without the replaced text |

## How to fetch and unpack

```bash
gh release download --repo OWNER/REPOSITORY --pattern 'hqd-data-rieth*' --pattern 'SHA256SUMS*'
sha256sum --check --ignore-missing SHA256SUMS-*.txt
mkdir -p downloads
cat hqd-data-rieth.tar* | tar -x -C downloads
```

The parts of an archive are named `NAME.tar.part-001`, `NAME.tar.part-002` and so on, and `cat` joins them in this order. An archive that fits one asset is named `NAME.tar`. The log archive is unpacked with `cat hqd-logs-LABEL.tar.gz* | tar -xz`.

## How the archives are built

```bash
python scripts/package_release.py --list                                                 # the groups and their files
python scripts/package_release.py --data DATA_DIR --logs LOG_DIR --out OUT_DIR --dry-run # sizes, missing files, replacements
python scripts/package_release.py --data DATA_DIR --logs LOG_DIR --out OUT_DIR --private TERMS_FILE --label batch1
gh release upload TAG OUT_DIR/hqd-* OUT_DIR/SHA256SUMS-* OUT_DIR/MANIFEST-*
```

The script searches each `--data` folder for the files of a group by the end of their path, so the files may sit in deeper folders. The first folder that holds a file wins. `--only` and `--skip` select groups by name, and the name `logs` stands for the logs. `--exclude` leaves out a folder by name, for example one with logs of runs that were not scored. A batch on another machine uses its own `--label`, so that the checksum files stay apart.

Before a log enters the archive, the script applies these rules in this order.

| Rule | What it replaces | Placeholder |
|---|---|---|
| `python_env` | Paths of conda and virtual environments | `$PYTHON_ENV` |
| `model_cache` | Paths of the model cache | `$HF_HOME` |
| `work_folder` | The folders given by `--data` and `--logs` | `$WORK` |
| `attempt_folder` | Dated folders inside a path, and the folders named by `--drop-folder` | Removed |
| `home` | Home folders with the user name | `$HOME` |
| `storage` | Mount points of storage with the next folder | `$STORE` |
| `private_term` | The current user name, the host name, and the terms of `--private` | `<user>`, `<host>` or the placeholder in the file |
| `address` | Local proxy addresses and network addresses with a port | `<proxy>`, `<host>` |
| `email`, `token` | Mail addresses and access tokens | `<email>`, `<token>` |

Archive paths use the job id or a cleaned file name, without the folders of the machine. Tar headers hold no owner and no time. A rewritten log that still holds a private term or a home path stops the batch with exit code 3, and no log archive is kept.
