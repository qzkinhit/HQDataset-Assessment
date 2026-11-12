#!/usr/bin/env bash
# Download the datasets of the experiments into the layout that the scripts read.
# Usage: scripts/download_data.sh [DIR] [DATASET ...]
# DATASET is any of rieth, taxi, cifar, cifair, coco, corpus, textpool, synthetic, labelerrors, dclm, cleanml, datacomp, mirflickr, wilds, llmtexts, agnews, release. The default is all of them.
# Ten of the 11 tables and classic TEP are in data/, the CSV files of Mercedes come from the REIN benchmark. docs/datasets.md gives the source and the license of each dataset.
# DRY_RUN=1 prints the commands without running them. A file that already exists is kept.
set -euo pipefail
ROOT="${1:-downloads}"
if [ "$#" -gt 0 ]; then shift; fi
if [ "$#" -gt 0 ]; then SETS="$*"; else SETS="rieth taxi cifar cifair coco corpus textpool synthetic labelerrors dclm cleanml datacomp mirflickr wilds llmtexts agnews release"; fi
HERE="$(cd "$(dirname "$0")" && pwd)"
RANGES="$HERE/../native/fetch_ranges.py"

run() {
  if [ "${DRY_RUN:-0}" = "1" ]; then echo "$*"; else "$@"; fi
}

fetch() {
  if [ "${DRY_RUN:-0}" = "1" ]; then echo "$2 <- $1"; return; fi
  mkdir -p "$(dirname "$2")"
  [ -s "$2" ] || curl -sS -L --retry 8 --connect-timeout 60 -C - -o "$2" "$1"
}

# A byte range of a published archive, resumable: fetch_range URL OUT START END
fetch_range() {
  if [ "${DRY_RUN:-0}" = "1" ]; then echo "$2 <- $1 bytes $3 to $4"; return; fi
  mkdir -p "$(dirname "$2")"
  python3 "$RANGES" "$1" "$2" "$3" "$4"
}

HF=https://huggingface.co/datasets

for name in $SETS; do
  case "$name" in
    rieth)
      # Rieth et al., doi:10.7910/DVN/6C3JR1, four R data files, about 1.4 GB
      fetch https://dataverse.harvard.edu/api/access/datafile/3031241 "$ROOT/rieth/TEP_FaultFree_Training.RData"
      fetch https://dataverse.harvard.edu/api/access/datafile/3031240 "$ROOT/rieth/TEP_FaultFree_Testing.RData"
      fetch https://dataverse.harvard.edu/api/access/datafile/3031242 "$ROOT/rieth/TEP_Faulty_Training.RData"
      fetch https://dataverse.harvard.edu/api/access/datafile/3031243 "$ROOT/rieth/TEP_Faulty_Testing.RData"
      ;;
    taxi)
      # NYC TLC yellow taxi trip records, 72 monthly Parquet files of 2019 to 2024
      for year in 2019 2020 2021 2022 2023 2024; do
        for month in 01 02 03 04 05 06 07 08 09 10 11 12; do
          fetch "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_$year-$month.parquet" "$ROOT/tlc/yellow_tripdata_$year-$month.parquet"
        done
      done
      ;;
    cifar)
      # CIFAR-10 in the original batch order and the human labels of CIFAR-N. CIFAR-100 comes through the datasets package in study/prep_vision.py
      fetch https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz "$ROOT/cifar-10-python.tar.gz"
      fetch https://github.com/UCSC-REAL/cifar-10-100n/raw/main/data/CIFAR-10_human.pt "$ROOT/cifar_n/CIFAR-10_human.pt"
      fetch https://github.com/UCSC-REAL/cifar-10-100n/raw/main/data/CIFAR-100_human.pt "$ROOT/cifar_n/CIFAR-100_human.pt"
      ;;
    cifair)
      # near-duplicate lists of the CIFAR test sets, Barz and Denzler, also in data/cifair
      fetch https://raw.githubusercontent.com/cvjena/cifair/master/meta/duplicates_cifar10.csv "$ROOT/cifair/duplicates_cifar10.csv"
      fetch https://raw.githubusercontent.com/cvjena/cifair/master/meta/duplicates_cifar100.csv "$ROOT/cifair/duplicates_cifar100.csv"
      ;;
    coco)
      # COCO 2017 validation images and annotations
      fetch http://images.cocodataset.org/annotations/annotations_trainval2017.zip "$ROOT/coco/annotations.zip"
      fetch http://images.cocodataset.org/zips/val2017.zip "$ROOT/coco/val2017.zip"
      ;;
    corpus)
      # three C4 shards, one Wikipedia shard and the tokenizer of the small language model
      for i in 00000 00001 00002; do
        fetch "$HF/allenai/c4/resolve/main/en/c4-train.$i-of-01024.json.gz" "$ROOT/corpus/c4-$i.json.gz"
      done
      fetch "$HF/wikimedia/wikipedia/resolve/main/20231101.en/train-00001-of-00041.parquet" "$ROOT/corpus/wiki-00001.parquet"
      for f in tokenizer.json tokenizer_config.json special_tokens_map.json config.json; do
        fetch "https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/main/$f" "$ROOT/corpus/tok/$f"
      done
      ;;
    textpool)
      # the five-domain text pool, built from pinned revisions of five public datasets
      run python3 "$HERE/../study/build_text_pool.py" --out "$ROOT/processed"
      ;;
    synthetic)
      # published synthetic CIFAR-10 images and the test set CIFAR-10.1
      # the DDPM archive holds 1M images in 3 GB, its first 100000 images and its label member are fetched by byte range
      fetch https://huggingface.co/wzekai99/DM-Improves-AT/resolve/main/edm_data/cifar10_step20_vp/200k.npz "$ROOT/edm_200k.npz"
      fetch_range https://storage.googleapis.com/dm-adversarial-robustness/cifar10_ddpm.npz "$ROOT/ddpm_head.bin" 0 307300000
      fetch_range https://storage.googleapis.com/dm-adversarial-robustness/cifar10_ddpm.npz "$ROOT/ddpm_tail.bin" 3072000000 3076000594
      fetch "$HF/yanbax/CIFAKE_autotrain_compatible/resolve/main/train.zip" "$ROOT/cifake_train.zip"
      fetch https://raw.githubusercontent.com/modestyachts/CIFAR-10.1/master/datasets/cifar10.1_v6_data.npy "$ROOT/cifar10.1_v6_data.npy"
      fetch https://raw.githubusercontent.com/modestyachts/CIFAR-10.1/master/datasets/cifar10.1_v6_labels.npy "$ROOT/cifar10.1_v6_labels.npy"
      ;;
    labelerrors)
      # predictions, original labels and reviewer votes of Northcutt et al., the votes are also in data/label_errors
      BASE=https://raw.githubusercontent.com/cleanlab/label-errors/main
      for pair in mnist:mnist_test_set cifar10:cifar10_test_set cifar100:cifar100_test_set caltech256:caltech256 imagenet:imagenet_val_set quickdraw:quickdraw 20news:20news_test_set imdb:imdb_test_set amazon:amazon audioset:audioset_eval_set; do
        d="${pair%%:*}"; f="${pair##*:}"
        fetch "$BASE/cross_validated_predicted_labels/${f}_pyx_argmax_predicted_labels.npy" "$ROOT/label_errors/${d}_cv_pred.npy"
        fetch "$BASE/original_test_labels/${f}_original_labels.npy" "$ROOT/label_errors/${d}_orig.npy"
        fetch "$BASE/mturk/${d}_mturk.json" "$ROOT/label_errors/${d}_mturk.json"
      done
      fetch "$BASE/dataset_indexing/imdb_test_set_index_to_filename.json" "$ROOT/label_errors/imdb_index.json"
      fetch "$BASE/dataset_indexing/imagenet_val_set_index_to_filepath.json" "$ROOT/label_errors/imagenet_index.json"
      fetch http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv "$ROOT/label_errors/class_labels_indices.csv"
      ;;
    dclm)
      # three shards of the unfiltered pool and of the filtered baseline of DataComp-LM
      for i in 0 1 2; do
        n="$(printf "%08d" "$i")"
        fetch "$HF/mlfoundations/dclm-pool-1b-1x/resolve/main/CC_shard_$n.jsonl.zst" "$ROOT/dclm_pool/CC_shard_$n.jsonl.zst"
        fetch "$HF/mlfoundations/dclm-baseline-1.0/resolve/main/global-shard_01_of_10/local-shard_0_of_10/shard_${n}_processed.jsonl.zst" "$ROOT/dclm_base/shard_$n.jsonl.zst"
      done
      ;;
    cleanml)
      # the code of the CleanML benchmark at a fixed commit and its archive of 13 tables with real errors
      if [ ! -d "$ROOT/cleanml/repo/.git" ]; then
        run git clone https://github.com/chu-data-lab/CleanML "$ROOT/cleanml/repo"
        run git -C "$ROOT/cleanml/repo" checkout 92bcb77316e808453b7df42e71454d405f90cc2a
      fi
      fetch "https://www.dropbox.com/s/nerfrhbrseev928/CleanML-datasets-2020.zip?dl=1" "$ROOT/cleanml/CleanML-datasets-2020.zip"
      run unzip -q -n "$ROOT/cleanml/CleanML-datasets-2020.zip" -d "$ROOT/cleanml/repo"
      run unzip -q -n "$ROOT/cleanml/repo/result.zip" -d "$ROOT/cleanml/published"
      ;;
    datacomp)
      # one shard of the DataComp small pool, its table and the two CLIP B/32 arrays, which are the first members of the uncompressed archive
      s=0020f0cbd157d470aa56bea08e304b90
      fetch "$HF/mlfoundations/datacomp_small/resolve/main/$s.parquet" "$ROOT/datacomp/$s.parquet"
      fetch_range "$HF/mlfoundations/datacomp_small/resolve/main/$s.npz" "$ROOT/datacomp/b32_part.bin" 0 1035614634
      ;;
    mirflickr)
      # MIRFLICKR-25000, images with their Flickr tags, and the manual concept annotations
      fetch http://press.liacs.nl/mirflickr/mirflickr25k.v3b/mirflickr25k.zip "$ROOT/mirflickr25k.zip"
      fetch http://press.liacs.nl/mirflickr/mirflickr25k.v3b/mirflickr25k_annotations_v080.zip "$ROOT/mirflickr25k_annotations_v080.zip"
      ;;
    wilds)
      # WILDS CivilComments as one table, and a subset of Camelyon17 fetched by native/prep_camelyon.py from the published Parquet files
      fetch "$HF/pietrolesci/civilcomments-wilds/resolve/main/raw/train-00000-of-00001.parquet" "$ROOT/wilds/civilcomments_raw.parquet"
      run python3 "$HERE/../native/prep_camelyon.py" --native "$ROOT"
      ;;
    llmtexts)
      # the texts of three test sets of Northcutt et al. for llm/prep_labels.py
      fetch https://s3.amazonaws.com/fast-ai-nlp/imdb.tgz "$ROOT/llm_texts/imdb.tgz"
      fetch https://ndownloader.figshare.com/files/5975967 "$ROOT/llm_texts/20news-bydate.tar.gz"
      for p in partaa partab; do
        fetch "https://github.com/cleanlab/label-errors/releases/download/amazon-reviews-dataset/amazon5core.tar.gz-$p" "$ROOT/llm_texts/amazon5core.tar.gz-$p"
      done
      ;;
    agnews)
      # AG News for the auxiliary runner study/run_study.py, written as train.csv and test.csv
      run python3 -c "import os, sys; from datasets import load_dataset; d = load_dataset('fancyzhx/ag_news'); os.makedirs(sys.argv[1], exist_ok=True); [d[s].to_csv(f'{sys.argv[1]}/{s}.csv', index=False) for s in ('train', 'test')]" "$ROOT/agnews"
      ;;
    release)
      # saved test predictions of the image runs and scores of the corpus documents, from the GitHub Release of this repository
      REL=https://github.com/qzkinhit/HQDataset-Assessment/releases/download/v1.0
      for f in hqd-data-predictions.tar hqd-data-scores.tar; do
        fetch "$REL/$f" "$ROOT/release/$f"
      done
      run tar -xf "$ROOT/release/hqd-data-predictions.tar" -C "$ROOT"
      run tar -xf "$ROOT/release/hqd-data-scores.tar" -C "$ROOT"
      ;;
    *)
      echo "Unknown dataset $name" >&2
      exit 2
      ;;
  esac
done
