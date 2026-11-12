# Release assets

The datasets come from their official sources through `scripts/download_data.sh`, and [datasets.md](datasets.md) lists the source and the license of each. The Release [v1.0](https://github.com/qzkinhit/HQDataset-Assessment/releases/tag/v1.0) holds the files that the runs of this repository wrote and that take GPU time to compute again.

| Asset | Content | Size | Written by |
|---|---|---|---|
| `hqd-data-predictions.tar` | `preds/*.npy`, the saved test predictions of the image runs | 9 MiB | `study/run_vision.py`, `study/run_synth.py` |
| `hqd-data-scores.tar` | `corpus/c4_ppl.npy`, `corpus/c4_llm_score.npy` and `corpus/c4_group.npy`, the perplexity, the Qwen score and the group of each C4 document, and `text_ppl.npz`, the perplexity of the text pool | 13 MiB | `corpus/prep_corpus.py`, `corpus/run_corpus.py`, `llm/judge_corpus.py`, `study/run_text.py` |

## How to fetch and unpack

```bash
scripts/download_data.sh downloads release
```

or by hand

```bash
gh release download v1.0 --repo qzkinhit/HQDataset-Assessment
tar -xf hqd-data-predictions.tar -C downloads
tar -xf hqd-data-scores.tar -C downloads
```

The paths inside the archives follow the layout of `scripts/download_data.sh`, so the files land where the scripts read them.
