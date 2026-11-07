# Language model baselines

Three baselines with Qwen 2.5 7B Instruct in bfloat16 at temperature zero. All data errors are native or come with the benchmark, nothing is injected.

| Track | What the model does | Scripts | Output |
|---|---|---|---|
| Table repair | Reads a dirty row with a few examples of the same table and replies with the corrected values as JSON, in the style of Narayan et al. (2022) | `prep_tables.py`, `repair_tables.py`, `merge_tables.py`, `score_tables.py` | `results/results_llm_cells.json`, `results/results_llm_tables.jsonl` |
| Corpus judge | Rates a web document from 1 to 5, the expected rating comes from the next-token probabilities of one forward pass | `judge_corpus.py`, arms `llm_high` and `llm_mid` of `corpus/run_corpus.py` | `c4_llm_score.npy`, `results/results_llm_judge.json`, `results/results_corpus.jsonl` |
| Test labels | Labels the text test sets of Northcutt et al. (2021) with zero-shot and five-shot prompts | `prep_labels.py`, `annotate_labels.py` | `results/results_llm_labels.jsonl` |

`qwen.py` holds the calls to the model. `run_llm.py` is the worker, it keeps the model on one card and takes units of all tracks from a shared list, a card is claimed once and released when the list is empty. On a shared card the worker waits inside `qwen.load` until 20000 MiB are free. `make_jobs_llm.py` writes the job lists.

## Choices that affect the numbers

Table repair. The candidate rows are the training records of the ten splits of `study/run_real_tables.py`, at most 3000 per table, drawn with seed 0. The model sees every candidate row as it stands in the dirty table and does not learn which rows hold errors. The examples are 5 rows, 3 for the two wide tables, that are no test record in any split. The model keeps nothing between prompts and a seed reads only the replies for its own training records, so no prompt read by a seed holds a test record of that seed. Rows without a reply stay dirty, the shares are in `results_llm_cells.json`. The two tables with more than 40 columns go in blocks of 30 columns. A cell counts as repaired when it was wrong and now equals the clean value under the comparison of `study/run_real_tables.py`. A reply that would add a class to the label of a classification table is dropped and counted in `label_replies_dropped`, because the classes of a task are fixed and one class with a single training record makes gradient boosting collapse on any table.

Scoring. `score_tables.py` calls the unchanged runners with the repaired table in the place of the dirty one. With no reply at all it reproduces the stored `keep` lines of `results_real_tables.jsonl` and `results_decision.jsonl` exactly. The field `f` is the mean share of training records the model has read.

Corpus judge. The documents are the training pool of `corpus/run_corpus.py`, 300032 of them at random, first 400 tokens of the model's own tokenizer, decoded from the stored token ids. `llm_high` takes the highest ratings up to the token budget, `llm_mid` samples the middle 80 percent of the rated documents. Other arms are not affected.

Test labels. IMDB and 20news are the full test sets, Amazon is a random sample of 50000 reviews plus the 1000 reviewed ones as a separate set, since the reviewed records were sampled from the flagged records. Each set is verified against the label array over all records and against the reviewer text of 40 flagged records. Texts are cut to 400 tokens, examples to 150 tokens. The answer is the option token with the highest next-token probability. Only flagged records were reviewed, so `confirmed_precision_all` is a lower bound and `confirmed_precision_reviewed` is reported beside it.

## How to run

```bash
# CPU side, the tables and the label arrays
python llm/prep_tables.py --root data --meta data/table_meta_real.json --out llm_work/prompts
python llm/prep_labels.py --native native_data --dl downloads --out llm_work/labels

# GPU side, HQD_LLM holds prompts/ and labels/, HQD_CORPUS holds the corpus arrays, QWEN_PATH the model when it is not fetched from the hub
export HQD_LLM=llm_work HQD_CORPUS=corpus_data
python llm/make_jobs_llm.py llm_work/jobs
python study/gpu_pool.py --jobs llm_work/jobs/workers.jsonl --script llm/run_llm.py --out llm_work/results_llm_workers.jsonl --gpus 0
python study/gpu_pool.py --jobs llm_work/jobs/corpus_llm.jsonl --script corpus/run_corpus.py --out results/results_corpus.jsonl --gpus 0

# CPU side again, with llm_work/units copied over
python llm/merge_tables.py --units llm_work/units/tables --prompts llm_work/prompts --root data --meta data/table_meta_real.json --out llm_work/tables --cells results/results_llm_cells.json
python llm/score_tables.py --root llm_work/tables --meta data/table_meta_real.json --cells results/results_llm_cells.json --out results/results_llm_tables.jsonl --procs 16 --seeds 10
```
