# Assessment of high-quality datasets

This repository holds the code, the data and the results of the survey *A survey on task-oriented assessment of multimodal high-quality datasets* (in Chinese). The survey organizes data quality standards and assessment methods by five assessment levels, counts the levels of the requirement items in ten assessment documents, and assesses 54 public datasets. The repository also holds the method package `hqd_assess` with its tests, and documents that the survey leaves out, namely a six-stage assessment procedure with its algorithms, the descriptions of all datasets and parameters, and comparisons of tools, standards and published results.

The Chinese version is [README_zh.md](README_zh.md).

## Scale of the work

| Item | Scale |
|---|---|
| Literature | 22 search queries on Semantic Scholar from 2010 to November 2026, 52213 hits, 21835 papers after screening by citations and removing duplicates, 207 references in the survey |
| Standards and policies | National standards, drafts for comment, technical documents of TC609, ISO/IEC 5259, ISO 8000, the EU AI Act and the NIST AI RMF, read in full where a public text exists |
| Datasets | 54 datasets in 11 groups, about $3.39\times 10^8$ records |
| Downloads | About 17 GiB from 146 files of the official sources, fetched by `scripts/download_data.sh`, besides the files in `data/` |
| Assessment practices | 66 practices from the five method families of the survey |
| Assessment of public datasets | 6 research questions over 11 groups of datasets, 44041 model trainings (each configuration counted once), about 760 hours of run time recorded in the result files, on a 64-core CPU server and A6000 GPUs |
| Code | About 7200 lines of Python for the assessment and the package `hqd_assess`, and 163 tests |

| Group of datasets | Datasets | Records |
|---|---:|---:|
| Tables with benchmark errors | 11 | 299,122 |
| Tables with real errors | 13 | 589,415 |
| Image labels | 2 | 120,000 |
| Language-model corpora | 2 | 1,564,345 |
| Multimodal data | 2 | 530,671 |
| Synthetic data | 3 | 350,000 |
| Dynamic data | 1 | 259,287,888 |
| Evaluation protocol | 6 | 15,971,700 |
| Data leakage | 3 | 24,000 |
| Test-set labels | 10 | 60,585,386 |
| Detector estimates | 1 | 5,000 |
| Total | 54 | 339,327,527 |

A static scan over a whole table counts its records. `analysis/big_tables.py` computes the records per dataset from the result files.

## Basic concepts and assessment levels

Section 2 of the survey writes a quality claim about a dataset as a six-tuple of object, dimension, measure, estimate with uncertainty, assessment level and usage context. The level records how a conclusion is obtained, and the survey organizes standards and methods by these five levels.

| Level | Name | What is done |
|---|---|---|
| L1 | Declared | The provider states a property in its documents |
| L2 | Matched | Statements or rules are compared with the data, logs and lineage |
| L3 | Measured | A defined estimator is computed on the data with its uncertainty |
| L4 | Task-tested | A given model is trained on the dataset and task utility is measured on held-out data |
| L5 | Interventional | One aspect of the data changes while the other conditions stay fixed, and the downstream results are compared |

The running example is the classic Tennessee Eastman fault-diagnosis benchmark. It has no missing values and almost no duplicates, so its record-level score is near full marks. On the same files the Macro-F1 of one random forest moves by 0.233 when the split changes, every test window of a random window split shares rows with training windows, and labels taken from file names mark 16.7 percent of each test fault file wrongly.

[docs/framework.md](docs/framework.md) restates the definitions of the survey. It also writes a six-stage assessment procedure as Algorithms 1 to 6, which is documentation of this repository and not part of the survey.

## The package `hqd_assess`

The package holds pure functions without file or network access, and it needs NumPy and SciPy. Run the code from the repository root, or install it with `pip install .`.

| Module | Content |
|---|---|
| `claim` | `QualityClaim`, `Estimate`, `UsageContext`, `DatasetVersion`, `AssessmentLevel`, `minimum_level`, `task_gain` |
| `scoring` | Scores of GB/T 48562-2026, `SUB_INDICATORS`, single-item floors, model fit |
| `sampling` | Sample sizes, Wilson and Jeffreys intervals, `rogan_gladen`, `calibrate_error_rate` (Algorithm 1) |
| `split_check` | Group and window overlap, `check_split_independence` (Algorithm 2) |
| `selection` | `select_version` (Algorithm 3), `static_rule`, `hindsight_best`, `replay_rules` |
| `interaction` | `interaction_test` (Algorithm 4), `paired_flags`, `additivity_counterexample` |
| `judge` | `judge` (Algorithm 5) |
| `renewal` | `renew` (Algorithm 6), `stages_to_redo`, `violation_trigger`, `trigger_counts` |

```python
from hqd_assess import (AssessmentLevel, Estimate, QualityClaim, Requirement, UsageContext, judge, minimum_observed_rate,
                        model_application_score, model_fit_value, proportion_sample_size, single_item_floor,
                        wilson_interval)

print(proportion_sample_size(0.9, 0.02))                    # 865 records for a half-width of 2 points
print(single_item_floor(8), single_item_floor(4))           # floors 0.2 and 0.6 of one item at 90 points
print(minimum_observed_rate(0.96, 4000))                    # 0.96625 reaches a lower bound of 96 percent
print(model_application_score(model_fit_value(0.697, 0.8))) # 87.9 points fail the pass line

usage = UsageContext("fault diagnosis", "Macro-F1", "split by run", ("random forest",), "base mode")
low, high = wilson_interval(3865, 4000)
safety = QualityClaim("TEP test set", "safety conformity", "share of safe records", Estimate(3865 / 4000, low, high, "Wilson"),
                      AssessmentLevel.L3, usage)
fit = QualityClaim("TEP test set", "model fit", "Macro-F1 by run", Estimate(0.636), AssessmentLevel.L4, usage)
verdict = judge([Requirement(safety, AssessmentLevel.L3, 0.96)], [Requirement(fit, AssessmentLevel.L4, 0.80)])
print(verdict.passed, verdict.records[0].below_threshold)   # False True
print(fit)
```

The examples and the tests run on the data of the repository.

```bash
pip install -r requirements.txt
python examples/tep_assessment.py                    # static scan, label check, split check, claims
python examples/table_claims.py --table hospitals    # error rates, audit, calibration, version choice, judgement
python analysis/print_tables.py                      # main results per dataset
pytest
```

## What is in the repository

| Path | Content |
|---|---|
| `hqd_assess/` | The method package |
| `tests/` | Tests that reproduce every number the survey derives by calculation, and that compare the package with the stored results |
| `examples/` | `tep_assessment.py` on classic TEP and `table_claims.py` on one table |
| `docs/framework.md`, `docs/framework_zh.md` | The definitions and assessment levels of the survey, and the six-stage procedure with six algorithms that the repository documents |
| `docs/datasets.md` | Official source, license as stated by the source, use, scale, errors, download command and scripts of every dataset |
| `docs/parameters.md` | Every engineering parameter with the file that holds it, and the runs stored per result file |
| `docs/release_assets.md` | The content of each release archive |
| `docs/tools.md`, `docs/published_results.md`, `docs/standards.md` | Comparison of tools, published results on static quality and task utility, standards with the levels of their indicators |
| `data/` | The small data files, see `data/README.md` |
| `case_tep/`, `big/`, `study/`, `native/`, `corpus/`, `llm/` | Scripts of the assessment tracks |
| `analysis/` | Summaries of the runs, the numbers of the survey and tables |
| `figures/` | The result figures |
| `results/` | Result records of the runs |
| `scripts/download_data.sh` | Download commands for the datasets from their sources |

## Assessment tracks

| Track | Data | Errors | Question | Scripts |
|---|---|---|---|---|
| Tables | 11 tables with dirty and clean versions | Benchmark | Does the score of the standard determine the task score, are repairs additive, how do shares of repair and deletion act | `study/run_real_tables.py`, `study/run_tables_fine.py` |
| Version choice | The same tables | Benchmark | What does a task test on $n$ records gain over the static rule | `study/run_decision.py`, `study/run_decision_fine.py` |
| Cleaning methods | 13 CleanML tables | Real | Which cleaning method of the benchmark a task test selects | `native/clean_cleanml.py`, `native/run_cleanml.py` |
| Detectors | The same 11 tables | Benchmark | How close do detectors and samples come to the true error rate | `study/run_detectors.py` |
| Running example | Classic TEP | Protocol | How far do split, window length and labels move a score | `case_tep/run_case.py`, `case_tep/run_windows.py` |
| Independent runs | Rieth TEP, 15.33 million records | Protocol | Does the number of independent runs matter at a fixed number of records | `big/run_rieth.py`, `big/run_rieth_models.py` |
| Time | NYC TLC yellow taxi records, 72 months, 259 million records | Real | Do rule violations follow the change of task utility, what do filtering and scale recover | `big/run_taxi.py` |
| Human label noise | CIFAR-10N, CIFAR-100N | Real | How do repair and deletion of human label errors act | `study/run_vision.py` |
| Coverage | CIFAR-100 superclasses | Coverage | How does a missing subpopulation act against the same loss of scale | `study/run_vision.py` |
| Near duplicates | ciFAIR lists on the CIFAR test sets | Benchmark | How much of a test score comes from images with a near duplicate in training | `study/fair.py` |
| Distribution shift | WILDS Camelyon17, CivilComments | Real | How far do hospital and subgroup shifts move a score | `native/prep_camelyon.py`, `native/run_camelyon.py`, `native/run_civilcomments.py` |
| Web corpus | C4 shards with Wikipedia as a reference, DCLM corpora | Real | Does the lowest perplexity select better text, under which criterion and scale | `corpus/run_corpus.py`, `native/prep_dclm.py` |
| Leakage and corpus defects | Five-domain text pool | Constructed | What does a benchmark with leaked test data report | `study/run_text.py` |
| Image-text | COCO val2017 | Constructed | Retrieval under one defect, error of a similarity proxy | `study/run_multimodal.py` |
| Image-text pools | DataComp shard, MIRFLICKR-25000 | Real | How do filters on pairs act on retrieval | `native/run_datacomp.py`, `native/embed_mirflickr.py`, `native/run_mirflickr.py` |
| Synthetic images | EDM, DDPM and CIFAKE images, CIFAR-10.1 | Real generators | What do synthetic training images score on real test sets, do fidelity scores follow the task score | `study/run_synth.py`, `native/fidelity.py` |
| Test-set label errors | Ten test sets of Northcutt et al. | Real | How far do reviewed labels move a test score | `native/label_errors.py` |
| Language model baselines | The tables, C4 and three test sets | As above | What do a repair, a corpus judge and a labeller based on a language model reach | `llm/`, see `llm/README.md` |

## Datasets

[docs/datasets.md](docs/datasets.md) has one entry per dataset with its official source, the license as stated by the source, its use and the download command. The tables with benchmark errors and the 44 classic TEP files are in `data/`, with the ciFAIR lists and the reviewer votes of the test-set label errors. `scripts/download_data.sh DIR` downloads the other datasets from their public hosts into `DIR`, and `scripts/download_data.sh DIR rieth taxi` selects some of them. All links of the script were tested on 7 November 2026.

| Group | Datasets |
|---|---|
| Tables with benchmark errors, from [REIN](https://github.com/mohamedyd/rein-benchmark) and UniClean | [Adult](https://archive.ics.uci.edu/dataset/2/adult), [Beers](https://www.kaggle.com/datasets/nickhould/craft-cans), [Breast Cancer](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original), [Flights](http://lunadong.com/fusionDataSets.htm), [Hospitals](https://github.com/BigDaMa/raha), [SmartFactory](https://www.kaggle.com/datasets/inIT-OWL/high-storage-system-data-for-energy-optimization), [Soccer](https://db.unibas.it/projects/bart/), [Bike](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), [Mercedes](https://www.kaggle.com/competitions/mercedes-benz-greener-manufacturing/data), [NASA](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise), [Soil Moisture](https://github.com/felixriese/hyperspectral-soilmoisture-dataset) |
| Tables with real errors, from [CleanML](https://github.com/chu-data-lab/CleanML) | Airbnb, Citation, Company, Credit, EEG, KDD, Marketing, Movie, Restaurant, Sensor, Titanic, University, USCensus |
| Image labels | [CIFAR-10N, CIFAR-100N](https://github.com/UCSC-REAL/cifar-10-100n) on [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html) |
| Language-model corpora | [C4](https://huggingface.co/datasets/allenai/c4), [DCLM pool](https://huggingface.co/datasets/mlfoundations/dclm-pool-1b-1x), [DCLM baseline](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0), [Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) as a reference, a five-domain pool from [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) and four other sources |
| Multimodal data | [DataComp](https://huggingface.co/datasets/mlfoundations/datacomp_small), [MIRFLICKR-25000](https://press.liacs.nl/mirflickr/), [COCO val2017](https://cocodataset.org/#download) |
| Synthetic data | [EDM images](https://github.com/wzekai99/DM-Improves-AT), [DDPM images](https://github.com/google-deepmind/deepmind-research/tree/master/adversarial_robustness), [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) |
| Dynamic data | [NYC TLC yellow taxi records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) |
| Evaluation protocol | [Classic TEP](http://web.mit.edu/braatzgroup/links.html), [Rieth TEP](https://doi.org/10.7910/DVN/6C3JR1), [Camelyon17 and CivilComments](https://wilds.stanford.edu/datasets/), [CIFAR-10.1](https://github.com/modestyachts/CIFAR-10.1), CIFAR-100 superclasses |
| Near duplicates | [ciFAIR-10, ciFAIR-100](https://cvjena.github.io/cifair), the near-duplicate groups of C4 |
| Test-set labels | [Northcutt et al.](https://github.com/cleanlab/label-errors) on MNIST, CIFAR-10, CIFAR-100, Caltech-256, ImageNet, QuickDraw, 20 News, IMDB, Amazon, AudioSet |

## Reproduction

Each script takes its paths from the command line or from the environment variables `HQD_ROOT`, `HQD_POOL` and `HQD_CORPUS`, and the defaults are relative to the repository, with `downloads/` for downloaded data and `results/` for result files. The job runners append to their result file and skip finished jobs, so a run can stop and resume. `--procs`, `--threads`, `--gpus` and `--per-gpu` set the hardware use. `docs/parameters.md` lists the settings. The commands run from the repository root.

```bash
pip install -r requirements.txt
DIR=downloads
scripts/download_data.sh $DIR

# Tables and version choice
python study/run_real_tables.py
python study/run_tables_fine.py
python study/run_decision.py
python study/run_decision_fine.py
python analysis/tables_summary.py results/results_real_tables.jsonl > results/tables_summary.json
python analysis/table_sizes.py data/tables results/table_sizes.json

# Detectors on the same tables
python study/run_detectors.py --root data/tables --meta data/tables/table_meta.json --out $DIR/detectors --stage detect
python study/run_detectors.py --root data/tables --meta data/tables/table_meta.json --out $DIR/detectors --stage raha    # needs the raha package
python study/run_detectors.py --root data/tables --meta data/tables/table_meta.json --out $DIR/detectors --stage measure
python study/run_detectors.py --root data/tables --meta data/tables/table_meta.json --out $DIR/detectors --stage check
cp $DIR/detectors/results_detectors.jsonl $DIR/detectors/detectors_summary.json $DIR/detectors/sanity_detectors.json results/
python analysis/validate_detector_intervals.py --flags-dir $DIR/detectors/flags --previous-results results/results_detectors.jsonl --skip-enumeration --output-dir $DIR/intervals
cp $DIR/intervals/results_finite_population.jsonl results/

# Classic TEP
python case_tep/run_case.py
python case_tep/run_windows.py
python case_tep/make_traces.py data/tep results/tep_traces.json

# Rieth TEP and taxi records
python big/run_rieth.py --seeds 5
python big/run_rieth_models.py
python big/run_taxi.py
python big/run_taxi.py --sizes 100000 --out results/results_taxi_100k.json
python big/run_taxi.py --sizes 10000,30000,300000,3000000 --out results/results_taxi_sizes.json

# Job lists of the GPU tracks, jobs per card come from study/gpu_pool.py
export HQD_ROOT=$DIR HQD_POOL=$DIR/processed HQD_CORPUS=$DIR/corpus
mkdir -p jobs
python study/make_jobs.py jobs
python corpus/make_jobs_corpus.py jobs/corpus.jsonl
python native/make_jobs_native.py jobs/synth.jsonl jobs/dclm.jsonl

# Human label noise, coverage and near duplicates
python study/prep_vision.py $DIR $DIR/cifar_n $DIR/cifar-10-python.tar.gz
python study/gpu_pool.py --jobs jobs/vision.jsonl --script study/run_vision.py --out results/results_vision.jsonl
python study/gpu_pool.py --jobs jobs/vision_fine.jsonl --script study/run_vision.py --out results/results_vision_fine.jsonl
python study/coverage_reference.py --root $DIR --results results/results_vision.jsonl --out results/results_vision_coverage_ref.json   # accuracy of the full-coverage models on the fine classes that smaller shares leave out
python study/fair.py $DIR results/results_fair.json

# Synthetic images, fidelity and the replay of the image versions
python native/prep_native.py $DIR $DIR/cifar10_real.npz
python study/gpu_pool.py --jobs jobs/synth.jsonl --script study/run_synth.py --out results/results_synth.jsonl
python study/embed_cifar.py --npz $DIR/cifar10_real.npz --out $DIR/cifar10_clip.npz
python native/fidelity.py --out results/results_fidelity.json
python native/replay_versions.py $DIR results/results_versions.jsonl

# Web corpus and DCLM
python corpus/prep_corpus.py --dir $DIR/corpus --out $DIR/corpus
cp $DIR/corpus/corpus_stats.json results/corpus_stats.json
python corpus/static_corpus.py $DIR/corpus results/corpus_static.json
python native/prep_dclm.py $DIR/corpus/tok/tokenizer.json $DIR $DIR/corpus
python corpus/run_corpus.py --job '{"mode": "ppl"}' --out /dev/null
python study/gpu_pool.py --jobs jobs/corpus.jsonl --script corpus/run_corpus.py --out results/results_corpus.jsonl
python study/gpu_pool.py --jobs jobs/dclm.jsonl --script corpus/run_corpus.py --out results/results_corpus.jsonl

# Text pool
python study/run_text.py --job '{"mode": "ppl"}' --out /dev/null
python study/gpu_pool.py --jobs jobs/text.jsonl --script study/run_text.py --out results/results_text.jsonl
python study/gpu_pool.py --jobs jobs/text_fine.jsonl --script study/run_text.py --out results/results_text_more.jsonl

# COCO image-text pairs, and the image-text pools
python study/embed_coco.py $DIR
python study/gpu_pool.py --jobs jobs/multimodal.jsonl --script study/run_multimodal.py --out results/results_multimodal.jsonl
python study/gpu_pool.py --jobs jobs/multimodal_fine.jsonl --script study/run_multimodal.py --out results/results_multimodal_fine.jsonl
python native/run_datacomp.py --native $DIR --out results/results_datacomp.jsonl
python native/embed_mirflickr.py --native $DIR --out $DIR/mirflickr_clip.npz --device cuda
python native/run_mirflickr.py --native $DIR --feats $DIR/mirflickr_clip.npz --out results/results_mirflickr.jsonl

# CleanML tables, each table is cleaned from inside the repository of the benchmark
REPO=$PWD
for t in Airbnb Citation Company Credit EEG KDD Marketing Movie Restaurant Sensor Titanic University USCensus
do
  (cd $DIR/cleanml/repo && python $REPO/native/clean_cleanml.py --dataset $t)
done
python native/run_cleanml.py --repo $DIR/cleanml/repo --out results/results_cleanml.jsonl
python native/audit_cleanml_acc.py --repo $DIR/cleanml/repo --results results/results_cleanml.jsonl --published $DIR/cleanml/published/result --out $DIR/cleanml/audit_cleanml_acc.jsonl
cp $DIR/cleanml/audit_cleanml_acc.jsonl.published.json results/audit_cleanml_published.json

# WILDS
python native/run_civilcomments.py --mode features --native $DIR
for s in 0 1 2 3 4
do
  for m in tfidf_lr tfidf_xgb
  do
    python native/run_civilcomments.py --native $DIR --out results/results_wilds.jsonl --job "{\"id\": \"civil_${m}_s$s\", \"model\": \"$m\", \"seed\": $s}"
  done
  python native/run_camelyon.py --out results/results_wilds.jsonl --job "{\"id\": \"cam_resnet18_s$s\", \"seed\": $s}"
done

# Test-set label errors
python native/label_errors.py $DIR/label_errors $DIR/preds results/label_errors.json
python native/label_errors_audioset.py $DIR/label_errors $DIR/label_errors/class_labels_indices.csv results/results_audioset_labels.json

# Static quality scores of the native tracks, which read the result files next to the data
cp results/results_datacomp.jsonl results/results_mirflickr.jsonl results/results_fidelity.json $DIR/
python native/static_native.py --native $DIR --rieth $DIR/rieth --out results

# Numbers, tables and figures
python analysis/summaries.py results
python analysis/make_numbers.py results                 # every number of the text as JSON
python analysis/print_tables.py                         # plain tables per dataset
python analysis/print_tables.py --latex                 # Table 3 and Table 4 of the survey in LaTeX
python figures/plot_results.py results figures/plots    # the single panels res_*.pdf form Figures 4 to 8
python native/audit_native.py results                   # finished runs of the native tracks against their job lists
python analysis/stored_runs.py --update docs/parameters.md
```

The language model baselines have their own commands in [llm/README.md](llm/README.md). The documentation scores in `results/dataset_docs.json` come from reading the source pages named in that file, with a quotation per item.

## Where each figure and table comes from

Figures 1 to 3 and the tables that summarize standards and methods are typeset with the text of the survey, and the sources of their content are listed here. The last column names the blocks of the commands above, or a command.

| Item of the survey | Result file | Script | Commands |
|---|---|---|---|
| Figure 1, static checks and task scores of TEP | `results/case_tep.json`, `results/tep_traces.json` | `case_tep/run_case.py`, `case_tep/make_traces.py` | `Classic TEP` |
| Figure 2, evolution of the field | | | Works cited in the survey |
| Figure 3, basic concepts and assessment levels | `docs/framework.md` | `hqd_assess/claim.py`, `hqd_assess/scoring.py` | Drawn from the definitions |
| Figure 4(a), choice of a version | `results/results_decision.jsonl`, `results/results_decision_fine.jsonl` | `study/run_decision.py`, `study/run_decision_fine.py`, `figures/plot_results.py` | `Tables and version choice`, `Numbers, tables and figures` |
| Figure 4(b), partial repair as a share of the full-repair gain | `results/results_real_tables.jsonl`, `results/results_tables_fine.jsonl` | `study/run_tables_fine.py`, `figures/plot_results.py` | `Tables and version choice`, `Numbers, tables and figures` |
| Figure 4(c), cleaning methods on the CleanML tables | `results/results_cleanml.jsonl` | `native/clean_cleanml.py`, `native/run_cleanml.py`, `figures/plot_results.py` | `CleanML tables`, `Numbers, tables and figures` |
| Figure 4(d), repair and deletion of human label errors | `results/results_vision.jsonl`, `results/results_vision_fine.jsonl` | `study/run_vision.py`, `figures/plot_results.py` | `Human label noise`, `Numbers, tables and figures` |
| Figure 5(a) to (d), gains of repair and deletion, random deletion of as many records, Spearman correlation along the repair and delete paths, joint against single repairs | `results/tables_summary.json`, `results/results_real_tables.jsonl`, `results/results_random_delete.jsonl` | `study/run_real_tables.py`, `study/run_random_delete.py`, `analysis/tables_summary.py`, `figures/plot_results.py` | `Tables and version choice`, `Numbers, tables and figures` |
| Figure 6(a), protocol against data errors | `results/tables_summary.json`, `results/case_tep.json` | `case_tep/run_case.py`, `figures/plot_results.py` | `Classic TEP`, `Numbers, tables and figures` |
| Figure 6(b), independent runs | `results/results_rieth.json`, `results/results_rieth_models.jsonl` | `big/run_rieth.py`, `big/run_rieth_models.py`, `figures/plot_results.py` | `Rieth TEP and taxi records`, `Numbers, tables and figures` |
| Figure 6(c), error-rate estimates and intervals | `results/results_detectors.jsonl`, `results/results_finite_population.jsonl` | `study/run_detectors.py`, `analysis/validate_detector_intervals.py`, `figures/plot_results.py` | `Detectors on the same tables`, `Numbers, tables and figures` |
| Figure 6(d), cells repaired and damaged by Qwen | `results/results_llm_cells.json`, `results/results_llm_tables.jsonl` | `llm/repair_tables.py`, `llm/score_tables.py`, `figures/plot_results.py` | [llm/README.md](llm/README.md), `Numbers, tables and figures` |
| Figure 7(a), coverage of fine classes | `results/results_vision.jsonl`, `results/results_vision_coverage_ref.json` | `study/run_vision.py`, `study/coverage_reference.py`, `figures/plot_results.py` | `Human label noise`, `Numbers, tables and figures` |
| Figure 7(b), (c), $R^2$ by training month, rule violations | `results/results_taxi.json`, `results/results_taxi_monthly.json` | `big/run_taxi.py`, `figures/plot_results.py` | `Rieth TEP and taxi records`, `Numbers, tables and figures` |
| Figure 7(d), training size and time | `results/results_taxi.json`, `results/results_taxi_100k.json`, `results/results_taxi_sizes.json` | `big/run_taxi.py`, `figures/plot_results.py` | `Rieth TEP and taxi records`, `Numbers, tables and figures` |
| Figure 8(a), (b), filters on C4 and by model size | `results/results_corpus.jsonl` | `corpus/run_corpus.py`, `figures/plot_results.py` | `Web corpus and DCLM`, `Numbers, tables and figures` |
| Figure 8(c), synthetic images | `results/results_synth.jsonl` | `study/run_synth.py`, `figures/plot_results.py` | `Synthetic images`, `Numbers, tables and figures` |
| Figure 8(d), filters on image-text pools | `results/results_datacomp.jsonl`, `results/results_mirflickr.jsonl` | `native/run_datacomp.py`, `native/run_mirflickr.py`, `figures/plot_results.py` | `COCO image-text pairs`, `Numbers, tables and figures` |
| Table 1, main standards and specifications on data quality | `docs/standards.md` | | |
| Table 2, comparison of the five method families | `docs/framework.md` | | |
| Table 3, public datasets by data category | `results/table_sizes.json`, `results/results_sizes_native.json`, `results/label_errors.json`, `results/results_audioset_labels.json`, `results/case_tep.json`, `results/results_rieth.json`, `results/results_taxi.json`, `results/corpus_stats.json`, `results/results_cleanml.jsonl`, `results/results_datacomp.jsonl`, `results/results_wilds.jsonl`, `results/results_multimodal.jsonl` | `analysis/datasets.py`, `analysis/big_tables.py` | `python analysis/print_tables.py --latex` |
| Table 4, assessment results on all datasets | `results/tables_summary.json`, `results/static_native.json`, `results/corpus_static.json`, `results/dataset_docs.json`, `results/results_decision_fine.jsonl`, `results/results_versions.jsonl`, `results/results_cleanml.jsonl`, `results/results_corpus.jsonl`, `results/results_taxi.json`, `results/results_datacomp.jsonl`, `results/results_mirflickr.jsonl`, `results/case_tep.json`, `results/results_rieth.json`, `results/results_wilds.jsonl`, `results/results_synth.jsonl`, `results/results_vision.jsonl`, `results/results_fair.json`, `results/results_multimodal.jsonl`, `results/label_errors.json` | `analysis/datasets.py`, `analysis/big_tables.py` | `python analysis/print_tables.py --latex` |
| Table 5, assessment levels of the requirement items of ten assessment documents | `data/requirements/items.jsonl` | `analysis/requirement_levels.py` | `python analysis/requirement_levels.py` |
| Numbers in the text | `results/numbers_*.json`, `results/tables_summary.json`, `results/insights_summary.json`, `results/results_random_delete.jsonl` | `analysis/summaries.py`, `analysis/insights.py`, `study/run_random_delete.py`, `analysis/make_numbers.py` | `Numbers, tables and figures` |
| Numbers that the survey derives by calculation | | `hqd_assess/`, `tests/test_paper_numbers.py` | `pytest tests/test_paper_numbers.py` |

## Results

`results/` holds the result record of each track, one JSON object per run in the JSONL files and a list of runs in the JSON files. `docs/parameters.md` gives the number of scored runs per file. The files `results/numbers_*.json` hold the numbers of the text, `results/tables_summary.json` holds the summary of the table runs, `results/audit_native.json` holds the comparison of the native tracks with their job lists, `results/audit_cleanml_published.json` holds the accuracies of the CleanML runs next to the published ones, `results/results_detectors.jsonl`, `results/detectors_summary.json` and `results/sanity_detectors.json` hold the detector estimates with their summary and checks, and `results/dataset_docs.json` holds the documentation scores with their quotations. `results/results_civilcomments.jsonl` holds the same CivilComments rows as `results/results_wilds.jsonl`. A row with an `error` field is a run that was not scored, for example a deletion that leaves fewer than 30 records.

The saved test predictions of the image runs and the scores of the corpus documents, which take GPU time to compute again, are attached to the GitHub Release [v1.0](https://github.com/qzkinhit/HQDataset-Assessment/releases/tag/v1.0). `scripts/download_data.sh DIR release` fetches and unpacks them. [docs/release_assets.md](docs/release_assets.md) lists their content.

## License

The code, documents and result records are under the MIT license in `LICENSE`. Data keep the terms of their sources, which `docs/datasets.md` lists per dataset. `data/tep/LICENSE` is the license text of the classic TEP distribution.

## Citation

```bibtex
@article{hqd_assessment,
  title   = {A survey on task-oriented assessment of multimodal high-quality datasets},
  author  = {Zekai Qian and Xiaoou Ding and Hongzhi Wang and Chen Wang},
  journal = {Scientia Sinica Informationis},
  year    = {2026},
  note    = {In Chinese, submitted}
}
```

## References of the benchmarks

- M. Abdelaal, C. Hammacher, H. Schöning. REIN: a comprehensive benchmark framework for data cleaning methods in ML pipelines. The 26th International Conference on Extending Database Technology, pp. 499-511, 2023.
- X. Ding, Z. Qian, H. Wang, et al. UniClean: a scalable data cleaning solution for mixed errors based on unified cleaners and optimized cleaning workflow. Proc. VLDB Endow., vol. 18, no. 11, pp. 4117-4130, 2025. Benchmark at https://github.com/qzkinhit/UniClean-bench-Result.
- P. Li, X. Rao, J. Blase, et al. CleanML: a study for evaluating the impact of data cleaning on ML classification tasks. 37th IEEE International Conference on Data Engineering, pp. 13-24, 2021.
