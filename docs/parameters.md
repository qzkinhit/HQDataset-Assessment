# Parameters of the experiments

This page lists the engineering parameters that the survey leaves out. Each section names the script that holds the values. The command line defaults are in the scripts, and the job lists fix the grids of the GPU tracks. All tree models, linear models and networks come from scikit-learn, LightGBM or PyTorch with their default settings unless a row says otherwise.

## Tables, `study/run_real_tables.py`

| Parameter | Value |
|---|---|
| Seeds | `--seeds` 20, seeds 0 to 19 |
| Split | Random 70 percent training and 30 percent test, permutation from `default_rng(seed + 500)` |
| Caps | 40000 training records and 15000 test records |
| Test records | Clean values of the test positions |
| Random draws of an action | `default_rng(crc32 of the job key)` |
| Shares of repair and deletion | 0.1, 0.25, 0.5, 0.75 and 1.0 of the dirty records |
| Shares by column | 0.1, 0.25, 0.5 and 0.75 of the wrong cells, most important columns first or last |
| Column order | Importance of 30 extra trees on the first 5000 clean training records, per seed |
| Repair by type | `repair_type_missing`, `repair_type_domain`, `repair_type_value` and `repair_labels`, each at share 1.0 |
| Candidate states per seed and model | 23, the raw table and the 22 actions above |
| Lower limit after deletion | 30 records, a run below it is not scored |
| Missing tokens | Empty string, `nan`, `NaN`, `NA`, `N/A`, `null`, `NULL`, `None`, `?`, `empty`, `-` |
| Equal numbers | Relative difference of at most $10^{-9}$ counts as equal |
| Error types | Missing token, out of domain, wrong value in domain. The domain is the clean value set for categorical columns and the clean range for numeric columns |
| Numeric preprocessing | Median imputation and `StandardScaler` |
| Categorical preprocessing | Most frequent imputation and `OneHotEncoder` with `max_categories=30` |
| Metric | Macro-F1 for classification, $R^2$ for regression |
| Threads | One thread per process, `--procs` 44 |

| Model | Classification | Regression |
|---|---|---|
| `linear` | `LogisticRegression(max_iter=300)` | `Ridge()` |
| `rf` | `RandomForestClassifier(n_estimators=100)` | `RandomForestRegressor(n_estimators=100)` |
| `gb` | `HistGradientBoostingClassifier(max_iter=100)` | `HistGradientBoostingRegressor(max_iter=100)` |
| `knn` | `KNeighborsClassifier(n_neighbors=15)` | `KNeighborsRegressor(n_neighbors=15)` |
| `mlp` | `MLPClassifier((64,), max_iter=60)` | `MLPRegressor((64,), max_iter=200)` with a standardized target |

Gradient boosting with automatic early stopping on more than 10000 records turns early stopping off when a class has one training record, because the internal stratified split needs two.

`study/run_tables_fine.py` adds the shares 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8 and 0.9 for `repair_records` and `delete_records`, with `--seeds` 10 and the same runner.

## Summary of the tables, `analysis/tables_summary.py`

| Parameter | Value |
|---|---|
| Static score | $P=\tfrac{100}{8}(s_{\mathrm{struct}}+s_{\mathrm{format}}+s_{\mathrm{auth}}+5)$, the other five items at full marks |
| Spearman correlation | Per dataset and model across states, with more than 2 distinct scores and at least 6 states |
| Paired comparison | Paired t test against `keep`, at least 3 seeds, Benjamini and Yekutieli correction over all comparisons, level 0.05 |
| Static score goes up | Change above $10^{-6}$ |
| Same static score | States within 0.5 points |
| Interaction | Comparisons with at least 3 runs and at least 2 error types with a mean absolute change above $10^{-9}$, one-sample t test, the same correction and level |

## Decision value, `study/run_decision.py`

| Parameter | Value |
|---|---|
| Seeds | `--seeds` 10 |
| Candidate versions | 15, `keep`, `repair_records` and `delete_records` at 0.25, 0.5, 0.75 and 1.0, `repair_cols_high` and `repair_cols_low` at 0.5, and the 4 repairs by type |
| Held-out set | The test records of the split above |
| Validation and test parts | Half and half, permutation from `default_rng(seed + 900)`, the same for every arm of a seed |
| Nested validation sizes | 25, 50, 100, 200, 400, 800, 1600 and 3200 |
| Comparison rules | Static rule by fewest errors, then fewest wrong labels, then most records. Best test score in hindsight |
| Minimum candidates in a replay | 14, in `analysis/summaries.py` |

`study/run_decision_fine.py` stores the nested validation sizes 5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200, 300, 400, 600, 800, 1200, 1600 and 3200 for the same runs and test scores.

## Classic TEP, `case_tep/run_case.py` and `case_tep/run_windows.py`

| Parameter | Value |
|---|---|
| Seeds | `--seeds` 20 for `official`, `row_random` and `repair`, 10 for the window arms |
| Fault onset | Test faults start after record 160, the run split scores records from there on |
| Hard faults | 3, 9 and 15 |
| Row split by run | Training files for training, test files for testing |
| Row split at random | 70 and 30 percent of the records of the training files and the post-onset test records, random stream `seed + 7` |
| Window length and stride | 20 records and stride 1, `run_windows.py` uses the lengths 5, 10, 15, 20, 30, 40, 60 and 80 with `--seeds` 5 |
| Window subsample | 8000 training windows and 6000 test windows |
| Bootstrap | 200 resamples of the test records |
| Label intervention | Train on the test files with file-name labels, onset labels or without the pre-onset records, score on the training files |
| Expected performance | $a^{*}=0.80$ for the model fit example, in `analysis/make_numbers.py` |

| Model | Definition |
|---|---|
| `rf` | `RandomForestClassifier(n_estimators=200)` |
| `gb` | `HistGradientBoostingClassifier(max_iter=100)` |
| `mlp` | `StandardScaler`, `MLPClassifier((128, 64), max_iter=150)` |
| `linear` | `StandardScaler`, `LogisticRegression(max_iter=400)` |
| `knn` | `StandardScaler`, `KNeighborsClassifier(n_neighbors=15)` |

## Rieth TEP, `big/run_rieth.py` and `big/run_rieth_models.py`

| Parameter | Value |
|---|---|
| Seeds | `--seeds` 3 in `run_rieth.py` and 5 in `run_rieth_models.py`, the stored results of both hold seeds 0 to 4 |
| Fault-present records | Training records after record 20, test records after record 160 |
| Runs per class | 1, 2, 5, 10, 20, 50, 100, 200 and 500 |
| Records per class | 480 at every number of runs. More records use 960 and 1920 per class with 5 runs, and 960 to 7680 per class with 500 runs in `run_rieth.py` or 960 to 3840 in `run_rieth_models.py` |
| Split protocol | Pool of 1500000 records, 80 percent training capped at 400000, against 20 percent of the runs held out with 400000 training records and 300000 test records |
| Test sample | A quarter of the test records in `run_rieth.py`, 400000 records in `run_rieth_models.py` |
| Gradient boosting | LightGBM, `n_estimators=150`, `learning_rate=0.05`, `num_leaves=31`, `min_child_samples=50`, `reg_lambda=1.0`, `subsample=0.8`, `colsample_bytree=0.8` |
| Other models | `RandomForestClassifier(n_estimators=100, min_samples_leaf=2)`, and on standardized inputs `LogisticRegression(max_iter=300)`, `MLPClassifier((128,), max_iter=80)` and `KNeighborsClassifier(n_neighbors=15)` |

## Taxi records, `big/run_taxi.py`

| Parameter | Value |
|---|---|
| Seeds | `--seeds` 2 |
| Training months | Every 6th month of the 72 months, `--stride` 6 |
| Training sizes | `--sizes` 1000000 in `results_taxi.json`, 100000 in `results_taxi_100k.json`, and 10000, 30000, 300000 and 3000000 in `results_taxi_sizes.json`, capped by the records of the month |
| Arms | `raw` and `rule_filtered` |
| Evaluation | Records of the training month and of each later month that pass all rules, at most 100000 per month, protocol `tlc_clean_month_100k_seed_v1` |
| Rules | Pickup in the file month, duration above 0 and at most 6 hours, distance above 0 and at most 200 miles, fare above 0 and at most 500 dollars, 1 to 8 passengers, rate code 1 to 6, speed at most 80 miles per hour, pickup and dropoff zones other than 264 and 265 |
| Target and features | $\log(1+\text{duration})$ with the duration in seconds, from distance, passengers, rate code, pickup zone, dropoff zone, hour, weekday and vendor |
| Model | LightGBM, `n_estimators=200`, `learning_rate=0.1`, `num_leaves=127`, `subsample=0.8`, categorical pickup zone, dropoff zone and rate code |
| Time horizon and groups | 48 months, models trained up to 2020-01 against the later models, in `analysis/summaries.py` |
| Trigger | Violation rate change above 2 points, $R^2$ change above 0.05, in `analysis/summaries.py` |

## Vision, `study/make_jobs.py`, `study/run_vision.py` and `study/fair.py`

| Parameter | Value |
|---|---|
| Seeds | 0 to 9 for the main grid, 0 to 4 for the finer scale grid |
| Models | ResNet-18 and VGG-11 with batch normalization, written in the script |
| Label sets | `aggre_label`, `random_label1`, `worse_label` of CIFAR-10N and `noisy_label` of CIFAR-100N |
| Actions | `keep`, repair at 0.25, 0.5 and 0.75, delete at 0.5 and 1.0, and clean labels |
| Coverage | 5, 4, 3, 2 and 1 fine classes per superclass |
| Scale | 0.8, 0.6, 0.4 and 0.2, and the finer grid 0.1, 0.3, 0.5, 0.7 and 0.9 |
| Training | 30 epochs, batch size 128, SGD with learning rate 0.1, momentum 0.9, weight decay $5\times 10^{-4}$ and Nesterov, `OneCycleLR`, mixed precision |
| Augmentation | Reflect padding of 4 pixels with random crop, random horizontal flip |
| Normalization | Mean 0.4914, 0.4822, 0.4465 and standard deviation 0.2470, 0.2435, 0.2616 |
| Near-duplicate score | Saved test predictions of the runs without a data change, against the lists of ciFAIR |

## Web corpus, `corpus/prep_corpus.py`, `corpus/make_jobs_corpus.py` and `corpus/run_corpus.py`

| Parameter | Value |
|---|---|
| Documents | At most 1200000 from the C4 shards, 20000 Wikipedia articles over 500 characters |
| Tokens | First 512 tokens of the first 6000 characters, tokenizer of `HuggingFaceTB/SmolLM2-135M` |
| Static signals | Alphabetic share of the first 20000 characters, share of tokens inside repeated 5-grams, stop-word share, MinHash signature of 32 permutations, on at most 4000 words |
| Near-duplicate groups | 8 bands of 4 rows, confirmed at a signature agreement of 0.75 |
| Held-out documents | 2 percent by CRC32 of the document position |
| Training pool | 40 percent by the same hash, documents of at least 32 tokens |
| Perplexity scorer | SmolLM2-135M on the first 128 tokens, batch size 128 |
| Filters | `random`, `ppl_low`, `ppl_mid` between the 0.1 and 0.9 quantiles, `ppl_high`, `rule`, `dedup` and `rule_dedup`, and `all` without a budget |
| Rule filter | Alphabetic share at least 0.6, repeated share at most 0.3, stop-word share at least 0.1, at least 50 words |
| Token budgets | 30000000 and 100000000, and 3, 5, 10, 15 and 20 million at model size M |
| Seeds | 0 to 4 for sizes S, M and L at the two main budgets, 0 to 2 for the other sizes and budgets |
| Model | GPT-2 architecture, 256 positions, no dropout, trained from scratch |
| Training | One pass over the tokens, at least 100 steps, batch size 48, sequence length 256, AdamW with learning rate $3\times 10^{-4}$, weight decay 0.01 and betas 0.9 and 0.99, warmup of 500 steps, cosine decay, gradient norm 1.0 |
| Evaluation | Per-token loss on 4000 held-out C4 documents, 4000 of them without a near duplicate, and 4000 Wikipedia articles, at most 256 tokens each, sampled with seed 99 |

| Size | Layers | Heads | Embedding | Parameters in millions |
|---|---|---|---|---|
| `XS` | 2 | 4 | 64 | 3.3 |
| `T` | 4 | 4 | 128 | 7.1 |
| `S` | 4 | 4 | 192 | 11.3 |
| `SM` | 6 | 8 | 256 | 17.4 |
| `MS` | 6 | 8 | 384 | 29.6 |
| `M` | 8 | 8 | 512 | 50.5 |
| `ML` | 10 | 10 | 640 | 80.9 |
| `L` | 12 | 12 | 768 | 123.0 |

## Text pool, `study/build_text_pool.py`, `study/make_jobs.py` and `study/run_text.py`

| Parameter | Value |
|---|---|
| Pool | 3000 high-quality and 2000 low-quality training records and 400 held-out records per domain, texts of 150 to 2000 characters |
| Seeds | 0 to 4 |
| Defects | `truncation`, `template`, `crossdomain`, `lowtier`, `duplicate`, `coverage`, `scale` and `contamination` at the rates 0.05, 0.10, 0.20, 0.30 and 0.40, with the finer contamination rates 0.01, 0.02, 0.15, 0.25 and 0.35 |
| Pairs of defects | Pairs of `truncation`, `template`, `duplicate`, `coverage` and `scale` at 0.2 |
| Filters | `none`, `random`, `ppl`, `ppl_mid` and `oracle` at low-quality shares of 0.25 and 0.67 |
| Base corpus | 6000 high-quality records |
| Model | `HuggingFaceTB/SmolLM2-135M`, at most 192 tokens per document |
| Training | 2 epochs, batch size 16, AdamW with learning rate $2\times 10^{-5}$ and weight decay 0.01, warmup of 50 steps, linear decay, gradient norm 1.0 |
| Score | Per-token negative log-likelihood on the 2000 held-out documents, per domain and split into leaked and unseen documents |

## Image-text pairs, `study/embed_coco.py`, `study/make_jobs.py` and `study/run_multimodal.py`

| Parameter | Value |
|---|---|
| Encoder | CLIP ViT-B/32, `openai/clip-vit-base-patch32`, batch size 256, text of at most 77 tokens |
| Seeds | 0 to 19 |
| Split | 1000 test pairs and 4000 training pairs, permutation from `default_rng(seed + 11)` |
| Defects | `mismatch`, `missing`, `duplicate`, `coverage`, `scale` and `contamination` at the rates 0.05, 0.10, 0.20, 0.30 and 0.40 |
| Finer rates | 0.01, 0.02, 0.15, 0.25, 0.35, 0.5, 0.6, 0.7 and 0.8 for `mismatch`, `duplicate`, `coverage` and `scale`, and 0.01, 0.02, 0.15 and 0.25 for `contamination` |
| Pairs of defects | All pairs of the first five defects at 0.2 |
| Head | Two networks with 512 hidden units, GELU and 256 outputs, learnable logit scale that starts at $\mathrm{e}^{2.66}$ |
| Training | 800 steps, batch size 256, AdamW with learning rate $10^{-3}$ and weight decay 0.01, cosine schedule, symmetric InfoNCE loss |
| Scores | Recall@5, Recall@1 and mean reciprocal rank of image to text retrieval |
| Consistency proxy | Threshold at the 5th percentile of the CLIP similarity of clean pairs |
| Audit | 200 pairs with truth from the construction, then the correction of Rogan and Gladen |

## Synthetic images and test-set label errors, `native/make_jobs_native.py`, `study/run_synth.py` and `native/label_errors.py`

| Parameter | Value |
|---|---|
| Seeds | 0 to 4 for the real training set, 0 to 2 for each generator |
| Generators | `edm` and `ddpm` with 100000 images each, `cifake` with its 50000 generated training images, in `native/prep_native.py`. The complete EDM archive of 200000 images is a second array, `synth_edm_all.npz` |
| Synthetic share | 0.1, 0.25, 0.5, 0.75, 0.9 and 1.0 of the training images replaced with ResNet-18, 0.5 and 1.0 with VGG-11 |
| Added images | 10000, 25000, 50000 and 100000 with ResNet-18, 50000 with VGG-11, and 200000 `edm` images with ResNet-18 from the complete archive. The job `add100000` of `cifake` adds its 50000 images |
| Training | The recipe of the vision track, 30 epochs and batch size 128 |
| Scores | Accuracy on the CIFAR-10 test set and on CIFAR-10.1 |
| Wrong test label | At least three of five reviewers chose the suggested label and fewer than three kept the given one |
| DCLM corpora in the corpus track | `dclm_pool` and `dclm_baseline`, seeds 0 to 2, model sizes `XS`, `S`, `M` and `L`, budget of 30000000 tokens, documents of at least 200 characters |

## CleanML tables, `native/clean_cleanml.py` and `native/run_cleanml.py`

| Parameter | Value |
|---|---|
| Seeds | `--seeds` 5, seeds 0 to 4 |
| Versions | `dirty` and the methods of the benchmark for the error type |
| Missing values | `delete`, `impute_holoclean`, and `impute_NUM_CAT` with NUM of mean, median or mode and CAT of mode or dummy |
| Outliers | `clean_HC_impute_holoclean`, and `clean_DET_impute_NUM_dummy` with DET of SD, IQR or IF and NUM of mean, mode or median |
| Duplicates | `clean` and `AutoER` |
| Inconsistency | `clean` |
| KDD | The versions without HoloClean |
| Split | The split of the archive, the seed of the benchmark is 1 for a table without a split |
| Validation and test parts | Half and half of the dirty test part, permutation from `default_rng(seed + 900)`, the same rows for every version of a seed |
| Nested validation sizes | 5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200, 300, 400, 600, 800, 1200, 1600 and 3200 |
| Caps | The first 40000 training rows and 15000 test rows in dirty order |
| Models | The five classifiers of the tables |
| Class imbalance | Down-sampling of the benchmark, per seed from `default_rng(seed + 500)` |
| Text columns | TF-IDF with English stop words, then the 200 best words by chi square |
| Metric | Macro-F1 |
| Accuracy against the published values | `native/audit_cleanml_acc.py` repeats the runs with accuracy and compares them with `result.zip` of the benchmark |

## Image-text pools, `native/run_datacomp.py`, `native/embed_mirflickr.py` and `native/run_mirflickr.py`

| Parameter | Value |
|---|---|
| Seeds | `--seeds` 5, seeds 0 to 4 |
| Filters | `random`, `sim_top`, `sim_mid`, `sim_bottom`, `text_rule` and `dedup`, at the kept shares 0.05, 0.1, 0.2, 0.3, 0.5, 0.7 and 0.9, and `all` without a filter |
| Text rule | 10 to 300 characters and at least 60 percent letters |
| Near duplicates | The first pair of every CRC32 hash of the lower-case letters of the text |
| Pairs per run | At most 200000 kept pairs of the DataComp shard, the arm `all` uses every pair. 20000 pairs of MIRFLICKR after 5000 held-out images from `default_rng(7)` |
| Head | The two networks of the COCO track, 800 steps, batch size 256, AdamW with learning rate $10^{-3}$ and weight decay 0.01, cosine schedule, symmetric InfoNCE loss |
| Scores | Recall@1, Recall@5 and mean reciprocal rank of image to text retrieval on 1000 validation pairs and 1000 test pairs of COCO, permutation from `default_rng(seed + 11)` |
| Consistency sub-indicator | Share of kept pairs above the 5th percentile of the CLIP similarity of clean COCO pairs |
| Concept precision on MIRFLICKR | Precision at 1 and 5 of the retrieved tag strings among the held-out images, a retrieved string counts when its image shares a manual concept |

## WILDS, `native/prep_camelyon.py`, `native/run_camelyon.py` and `native/run_civilcomments.py`

| Parameter | Value |
|---|---|
| Camelyon17 subset | 80000 training patches and 10000 held-out patches of the training hospitals, 10000 patches of the official test hospital, row groups from `default_rng(17)` |
| Camelyon17 model | ResNet-18 with the ImageNet stem, trained from scratch, 10 epochs, batch size 128, SGD with learning rate 0.1, momentum 0.9, weight decay $5\times 10^{-4}$ and Nesterov, `OneCycleLR`, flips per batch |
| Camelyon17 scores | Accuracy and Macro-F1 on the held-out patches and on the test hospital |
| CivilComments features | TF-IDF with sublinear term frequency, `min_df=5`, word pairs, 300000 features, fitted on the training part |
| CivilComments models | `LogisticRegression(solver="saga", max_iter=30, tol=1e-3)`, and XGBoost on the 5000 words with the highest chi square, `n_estimators=200`, `max_depth=6`, `learning_rate=0.2`, `subsample=0.8`, `colsample_bytree=0.5` |
| CivilComments scores | Average accuracy, worst-group accuracy over the 16 groups of WILDS, Macro-F1 |
| Seeds | 0 to 4 |

## Fidelity and replay of the image versions, `native/fidelity.py` and `native/replay_versions.py`

| Parameter | Value |
|---|---|
| Features | CLIP ViT-B/32 image features |
| Fidelity scores | Precision, recall, density and coverage of 10000 synthetic images against 10000 real training images, $k=5$ |
| Reference level | Real test images against the same real images |
| Replay | The test images of the saved predictions split once per seed into a validation part and a test part, permutation from `default_rng(seed + 900)`, nested validation sizes as in the CleanML track |

## Static scores of the native tracks, `native/static_native.py` and `corpus/static_corpus.py`

| Parameter | Value |
|---|---|
| Score | $P=\tfrac{100}{8}(8-\sum_i r_i)$ over the rates $r_i$ that can be measured on the dataset, the formula of GB/T 48562-2026 on the data quality dimension |
| CleanML tables | One rate per error type, records with a missing value, records removed by the deduplicated version, records with a cell changed by the inconsistency repair, and records with a cell changed by an outlier detector, the median over the detectors |
| Corpora | Share of documents that fail the rule filter and share of documents in a near-duplicate group, on all documents of the shards |
| Image-text pools | One minus the consistency sub-indicator of the whole pool |
| Synthetic images | One minus the precision of `native/fidelity.py` |
| TEP, large | Missing cells, non-finite cells and duplicate records |
| Camelyon17, CivilComments | Missing labels, labels outside the range and duplicate items |

## Detectors, `study/run_detectors.py`

The docstring of the script gives the detectors D1 to D6, the sample sizes 50, 100, 200, 400 and 800 and the four estimates of an error rate.

## Language model baselines, `llm/`

[llm/README.md](../llm/README.md) gives the choices that affect the numbers and the commands.

## Job runner, `study/gpu_pool.py`

| Parameter | Value |
|---|---|
| Jobs per card | `--per-gpu` 6 |
| Cards | `--gpus` as a comma separated list |
| Start condition | Memory of other processes on the card below `--foreign-mib` 46000 |
| Resume | Finished job ids are read from the result file |

## Auxiliary runner with injected defects, `study/run_study.py`, `study/prepare_data.py` and `study/embed_cifar.py`

| Parameter | Value |
|---|---|
| Seeds | `--seeds` 20, split from `default_rng(seed + 1000)` |
| Defects | `missing`, `format`, `value`, `duplicate`, `label`, `coverage`, `scale` and `contamination` at the rates 0.05, 0.10, 0.20, 0.30 and 0.40 |
| Pairs of defects | Pairs of `missing`, `value`, `duplicate`, `label`, `coverage` and `scale` at 0.10 and 0.20 |
| Caps | 8000 training records and 4000 test records |
| Models | The classification models of the tables, and `Ridge(alpha=1.0)`, the same forest and boosting regressors, 15 neighbors and `MLPRegressor((64,), max_iter=60)` for regression |
| AG News encoding | 40000 training documents, TF-IDF with 50000 features, truncated SVD to 300 dimensions, corruption that removes half of the tokens |
| CIFAR-10 features | CLIP ViT-B/32, corruption by Gaussian noise of standard deviation 40 and a box blur of size 3 |

## Derived values of the survey, `hqd_assess` and `analysis/make_numbers.py`

| Parameter | Value |
|---|---|
| Pass line | 90 points |
| Sample size example | $p=0.9$, $h=0.02$, 95 percent confidence |
| Sampling standard example | 4000 records and a threshold of 96 percent |
| Proposition 3 example | $q=q_j=0.1$, $N=20$, $\delta=0.05$, $\gamma=0.02$ in both parts |
| Weight perturbation in the judgement | 10 percent of each weight, `weight_perturbation` of `judge` |
| Bootstrap resamples in the package | 1000 in `calibrate_error_rate`, 200 in `select_version` and `bootstrap_metric_interval` |

## Stored runs

The job lists above define the grids. The table gives the runs that each result file holds, from `python analysis/stored_runs.py`. `python analysis/stored_runs.py --update docs/parameters.md` rewrites it.

<!-- STORED RUNS BEGIN -->
| File | Scored runs | Seeds |
|---|---|---|
| `case_tep.json` | 400 | 0 to 19 |
| `results_cleanml.jsonl` | 3175 | 0 to 4 |
| `results_corpus.jsonl` | 394 | 0 to 4 |
| `results_datacomp.jsonl` | 215 | 0 to 4 |
| `results_decision.jsonl` | 8100 | 0 to 9 |
| `results_decision_fine.jsonl` | 8100 | 0 to 9 |
| `results_fair.json` | 40 |  |
| `results_finite_population.jsonl` | 1528 | 0 |
| `results_llm_labels.jsonl` | 8 |  |
| `results_llm_tables.jsonl` | 550 | 0 to 9 |
| `results_mirflickr.jsonl` | 215 | 0 to 4 |
| `results_multimodal.jsonl` | 820 | 0 to 19 |
| `results_multimodal_fine.jsonl` | 133 | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19 |
| `results_random_delete.jsonl` | 4800 | 0 to 19 |
| `results_real_tables.jsonl` | 25000 | 0 to 19 |
| `results_rieth.json` | 90 | 0 to 4 |
| `results_rieth_models.jsonl` | 280 | 0 to 4 |
| `results_synth.jsonl` | 130 | 0 to 4 |
| `results_tables_fine.jsonl` | 8800 | 0 to 9 |
| `results_taxi.json` | 48 | 0, 1 |
| `results_taxi_100k.json` | 48 | 0, 1 |
| `results_taxi_monthly.json` | 288 | 0, 1 |
| `results_taxi_sizes.json` | 192 | 0, 1 |
| `results_tep_windows.jsonl` | 400 | 0 to 4 |
| `results_text.jsonl` | 143 | 0 to 4 |
| `results_text_more.jsonl` | 18 | 0 to 4 |
| `results_versions.jsonl` | 470 | 0 to 9 |
| `results_vision.jsonl` | 780 | 0 to 9 |
| `results_vision_fine.jsonl` | 50 | 0 to 4 |
| `results_wilds.jsonl` | 15 | 0 to 4 |
<!-- STORED RUNS END -->
