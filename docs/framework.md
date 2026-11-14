# The graded assessment framework

This page states the framework of the survey *A survey on task-oriented assessment of multimodal high-quality datasets*. It covers the quality claim, the five assessment levels, the three dimensions of the standard, the five method families and the six-stage procedure with its algorithms. The package `hqd_assess` implements these parts, and every function named below is in that package. The survey describes the procedure in running text, and this page writes the steps as Algorithms 1 to 6.

## What is assessed

A dataset holds records, and it also holds the choice of samples, the labels, the split, the version and the documents. Quality problems sit on four kinds of objects, and each kind needs its own check.

| Object | Questions | Example on the Tennessee Eastman benchmark |
|---|---|---|
| Single records | Format, values, missing cells | One file of the 44 has rows and columns swapped |
| The dataset as a whole | Distribution, representativeness, duplicates, scale | The files carry no operating mode, while the process defines six |
| Labels and metadata | Reliability of annotation, agreement of documents and data | File names label the rows before the fault onset as faulty |
| Versions and splits | Independence of training and test data, validity after a change | A random window split puts rows of every test window into training |

Quality assessment produces a conclusion about a dataset and leaves the data unchanged. Cleaning, selection and augmentation change the data and can supply signals for an assessment. Model benchmarks fix the data and compare models, and dataset benchmarks fix the model recipe and change the data, so the second kind serves as an experimental tool for assessment. Compliance checks decide whether data meets legal and safety requirements, and their result is a precondition of using the data, so it does not offset other dimensions.

## Definitions

This section gives Definitions 1 to 10 of the survey and keeps their numbers.

**Definition 1, data version.** A data version $D=(R,L,M,\Pi)$ consists of records $R$, labels $L$, metadata and documents $M$, and a split $\Pi$. In the package this is `DatasetVersion`.

**Definition 2, usage context.** A usage context $U=(T,\mathcal{F},S,B)$ consists of the task with its metric and reference version $T$, the downstream model class with its training procedure $\mathcal{F}$, the target scene $S$, and the available budget $B$. The training procedure includes preprocessing and the choice of hyperparameters. In the package this is `UsageContext`.

**Definition 3, task utility.** Given a usage context $U$, let $f_{D,\xi}$ be the model trained by $\mathcal{F}$ on the data version $D$, where $\xi$ is the randomness of training, and let $\mathrm{Perf}(f;\mathcal{P}_S)$ be the metric of a model $f$ on the target scene distribution $\mathcal{P}_S$, with larger values better and error and loss metrics negated. The task utility of $D$ is $\mathrm{Util}_U(D)=\mathbb{E}_{\xi}\mathrm{Perf}(f_{D,\xi};\mathcal{P}_S)$, and its gain over a reference version $D_0$ is $\Delta_U(D,D_0)=\mathrm{Util}_U(D)-\mathrm{Util}_U(D_0)$. A metric such as accuracy is the expectation of a single-record score on $\mathcal{P}_S$, and Macro-F1 is computed from the population confusion matrix on $\mathcal{P}_S$. In the package the gain is `task_gain`.

**Definition 4, quality claim.** A quality claim is the six-tuple $c=\langle o,d,m,\hat{v},l,U\rangle$ of assessment object, quality dimension, measure, estimate with uncertainty, assessment level and usage context. In the package these are `QualityClaim` and `Estimate`. The definition borrows the concept of a data quality claim from ISO/IEC 5259-3 and writes the level and the usage context into the claim.

**Definition 5, assessment levels.** The level is determined by how a quality claim is obtained, and it is independent of the quality dimension.

| Level | Name | What is done | Conclusion it supports |
|---|---|---|---|
| L1 | Declared | Only the statements of the provider in the metadata and documents are read | The provider claims to cover a condition |
| L2 | Matched | Statements or rules are compared item by item with the records, labels, logs and data lineage, with the result match or mismatch for each item | The documents agree with the data |
| L3 | Measured | A defined estimator gives the value for the whole assessment object with its uncertainty | The error rate lies in an interval |
| L4 | Task-tested | A given model is trained on the dataset and task utility is measured on held-out data of the target scene | The model reaches a score in the given scene |
| L5 | Interventional | One aspect of the data changes while the other conditions stay fixed, and the task utilities are compared | The change produces a gain |

In the package the levels are `AssessmentLevel.L1` to `AssessmentLevel.L5`. A higher level does not replace a lower one, because a model score says nothing about a license. A score without uncertainty and usage context cannot be compared across datasets, institutions or versions.

The following claim comes from the running example, and `examples/tep_assessment.py` prints it. The object is the classic TEP test set, the dimension is annotation accuracy, the measure is the accuracy of labels taken from file names, the estimate is 0.841 from a full scan, the level is L3, and the usage context is fault diagnosis with Macro-F1, a split by run as reference, the random forest of the example and the base operating mode.

**Definition 6, static indicators and static quality score.** Quantities that can be computed from the data and documents alone, without training a downstream model, are static indicators, and they do not read the usage context. A ratio sub-indicator $s_i\in[0,1]$ is the share of records that meet requirement $i$ among all records. GB/T 48562-2026 computes a score $P$ out of 100 per dimension, and the survey calls the equal-weight score $P=\tfrac{100}{8}\sum_{i=1}^{8}s_i$ of the 8 ratio sub-indicators of the data quality dimension the static quality score. In the package these are `dimension_score` and `data_quality_score`.

**Definition 7, error rate and its estimates.** A record that fails a requirement is wrong, $Y=1$, and otherwise $Y=0$. The share $\pi$ of wrong records among all records is the error rate, and $\pi=1-s_i$ for the ratio sub-indicator $i$, likewise when the statistical unit is a cell or a sample. A detector flags each record with $H\in\{0,1\}$, and a scan of all records gives the flagged share $\hat{p}$. The detection rate $\phi_1$ is the share of wrong records that are flagged, the false alarm rate $\phi_0$ is the share of correct records that are flagged, and $\lambda=\phi_1-\phi_0$. Determining $Y$ on $n$ records drawn at random from all records, called the labelled sample below, gives three estimates of $\pi$. The direct estimate is the share of wrong records in the sample. The post-stratified estimate splits the sample into two strata by the flags and is $\hat{p}\bar{Y}_{H=1}+(1-\hat{p})\bar{Y}_{H=0}$, where $\bar{Y}_{H=1}$ and $\bar{Y}_{H=0}$ are the shares of wrong records among the flagged and the unflagged records of the sample. The corrected estimate accounts for the errors of the detector and is $\hat{\pi}=\tfrac{\hat{p}-\phi_0}{\phi_1-\phi_0}$, with the detection rate and the false alarm rate estimated on the labelled sample when they are unknown.

**Definition 8, version selection rules.** The candidate versions $D_j$, $j\in\mathcal{J}$, are the data versions to choose from, and they usually include the reference version $D_0$ and versions obtained from it by repair, deletion or filtering. The static rule reads only static indicators and chooses the version with the fewest errors. The task test trains a model on each candidate version and chooses the version with the highest performance on $n$ held-out validation records. Best in hindsight chooses the version with the highest test performance, and the version that any rule chooses has a test performance no higher than that of this version. The loss of a rule is the task utility of a best version minus that of the chosen version. In the package these are `static_rule`, `select_version` and `hindsight_best`.

**Definition 9, interaction.** Let the version before the changes be the reference version $D_0$, let $D_i$ result from the $i$-th change alone, for example repairing only the $i$-th error type, and let $D_{\mathrm{all}}$ result from all changes together. The interaction $I=\Delta_U(D_{\mathrm{all}},D_0)-\sum_i\Delta_U(D_i,D_0)$ is the gain of the joint change minus the sum of the gains of the single changes. It is called synergy for $I>0$ and antagonism for $I<0$. In the package this is `interaction_quantity`.

**Definition 10, reference corpus.** The task utility of a language model corpus is the negative of the mean negative log-likelihood per token that the trained model gives on held-out documents, called the held-out loss per token. The corpus of these held-out documents is the reference corpus, and its distribution is the target scene of Definition 3. When filters of a corpus are compared, the reference version is a corpus drawn at random under the same token budget.

## The three dimensions of the standard

GB/T 48562-2026 scores a dataset on three dimensions with a pass line of 90 points. The codes, formulas and decision rules follow the public draft and the technical document TC609-5-2025-04. [standards.md](standards.md) maps all 17 sub-indicators to levels.

| Dimension | Codes | Items | Definition of the ratio | Level of the calculation |
|---|---|---|---|---|
| Documentation | 0101 to 0104 | 4 | Aspects that meet the requirement against aspects required | L1 |
| Data quality | 0201 to 0208 | 8 | Records that meet the requirement against all records | L1 to L3 |
| Model application | 0301 to 0305 | 5 | Mixed, with model fit $s_m=\tfrac{a}{a^{*}}$ for performance $a$ and expected performance $a^{*}$, or $\tfrac{a^{*}}{a}$ when lower is better | L2 to L4 |

A dimension score is $P=100\sum_i \omega_i s_i$ with weights $\omega_i$, and the standard recommends equal weights. The model application score is $P=100F_m\bar{s}$, where $F_m=\mathrm{e}^{s_m-1}$ for $s_m<1$, $F_m=1$ otherwise, and $\bar{s}$ is the mean of the other items. In the package these are `dimension_score`, `model_fit_value`, `model_fit_factor` and `model_application_score`.

The calculation of $s_m$ needs a baseline, a model class and an evaluation protocol, and each choice can change the verdict. On classic TEP with an expected performance of 0.80 and the other items at full marks, a split by run gives $s_m=0.871$ and 87.9 points, which fails, and a random window split gives $s_m>1$ and 100 points, which passes. For any expected performance from 0.78 to 0.97 the split alone decides the verdict, as `expected_performance_band` computes.

The mapping shows three links for further work. Documentation items stay at L1 and need a comparison with the data. Ratio items have no sampling plan, sample size or interval. Cleanliness, diversity, scale and timeliness do not read task results, and model fit stops at L4, so it does not show which change of the data caused a score.

The composite score lets sub-indicators offset each other. Let a dimension hold $K$ sub-indicators $s_i\in[0,1]$ with the equal-weight score $P=\tfrac{100}{K}\sum_{i=1}^K s_i$. Since $P=100-\tfrac{100}{K}\sum_{i=1}^K(1-s_i)$, the score reaches 90 points if and only if $\sum_{i=1}^K(1-s_i)\le 0.1K$, and with the other sub-indicators at full marks a single sub-indicator can fall to $\max(0,1-0.1K)$. Section 3.2 of the survey gives this computation.

GB/T 48562-2026 recommends equal weights. The floor of a single sub-indicator is then 0.2 for the $K=8$ items of data quality and 0.6 for the $K=4$ items of documentation. With adjustable weights the floor of item $i$ with weight $\omega_i$ is $\max(0,1-\tfrac{0.1}{\omega_i})$, and an item with a weight of at most 0.1 can reach 0 while the dimension passes. Safety conformity or content authenticity can therefore fall to 0.2 at 90 points. Model fit cannot fall below about 0.895, because it enters the exponential factor of the model application score. Since $P\le 100\,\mathrm{e}^{s_m-1}$, that dimension fails for $s_m<1+\ln 0.9$ whatever values the other items take. The survey makes safety, license and authenticity veto items, which enter Algorithm 5 as the set $\mathcal{A}$, and computes the composite score on the other items above their own thresholds. The functions are `compensation_budget`, `within_compensation_budget`, `single_item_floor`, `weighted_item_floor` and `model_fit_threshold`.

## The five method families

| Family | Representative work | Truth needed | Level | Fails when |
|---|---|---|---|---|
| Rule constraints | CFD, denial constraints, time series dependencies, Auto-Test | Given or mined rules | L2, L3 | Errors lie outside the rules, or violations are not errors |
| Statistical sampling | SampleClean, DQM, PPI, DSL | Probability sample labels | L3 | Samples are not random, errors are rare, or fine precision is too costly |
| Learned models | cleanlab, ZeroED, Raha, DemandClean | Few labels, reference data | L3 | Scores change with the model and the reference |
| Task valuation | CleanML, CPClean, Shapley values, TRAK | Held-out data and retraining | L4, L5 | Cost is high, noise dominates, or the result binds to a model class |
| Human judgment | Truth inference, language-model labels, $\kappa$ and $\alpha$ | Experts and source records | L2, L3 | Reviewers disagree or drift, and cost grows with scale |

The choice depends on the available truth and the budget. Clear business rules come first, because a full scan is cheap and its result is definite. A review budget goes to a stratified sample that calibrates rules and proxy metrics and turns ratio items into estimates with intervals. Without rules and a review budget the assessment relies on proxy metrics and has to state the reference and the model. The question whether data supports a task needs a task test and a check of the evaluation protocol. The question which change works needs an intervention.

### Formulas for sampling

Estimating a proportion $p$ to a half-width $h$ needs $n=\tfrac{z^2p(1-p)}{h^2}$ records, which gives 865 records for $p=0.9$, $h=0.02$ and 95 percent confidence (`proportion_sample_size`). At 4000 records and 96 percent the half-width is about 0.6 points (`proportion_half_width`). When a sample of $n$ records holds no error, the 95 percent upper bound on the error rate is about $\tfrac{3}{n}$ (`zero_error_upper_bound`). Intervals use the Wilson or Jeffreys form (`wilson_interval`, `jeffreys_interval`), because the coverage of the Wald interval is unstable at common sample sizes.

A detector that flags a share $\hat{p}$ of all records, a share $\phi_1$ of the wrong records and a share $\phi_0$ of the correct records gives the corrected estimate $\hat{\pi}=\tfrac{\hat{p}-\phi_0}{\phi_1-\phi_0}$ of Rogan and Gladen. With the sensitivity $s_e=\phi_1$ and the specificity $s_p=1-\phi_0$ it reads $\tfrac{\hat{p}+s_p-1}{s_e+s_p-1}$ (`rogan_gladen`). When $\lambda$ is near 0, the detector flags wrong and correct records at similar rates, its flags are almost unrelated to the true errors, and the denominator of the corrected estimate is also near 0. The post-stratified estimate uses the flags only to form two groups and does not divide by $\lambda$. Algorithm 1 computes the corrected estimate with its interval, and `detector_interval_from_counts` in `hqd_assess.finite_population` computes the post-stratified estimate with a finite-population interval. Finding 6 of the survey compares the errors of the two uses.

### Non-additivity of the dimensions

The effects of quality dimensions do not add up, so an attribution by dimension has no unique decomposition. Take the training mean as a constant predictor with squared loss on true values that are all 0. Eight records are correct, record $a$ holds $+2$ outside the value range, and records $b$ and $c$ hold $-1$ inside the range. The mean is 0 and the risk is 0. Repairing $a$ lowers the share of wrong records from $\tfrac{3}{11}$ to $\tfrac{2}{11}$ and raises the risk to $\tfrac{4}{121}$. Repairing $b$ and $c$ also raises the risk to $\tfrac{4}{121}$, and repairing all three returns it to 0. The single effects sum to $\tfrac{8}{121}$, while the joint effect is 0 (`additivity_counterexample`).

## The six-stage procedure

The result of each stage decides what the next stage does. When the data or the usage context changes, the assessment returns to the affected stage.

1. **State the purpose and list the claims.** Compliance decides whether the data may be used, and license, source and safety need L2 or L3 with non-compensable thresholds. Selection decides which candidate fits, and the candidates need L4 on one held-out set and one model class. Improvement decides which change works and needs L5. The output is a list of claims with statistical units and lowest levels (`minimum_level`).
2. **Fix the materials and the sampling plan before reading results.** The data version, split rule, documents and logs, strata and sample sizes are fixed. The formulas above give the sizes, and when the budget falls short the report states the precision that it can guarantee.
3. **Measure ratios with uncertainty and check the split.** Format, missing cells and duplicates get a full scan. Items that need human truth use stratified sampling and Wilson intervals. Items read from a detector use the direct, the post-stratified or the corrected estimate, and Algorithm 1 computes the corrected one. The stage fixes the scope of each ratio, separates not applicable from missing, and runs Algorithm 2.
4. **Test the task.** Algorithm 3 chooses a version and reports its gain, with a comparison across model classes. Benchmarks and LLM judges are checked for data leakage and bias first. A claim about one change alters only that change, and several changes go through Algorithm 4.
5. **Judge.** Algorithm 5 gives the verdict. Veto items stay out of the composite score, because in an equal-weight score other sub-indicators offset a defect of one, as the floors under [The three dimensions of the standard](#the-three-dimensions-of-the-standard) show. The report lists the veto items, the claims per dimension, and the composite score with its sensitivity to the weights.
6. **Record the conditions for re-assessment.** Algorithm 6 names the stages to redo after a change. When the target scene changes, stages 4 and 5 are redone on data of the new scene.

The lowest level also follows the type of a dataset in GB/T 48561-2026. A general dataset serves many downstream uses and can rest on L3 with L4 on representative benchmarks. An industry general dataset needs L4 on industry tasks with a split by business unit. An industry specific dataset and a high-risk scene need L2 for sources, L3 by expert samples for labels, L4 on the target scene for model fit, and L5 for a claim that a data change produces a gain.

### Algorithm 1, sampling calibration of a ratio (`calibrate_error_rate`)

Input is a data version $D$, a detector $g$, an audit size $n$ and strata. Output is the corrected error rate $\hat{\pi}$ with its interval.

1. Run $g$ on all records of $D$ to get the positive share $\hat{p}$.
2. Draw $n$ records by the strata and determine their truth by people or source records.
3. Set $s_e$ to the share of truly wrong records that $g$ flags.
4. Set $s_p$ to the share of truly correct records that $g$ passes.
5. If $s_e+s_p\le 1$, the detector has no discriminating power, and the algorithm returns the Wilson interval of the audit error rate.
6. Set $\hat{\pi}=\tfrac{\hat{p}+s_p-1}{s_e+s_p-1}$, cut to $[0,1]$.
7. Resample the audit and all records to get the interval of $\hat{\pi}$.

### Algorithm 2, independence of the split (`check_split_independence`)

Input is a split $\Pi$, a set of group keys $\mathcal{G}$ such as run, time, source and near-duplicate group, a model class $\mathcal{F}$ and a metric $\mu$. Output is the overlap $\rho_b$ and the score gap $\eta_b$ of each key.

1. For each key $b$, set $\rho_b$ to the share of test items that share a value of $b$ with a training item.
2. If $\rho_b>0$, split again by $b$ into $\Pi_b$, train the models of $\mathcal{F}$ on $\Pi$ and on $\Pi_b$, and set $\eta_b=\mu(\Pi)-\mu(\Pi_b)$.
3. Report the score on the grouped split in the claim.

`group_overlap` and `window_row_overlap` compute the overlaps.

### Algorithm 3, choice of a version by a task test (`select_version`)

Input is a set of candidate versions $\mathcal{V}\ni D_0$, a model class $\mathcal{F}$, a held-out set $\mathcal{E}$ and a validation size $n$. Output is the selected version $D^{*}$, its gain $\hat{\Delta}$ and the precision $\gamma(n)$.

1. Split $\mathcal{E}$ at random into a validation part $\mathcal{E}_v$ and a test part $\mathcal{E}_t$.
2. For each $D\in\mathcal{V}$, train a model $f_D$ of $\mathcal{F}$ and set $\hat{u}_D$ to its score on $n$ records of $\mathcal{E}_v$.
3. Set $D^{*}=\arg\max_{D\in\mathcal{V}}\hat{u}_D$.
4. Set $\hat{\Delta}$ to the score difference of $f_{D^{*}}$ and $f_{D_0}$ on $\mathcal{E}_t$, with its interval.
5. Take $\gamma(n)$ from the Bernstein inequality with a union bound over the candidates when the metric is a mean over records (`resolution_for_sample_size`), and from the bootstrap otherwise (`gain_precision_bootstrap`).

`static_rule` is the comparison rule that reads only static indicators, and it picks the version with the fewest errors. `hindsight_best` picks the version with the best test score and is an upper reference for any rule.

### Algorithm 4, interaction of quality dimensions (`interaction_test`)

Input is a raw table $D_0$, a set of error types $\mathcal{K}$, a model $f$, a set of repeated runs $\mathcal{S}$ and a level $\alpha$. Output is the mean interaction $\bar{I}$ and a verdict.

1. For each run $s$, let $y_0$ be the task score on $D_0$, $y_k$ the score after repairing only type $k$, and $y_J$ the score after repairing all types.
2. Set $I_s=(y_J-y_0)-\sum_{k\in\mathcal{K}}(y_k-y_0)$.
3. Run a two-sided t test on $\{I_s\}$ and correct all comparisons together by Benjamini and Yekutieli to get $p_{\mathrm{BY}}$.
4. If $p_{\mathrm{BY}}<\alpha$, the verdict is synergy for $\bar{I}>0$ and antagonism for $\bar{I}<0$. Otherwise no interaction is detected.

`paired_flags` compares an action with the raw table in the same way and flags it as positive, not significant or negative, following CleanML.

### Algorithm 5, graded judgement (`judge`)

Input is a set of veto items $\mathcal{A}$, a set of other claims $\mathcal{C}$, and for each claim $c$ a required level $l_c^{*}$, a threshold $\tau_c$ and an interval lower bound $\hat{v}_c^{-}$. Output is the verdict, the list of claims and the composite score $P$.

1. For each $c\in\mathcal{A}$, if $l_c<l_c^{*}$ or $\hat{v}_c^{-}<\tau_c$, return a failure and report $c$, because the condition cannot be compensated.
2. For each $c\in\mathcal{C}$, record $l_c<l_c^{*}$ as an insufficient level and $\hat{v}_c^{-}<\tau_c$ as a value below the threshold.
3. Compute $P$ and its range under a perturbation of the weights (`score_range`).
4. Return a pass when $\mathcal{C}$ has no such record, with the list of claims and $P$.

The dataset passes under $U$ if and only if all of $\mathcal{A}$ hold and every $c\in\mathcal{C}$ has $l_c\ge l_c^{*}$ and $\hat{v}_c^{-}\ge\tau_c$. Comparing the lower bound with the threshold brings the sample size into the verdict. Under this rule the 4000 records of GB/T 45652-2025 need an observed rate of about 96.6 percent for a lower bound of 96 percent (`minimum_observed_rate`).

### Algorithm 6, re-assessment after a change (`renew`)

Input is the claim list of the last version, a change event $\epsilon$, a monitored series $\{r_t\}$ and a trigger threshold $\tau_r$. Output is the updated claim list and its validity.

1. If $|r_t-r_{t-1}|>\tau_r$, redo stage 4 on recent data (`violation_trigger`).
2. If $\epsilon$ concerns the data version or the scene, redo stages 2 to 5.
3. Otherwise, if $\epsilon$ concerns the labels, redo stages 3 to 5.
4. Otherwise, if $\epsilon$ concerns the downstream model or the reference, redo stages 4 and 5.
5. Return the paired comparison with the last version, the new claim list and its validity.

`stages_to_redo` gives the stages of a change event. The taxi experiment supports rule violation rates as a low-cost trigger, and `trigger_counts` reproduces that count.
