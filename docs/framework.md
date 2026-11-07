# The graded assessment framework

This page states the framework of the survey *A survey on the assessment of high-quality datasets*. It covers the quality claim, the five assessment levels, the three dimensions of the standard, the five method families, the six-stage procedure with its algorithms, and the three propositions. The package `hqd_assess` implements these parts, and every function named below is in that package. The survey describes the procedure in running text, and this page writes the steps as Algorithms 1 to 6.

## What is assessed

A dataset holds records, and it also holds the choice of samples, the labels, the split, the version and the documents. Quality problems sit on four kinds of objects, and each kind needs its own check.

| Object | Questions | Example on the Tennessee Eastman benchmark |
|---|---|---|
| Single records | Format, values, missing cells | One file of the 44 has rows and columns swapped |
| The dataset as a whole | Distribution, representativeness, duplicates, scale | The files carry no operating mode, while the process defines six |
| Labels and metadata | Reliability of annotation, agreement of documents and data | File names label the rows before the fault onset as faulty |
| Versions and splits | Independence of training and test data, validity after a change | A random window split puts rows of every test window into training |

Quality assessment produces a judgement about a dataset and leaves the data unchanged. Cleaning, selection and augmentation change the data and can supply signals for an assessment. Model benchmarks fix the data and compare models, and dataset benchmarks fix the model recipe and change the data, so the second kind serves as an experimental tool for assessment. Compliance checks decide whether data meets legal and safety requirements, and their result acts as an admission condition.

## Definitions

**Definition 1, dataset version and usage context.** A dataset version $D=(R,L,M,\Pi)$ consists of records $R$, labels $L$, metadata and documents $M$, and a split $\Pi$. A usage context $U=(T,\mathcal{F},S,B)$ consists of the task with its metric and reference version $T$, the target model class $\mathcal{F}$, the scene with its risk $S$, and the available budget $B$. In the package these are `DatasetVersion` and `UsageContext`.

**Definition 2, task utility.** Given a usage context $U$, the task utility $\mathrm{Util}_U(D)$ of a version $D$ is the expected metric on the target scene distribution $\mathcal{P}_S$ after training the target model on $D$. The gain over a reference version $D_0$ is $\Delta_U(D,D_0)=\mathrm{Util}_U(D)-\mathrm{Util}_U(D_0)$. In the package this is `task_gain`.

**Definition 3, quality claim.** A quality claim is the six-tuple $c=\langle o,d,m,\hat{v},l,U\rangle$ of assessment object, quality dimension, measure, estimate with uncertainty, assessment level and usage context. In the package these are `QualityClaim` and `Estimate`. The definition borrows the concept of a data quality claim from ISO/IEC 5259-3 and writes the level and the usage context into the claim.

**Definition 4, assessment levels.** The level records how a conclusion is obtained, and it is independent of the quality dimension.

| Level | Name | What is done | Conclusion it supports |
|---|---|---|---|
| L1 | Declared | The provider states a property in its documents | The provider claims to cover a condition |
| L2 | Matched | Statements or rules are compared with the data, logs and lineage, with the result match or mismatch | The documents agree with the data |
| L3 | Measured | A defined estimator is computed on the data with its uncertainty | The error rate lies in an interval |
| L4 | Task-tested | A given model is trained on the dataset and task utility is measured on held-out data | The model reaches a score in the given scene |
| L5 | Interventional | One aspect of the data changes while the other conditions stay fixed, and the downstream results are compared | The change produces a gain |

In the package the levels are `AssessmentLevel.L1` to `AssessmentLevel.L5`. A higher level does not replace a lower one, because a model score says nothing about a license. A score without uncertainty and usage context cannot be compared across datasets, institutions or versions.

The following claim comes from the running example, and `examples/tep_assessment.py` prints it. The object is the classic TEP test set, the dimension is annotation accuracy, the measure is the accuracy of labels taken from file names, the estimate is 0.841 from a full scan, the level is L3, and the usage context is fault diagnosis with Macro-F1, a split by run as reference, the random forest of the example and the base operating mode.

## The three dimensions of the standard

GB/T 48562-2026 scores a dataset on three dimensions with a pass line of 90 points. The codes, formulas and decision rules follow the public draft and the technical document TC609-5-2025-04. [standards.md](standards.md) maps all 17 sub-indicators to levels.

| Dimension | Codes | Items | Definition of the ratio | Level of the calculation |
|---|---|---|---|---|
| Documentation | 0101 to 0104 | 4 | Aspects that meet the requirement against aspects required | L1 |
| Data quality | 0201 to 0208 | 8 | Records that meet the requirement against all records | L1 to L3 |
| Model application | 0301 to 0305 | 5 | Mixed, with model fit $s_m=\tfrac{a}{a^{*}}$ for performance $a$ and expected performance $a^{*}$, or $\tfrac{a^{*}}{a}$ when lower is better | L2 to L4 |

A dimension score is $P=100\sum_i w_i s_i$, and the standard recommends equal weights. The model application score is $P=100F_m\bar{s}$, where $F_m=\mathrm{e}^{s_m-1}$ for $s_m<1$, $F_m=1$ otherwise, and $\bar{s}$ is the mean of the other items. In the package these are `dimension_score`, `model_fit_value`, `model_fit_factor` and `model_application_score`.

The calculation of $s_m$ needs a baseline, a model class and an evaluation protocol, and each choice can change the verdict. On classic TEP with an expected performance of 0.80 and the other items at full marks, a split by run gives $s_m=0.871$ and 87.9 points, which fails, and a random window split gives $s_m>1$ and 100 points, which passes. For any expected performance from 0.78 to 0.97 the split alone decides the verdict, as `expected_performance_band` computes.

The mapping shows three links for further work. Documentation items stay at L1 and need a comparison with the data. Ratio items have no sampling plan, sample size or interval. Cleanliness, diversity, scale and timeliness do not read task results, and model fit stops at L4, so it does not show which change of the data caused a score.

## The five method families

| Family | Representative work | Truth needed | Level | Fails when |
|---|---|---|---|---|
| Rule constraints | CFD, denial constraints, time series dependencies, Auto-Test | Given or mined rules | L2, L3 | Errors lie outside the rules, or violations are not errors |
| Statistical sampling | SampleClean, DQM, PPI, DSL | Probability sample labels | L3 | Samples are not random, errors are rare, or fine precision is too costly |
| Learned models | cleanlab, ZeroED, Raha, DemandClean | Few labels, reference data | L3 | Scores change with the model and the reference |
| Task valuation | CleanML, CPClean, Shapley values, TRAK | Held-out data and retraining | L4, L5 | Cost is high, noise dominates, or the result binds to a model class |
| Human judgment | Truth inference, language-model labels, $\kappa$ and $\alpha$ | Experts and source records | L2, L3 | Reviewers disagree or drift, and cost grows with scale |

The choice depends on the available truth and the budget. Clear business rules come first, because a full scan is cheap and its result is definite. A review budget goes to a stratified sample that calibrates rule and model signals and turns ratio items into estimates with intervals. Without rules and a review budget the assessment relies on model signals and has to state the reference and the model. The question whether data supports a task needs a task test and a check of the evaluation protocol. The question which change works needs an intervention.

### Formulas for sampling

Estimating a proportion $p$ to a half-width $h$ needs $n=\tfrac{z^2p(1-p)}{h^2}$ records, which gives 865 records for $p=0.9$, $h=0.02$ and 95 percent confidence (`proportion_sample_size`). At 4000 records and 96 percent the half-width is about 0.6 points (`proportion_half_width`). When a sample of $n$ records holds no error, the 95 percent upper bound on the error rate is about $\tfrac{3}{n}$ (`zero_error_upper_bound`). Intervals use the Wilson or Jeffreys form (`wilson_interval`, `jeffreys_interval`), because the coverage of the Wald interval is unstable at common sample sizes.

A detector with positive share $\hat{p}$, sensitivity $s_e$ and specificity $s_p$ gives the corrected error rate $\hat{\pi}=\tfrac{\hat{p}+s_p-1}{s_e+s_p-1}$ of Rogan and Gladen (`rogan_gladen`).

### Non-additivity of the dimensions

The effects of quality dimensions do not add up, so an attribution by dimension has no unique decomposition. Take the training mean as a constant predictor with squared loss on true values that are all 0. Eight records are correct, record $a$ holds $+2$ outside the value range, and records $b$ and $c$ hold $-1$ inside the range. The mean is 0 and the risk is 0. Repairing $a$ lowers the share of wrong records from $\tfrac{3}{11}$ to $\tfrac{2}{11}$ and raises the risk to $\tfrac{4}{121}$. Repairing $b$ and $c$ also raises the risk to $\tfrac{4}{121}$, and repairing all three returns it to 0. The single effects sum to $\tfrac{8}{121}$, while the joint effect is 0 (`additivity_counterexample`).

## The six-stage procedure

The result of each stage decides what the next stage does. When the data or the usage context changes, the assessment returns to the affected stage.

1. **State the purpose and list the claims.** Admission decides whether the data may be used, and license, source and safety need L2 or L3 with non-compensable thresholds. Selection decides which candidate fits, and the candidates need L4 on one held-out set and one model class. Improvement decides which change works and needs L5. The output is a list of claims with statistical units and lowest levels (`minimum_level`).
2. **Fix the materials and the sampling plan before reading results.** The data version, split rule, documents and logs, strata and sample sizes are fixed. The formulas above and Proposition 3 give the sizes, and when the budget falls short the report states the precision that it can guarantee.
3. **Measure ratios with uncertainty and check the split.** Format, missing cells and duplicates get a full scan. Items that need human truth use stratified sampling and Wilson intervals. Items read from a detector use Algorithm 1. The stage fixes the scope of each ratio, separates not applicable from missing, and runs Algorithm 2.
4. **Test the task.** Algorithm 3 chooses a version and reports its gain, with a comparison across model classes. Benchmarks and judge models are checked for data leakage and bias first. A claim about one change alters only that change, and several changes go through Algorithm 4.
5. **Judge.** Algorithm 5 gives the verdict. The report lists the admission conditions, the claims per dimension, and the composite score with its sensitivity to the weights.
6. **Record the conditions for re-assessment.** Algorithm 6 names the stages to redo after a change.

The lowest level also follows the type of a dataset in GB/T 48561-2026. A general dataset serves many downstream uses and can rest on L3 with L4 on representative benchmarks. An industry general dataset needs L4 on industry tasks with a split by business unit. An industry specific dataset and a high-risk scene need L2 for sources, L3 by expert samples for labels, L4 on the target scene for model fit, and L5 for a claim that a data change produces a gain.

### Algorithm 1, sampling calibration of a ratio (`calibrate_error_rate`)

Input is a dataset version $D$, a detector $g$, an audit size $n$ and strata. Output is the corrected error rate $\hat{\pi}$ with its interval.

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
5. Take $\gamma(n)$ from Proposition 3 when the metric is a mean over records, and from the bootstrap otherwise.

`static_rule` is the comparison rule that reads only static scores, and it picks the version with the fewest errors. `hindsight_best` picks the version with the best test score and is an upper reference for any rule.

### Algorithm 4, interaction of quality dimensions (`interaction_test`)

Input is a raw table $D_0$, a set of error types $\mathcal{K}$, a model $f$, a set of repeated runs $\mathcal{S}$ and a level $\alpha$. Output is the mean interaction $\bar{I}$ and a verdict.

1. For each run $s$, let $y_0$ be the task score on $D_0$, $y_k$ the score after repairing only type $k$, and $y_J$ the score after repairing all types.
2. Set $I_s=(y_J-y_0)-\sum_{k\in\mathcal{K}}(y_k-y_0)$.
3. Run a two-sided t test on $\{I_s\}$ and correct all comparisons together by Benjamini and Yekutieli to get $p_{\mathrm{BY}}$.
4. If $p_{\mathrm{BY}}<\alpha$, the verdict is synergy for $\bar{I}>0$ and antagonism for $\bar{I}<0$. Otherwise no interaction is detected.

`paired_flags` compares an action with the raw table in the same way and flags it as positive, not significant or negative, following CleanML.

### Algorithm 5, graded judgement (`judge`)

Input is a set of admission claims $\mathcal{A}$, a set of other claims $\mathcal{C}$, and for each claim $c$ a required level $l_c^{*}$, a threshold $\tau_c$ and an interval lower bound $\hat{v}_c^{-}$. Output is the verdict, the list of claims and the composite score $P$.

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
4. Otherwise, if $\epsilon$ concerns the target model or the reference, redo stages 4 and 5.
5. Return the paired comparison with the last version, the new claim list and its validity.

`stages_to_redo` gives the stages of a change event. The taxi experiment supports rule violation rates as a low-cost trigger, and `trigger_counts` reproduces that count.

## The three propositions

**Proposition 1, limit of static decisions.** Let the static observables of versions $D_a,D_b$ not vary across usage contexts $U_1,U_2$, with $\Delta_{U_1}(D_a,D_b)=\gamma>0$ and $\Delta_{U_2}(D_a,D_b)=-\gamma$. Let the loss be the utility gap between the chosen version and the better version under the context. A selection rule that reads only static observables and runs no task test has an expected loss of $\tfrac{\gamma}{2}$ when the two contexts are equally likely.

The rule picks $D_a$ with some probability $p$ that does not depend on the context, so the expected loss is $\tfrac{1}{2}(1-p)\gamma+\tfrac{1}{2}p\gamma=\tfrac{\gamma}{2}$. Static observables are the quantities that can be computed from data and documents without training the target model. The function is `static_rule_expected_loss`.

**Proposition 2, compensation bound of equal-weight scores.** Let a dimension hold $k$ sub-indicators in $[0,1]$ with the equal-weight score $P=\tfrac{100}{k}\sum_{i=1}^k s_i$ and a pass line of 90 points. Then $P\ge 90$ if and only if $\sum_{i=1}^k(1-s_i)\le 0.1k$, and with the other items at full marks a single item can fall to $\max(0,1-0.1k)$. The model application score is $P=100F_m\bar{s}$ with $F_m=\mathrm{e}^{s_m-1}$ for $s_m<1$ and $\bar{s}\le 1$, so for $s_m<1+\ln 0.9\approx 0.895$ the dimension fails for any values of the other items.

The first statement follows from $P=100-\tfrac{100}{k}\sum_{i=1}^k(1-s_i)$, and the second from $P\le 100\,\mathrm{e}^{s_m-1}$. At equal weights the single-item floor is 0.2 for the 8 items of data quality and 0.6 for the 4 items of documentation. With adjustable weights the floor of item $i$ is $\max(0,1-\tfrac{0.1}{w_i})$, and an item with a weight of at most 0.1 can reach 0 while the dimension passes. Safety conformity or content authenticity can therefore fall to 0.2 at 90 points, while model fit cannot fall below about 0.895. The survey treats safety, license and authenticity as non-compensable admission conditions, and computes the composite score on the other items above their own thresholds. The functions are `single_item_floor`, `weighted_item_floor` and `model_fit_threshold`.

**Proposition 3, sample size of task tests.** Let $N$ candidate versions be compared with the reference on $n$ independent held-out records, let the metric be a mean of single-record scores, and let the paired difference per record satisfy $Z\in[-1,1]$ with variance at most $q$. For all $N$ gain estimates to lie within $\tfrac{\gamma}{2}$ of their true values with probability at least $1-\delta$, it suffices that $n\ge\big(\tfrac{8q}{\gamma^2}+\tfrac{8}{3\gamma}\big)\ln\tfrac{2N}{\delta}$.

The proof takes $t=\tfrac{\gamma}{2}$ and $|Z-\mathbb{E}Z|\le 2$. By the Bernstein inequality a single estimate fails with probability at most $2\exp\big(-\tfrac{nt^2}{2q+\frac{4t}{3}}\big)$, and the union bound over $N$ estimates gives the size. The variance bound $q$ can be the share of records on which two versions predict differently. With $q=0.1$, $N=20$ and $\delta=0.05$, about 55300, 14300 and 2500 held-out records suffice for a precision of 1, 2 and 5 points, and for a metric that averages over records the 713 test records of the Flights table reach about 10 points by this bound. The bound is a sufficient condition for the simultaneous guarantee on all candidates, and a choice among versions needs only their order, so the measured gains level off at a few hundred validation records. Macro-F1 and $R^2$ do not average over records, and their precision comes from `gain_precision_bootstrap`. The functions are `task_test_sample_size` and `resolution_for_sample_size`.
