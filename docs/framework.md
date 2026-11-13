# The graded assessment framework

This page states the framework of the survey *A survey on task-oriented assessment of multimodal high-quality datasets*. It covers the quality claim, the five assessment levels, the three dimensions of the standard, the five method families, the six-stage procedure with its algorithms, and the theorems and propositions of the survey with their proofs. The package `hqd_assess` implements these parts, and every function named below is in that package. The survey describes the procedure in running text, and this page writes the steps as Algorithms 1 to 6.

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

This section gives Definitions 1 to 9 of the survey and keeps their numbers.

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

A detector that flags a share $\hat{p}$ of all records, a share $\phi_1$ of the wrong records and a share $\phi_0$ of the correct records gives the corrected estimate $\hat{\pi}=\tfrac{\hat{p}-\phi_0}{\phi_1-\phi_0}$ of Rogan and Gladen. With the sensitivity $s_e=\phi_1$ and the specificity $s_p=1-\phi_0$ it reads $\tfrac{\hat{p}+s_p-1}{s_e+s_p-1}$ (`rogan_gladen`). Post-stratifying the labelled sample by the flags needs no more labels than this estimate for any detector, as the variance comparison under [Theorems and propositions](#theorems-and-propositions) shows.

### Non-additivity of the dimensions

The effects of quality dimensions do not add up, so an attribution by dimension has no unique decomposition. Take the training mean as a constant predictor with squared loss on true values that are all 0. Eight records are correct, record $a$ holds $+2$ outside the value range, and records $b$ and $c$ hold $-1$ inside the range. The mean is 0 and the risk is 0. Repairing $a$ lowers the share of wrong records from $\tfrac{3}{11}$ to $\tfrac{2}{11}$ and raises the risk to $\tfrac{4}{121}$. Repairing $b$ and $c$ also raises the risk to $\tfrac{4}{121}$, and repairing all three returns it to 0. The single effects sum to $\tfrac{8}{121}$, while the joint effect is 0 (`additivity_counterexample`). The example is part (2) of Proposition 2 with the biases $b_1=\tfrac{2}{11}$ and $b_2=-\tfrac{2}{11}$, whose interaction on the utility, the negative risk, is $-2b_1b_2=\tfrac{8}{121}$.

## The six-stage procedure

The result of each stage decides what the next stage does. When the data or the usage context changes, the assessment returns to the affected stage.

1. **State the purpose and list the claims.** Compliance decides whether the data may be used, and license, source and safety need L2 or L3 with non-compensable thresholds. Selection decides which candidate fits, and the candidates need L4 on one held-out set and one model class. Improvement decides which change works and needs L5. The output is a list of claims with statistical units and lowest levels (`minimum_level`).
2. **Fix the materials and the sampling plan before reading results.** The data version, split rule, documents and logs, strata and sample sizes are fixed. The formulas above and Proposition 3 give the sizes, and when the budget falls short the report states the precision that it can guarantee.
3. **Measure ratios with uncertainty and check the split.** Format, missing cells and duplicates get a full scan. Items that need human truth use stratified sampling and Wilson intervals. Items read from a detector use the direct, the post-stratified or the corrected estimate, and Algorithm 1 computes the corrected one. The stage fixes the scope of each ratio, separates not applicable from missing, and runs Algorithm 2.
4. **Test the task.** Algorithm 3 chooses a version and reports its gain, with a comparison across model classes. Benchmarks and LLM judges are checked for data leakage and bias first. A claim about one change alters only that change, and several changes go through Algorithm 4.
5. **Judge.** Algorithm 5 gives the verdict. Veto items stay out of the composite score, because in an equal-weight score other sub-indicators offset a defect of one (Proposition 1). The report lists the veto items, the claims per dimension, and the composite score with its sensitivity to the weights.
6. **Record the conditions for re-assessment.** Algorithm 6 names the stages to redo after a change. When the scene changes, Theorem 2 (2) gives the lower bound $\hat{\Delta}_Q-\varepsilon-2\beta$ of the gain (`shifted_gain_lower_bound`), and a bound below the threshold of a claim means that the claim loses this guarantee and stages 4 and 5 are redone on data of the new scene.

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
5. Take $\gamma(n)$ from Proposition 3 (1) when the metric is a mean over records, and from the bootstrap otherwise.

Proposition 3 (2) bounds the probability that $D^{*}$ falls more than $\gamma$ below the best candidate (`wrong_choice_bound`), and `selection_sample_size` gives a validation size that keeps this probability below $\delta$. `static_rule` is the comparison rule that reads only static indicators, and it picks the version with the fewest errors. `hindsight_best` picks the version with the best test score and is an upper reference for any rule.

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

## Theorems and propositions

The survey proves two theorems and three propositions, and its Section 4.1 compares three estimates of an error rate. They follow with the numbering of the survey, each with its premises, its statement and its proof.

**Theorem 1, decision sufficiency of assessment results.** Let the candidate versions $D_j$ have indices in a finite set $\mathcal{J}$, and let the usage situations $w$ lie in a finite set $\mathcal{W}$. Each situation fixes a usage context $U$, the task utility of $D_j$ in it is $\mu_j(w)=\mathrm{Util}_U(D_j)$, and its set of best versions is $\mathcal{J}^{*}(w)=\arg\max_{j\in\mathcal{J}}\mu_j(w)$. A decision rule reads only the assessment result $E(w)$, so on reading $e$ it cannot tell apart the situations in $\mathcal{W}_e=\{w:E(w)=e\}$. The rule may choose at random, and its loss in a situation is the task utility of a best version minus that of the chosen version.

1. A rule with zero loss in every situation exists if and only if $\bigcap_{w\in\mathcal{W}_e}\mathcal{J}^{*}(w)\neq\varnothing$ for every assessment result $e$.
2. For a choice between $D_a$ and $D_b$ only, let $a_+=\max_{w\in\mathcal{W}_e}\big(\mu_a(w)-\mu_b(w)\big)>0$ and $a_-=-\min_{w\in\mathcal{W}_e}\big(\mu_a(w)-\mu_b(w)\big)>0$. On reading $e$, the smallest worst-case expected loss, called the minimax loss, is $\tfrac{a_+a_-}{a_++a_-}$, and the rule that chooses $D_a$ with probability $\tfrac{a_+}{a_++a_-}$ attains it. Without randomization the smallest value is $\min(a_+,a_-)$.
3. If an assessment result $E'$ is a function of an assessment result $E$, the minimax loss of the rules that read $E$ is at most that of the rules that read $E'$.

*Proof.* (1) If the situations in each $\mathcal{W}_e$ share a best version, the rule that chooses it on reading $e$ has zero loss in every situation. Conversely, if a rule has zero expected loss in every situation, every version that it chooses with positive probability on reading $e$ is a best version in each situation of $\mathcal{W}_e$, so the intersection is not empty. (2) A rule that chooses $D_a$ with probability $u$ has the worst-case expected loss $\max\{(1-u)a_+,ua_-\}$. The minimum lies where the two terms are equal, at $u=\tfrac{a_+}{a_++a_-}$ with the loss $\tfrac{a_+a_-}{a_++a_-}$. When $u$ takes only the values 0 and 1, the minimum is $\min(a_+,a_-)$. (3) A rule that reads $E'$ becomes a rule that reads $E$ when it first computes $E'$ from $E$, and the two rules choose the same version in every situation.

Static indicators do not change with the usage context. When one set of static indicators belongs to two usage contexts with opposite best versions, every rule that reads only static indicators keeps a worst-case loss that it cannot remove. The experiments of the survey contain such a case. Deleting all wrong records of Hospitals raises the Macro-F1 of KNN by 0.487 and lowers that of gradient boosting by 0.579, while the static indicators of the raw table and of the version after deletion stay the same under both model classes. With $a_+=0.487$ and $a_-=0.579$, part (2) gives a minimax loss of about 0.265 for a rule that reads only static indicators, and about 0.487 without randomization. To support a choice of version, an assessment result therefore states the downstream model class and the task, or adds a task test. Part (3) gives the order of the levels. A higher level reads all inputs of the lower level, and when the report also gives the result of the lower level, that result is a function of the higher result, so a rule that reads the higher result has a minimax loss no larger than a rule that reads the lower result. The functions are `zero_loss_rule_exists`, `two_version_gaps`, `minimax_loss` and `minimax_choice_probability`.

**Proposition 1, compensation bound of equal-weight scores.** Let a dimension hold $K$ sub-indicators with values in $[0,1]$, the equal-weight score $P=\tfrac{100}{K}\sum_{i=1}^K s_i$ and a pass line of 90 points. Then $P\ge 90$ if and only if $\sum_{i=1}^K(1-s_i)\le 0.1K$, and with the other sub-indicators at full marks a single sub-indicator can fall to $\max(0,1-0.1K)$.

*Proof.* The first statement follows from $P=100-\tfrac{100}{K}\sum_{i=1}^K(1-s_i)$, and setting the other $K-1$ sub-indicators to 1 gives the floor of a single one.

GB/T 48562-2026 recommends equal weights. The floor of a single sub-indicator is then 0.2 for the $K=8$ items of data quality and 0.6 for the $K=4$ items of documentation. With adjustable weights the floor of item $i$ with weight $\omega_i$ is $\max(0,1-\tfrac{0.1}{\omega_i})$, and an item with a weight of at most 0.1 can reach 0 while the dimension passes. Safety conformity or content authenticity can therefore fall to 0.2 at 90 points. Model fit cannot fall below about 0.895, because it enters the exponential factor of the model application score. Since $P\le 100\,\mathrm{e}^{s_m-1}$, that dimension fails for $s_m<1+\ln 0.9$ whatever values the other items take. The survey makes safety, license and authenticity veto items, which enter Algorithm 5 as the set $\mathcal{A}$, and computes the composite score on the other items above their own thresholds. The functions are `compensation_budget`, `within_compensation_budget`, `single_item_floor`, `weighted_item_floor` and `model_fit_threshold`.

**Variance of three estimates of an error rate, Section 4.1 of the survey.** The comparison concerns the labels that the three estimates of Definition 7 need. Let $\pi_1=\mathbb{E}(Y\mid H=1)$ and $\pi_0=\mathbb{E}(Y\mid H=0)$ be the shares of wrong records among the flagged and the unflagged records, and let $\rho$ be the correlation coefficient of $Y$ and $H$.

To first order, each estimate minus $\pi$ equals the mean over the $n$ sampled records of $Y-\pi$, of $Y-\pi-(\pi_1-\pi_0)(H-\hat{p})$ and of $Y-\pi-\tfrac{1}{\lambda}(H-\hat{p})$, in this order. With $\rho^2=\tfrac{\lambda^2\pi(1-\pi)}{\hat{p}(1-\hat{p})}$, the asymptotic variances of the three estimates are $\pi(1-\pi)$, $\pi(1-\pi)(1-\rho^2)$ and $\pi(1-\pi)(\rho^{-2}-1)$. The post-stratified estimate therefore needs no more labels than the other two for every detector, and the corrected estimate needs fewer labels than the direct estimate only when $\rho^2>\tfrac{1}{2}$.

*Proof.* The flagged share $\hat{p}$ comes from a full scan and carries no sampling error, so all three errors come from the $n$ labels. The direct estimate is the sample mean of $Y$. In the post-stratified estimate, $\bar{Y}_{H=1}-\pi_1$ is to first order the sample mean of $\tfrac{H(Y-\pi_1)}{\hat{p}}$ and $\bar{Y}_{H=0}-\pi_0$ that of $\tfrac{(1-H)(Y-\pi_0)}{1-\hat{p}}$. Since $\pi=\hat{p}\pi_1+(1-\hat{p})\pi_0$, the sum $H(Y-\pi_1)+(1-H)(Y-\pi_0)$ equals $Y-\pi-(\pi_1-\pi_0)(H-\hat{p})$. In the corrected estimate, the partial derivatives of $\tfrac{\hat{p}-\phi_0}{\phi_1-\phi_0}$ with respect to $\phi_1$ and $\phi_0$ are $-\tfrac{\pi}{\lambda}$ and $-\tfrac{1-\pi}{\lambda}$, and the sample estimates of $\phi_1$ and $\phi_0$ deviate to first order by the sample means of $\tfrac{Y(H-\phi_1)}{\pi}$ and $\tfrac{(1-Y)(H-\phi_0)}{1-\pi}$. The combination is $-\tfrac{1}{\lambda}(H-\phi_0-\lambda Y)$, which equals $Y-\pi-\tfrac{1}{\lambda}(H-\hat{p})$ because $\hat{p}=\phi_0+\lambda\pi$. From $\mathbb{E}(YH)=\pi\phi_1$ follows $\mathrm{Cov}(Y,H)=\lambda\pi(1-\pi)$, and with $\mathrm{Var}(H)=\hat{p}(1-\hat{p})$ this gives $\rho^2$. Since $\pi_1-\pi_0=\tfrac{\mathrm{Cov}(Y,H)}{\mathrm{Var}(H)}$, the second quantity has the variance $\pi(1-\pi)(1-\rho^2)$. The third has the variance $\pi(1-\pi)+\tfrac{\hat{p}(1-\hat{p})}{\lambda^2}-2\pi(1-\pi)=\pi(1-\pi)(\rho^{-2}-1)$. By the sample size formula above, the labels for one half-width are proportional to these variances. The comparison follows from $1-\rho^2\le 1$, from $1-\rho^2\le\tfrac{1-\rho^2}{\rho^2}$ for $\rho^2\le 1$, and from $\rho^{-2}-1<1$ exactly when $\rho^2>\tfrac{1}{2}$.

A detector without errors has $\lambda=1$ and $\rho=1$, and a detector whose detection rate equals its false alarm rate has $\lambda=0$ and $\rho=0$. The corrected estimate divides by $\lambda$, so its variance grows without bound as $\lambda$ goes to 0. The post-stratified estimate uses the flags only to form two groups and does not divide by $\lambda$. The function `estimator_variances` gives the three variances and $\rho^2$. `rogan_gladen` and Algorithm 1 compute the corrected estimate, and `detector_interval_from_counts` in `hqd_assess.finite_population` computes the post-stratified estimate with a finite-population interval.

**Proposition 2, risk decomposition of data operations.** The target is the mean $\mu=\mathbb{E}_{\mathcal{P}}X$ of a variable $X$ on the clean population $\mathcal{P}$, and the observed value is $X+C$ with the error $C$. A data operation keeps a record with probability $r(X,C)$, $p_r=\mathbb{E}_{\mathcal{P}}r>0$, and the kept records follow $\mathrm{d}\tilde{\mathcal{P}}=\tfrac{r}{p_r}\mathrm{d}\mathcal{P}$. The prediction $\hat{\mu}$ is a weighted mean of the observed values of the kept records. Its weight vector $\omega$ is a constant that does not depend on the observed values and sums to 1, and the observed values all follow $\tilde{\mathcal{P}}$ with the covariance matrix $\Sigma$. The risk of $\hat{\mu}$ is the squared loss $\mathcal{R}(\hat{\mu})=\mathbb{E}_{\mathcal{P}}(X-\hat{\mu})^2$ with the expectation over $X$ only, and $\mathcal{R}(\mu)$ is defined in the same way.

1. The excess risk is $\mathbb{E}\mathcal{R}(\hat{\mu})-\mathcal{R}(\mu)=(b_{\mathrm{sel}}+b_{\mathrm{err}})^2+\omega^{\top}\Sigma\omega$, with the selection bias $b_{\mathrm{sel}}=\tfrac{\mathrm{Cov}_{\mathcal{P}}(r,X)}{p_r}$ and the error bias $b_{\mathrm{err}}=\tfrac{\mathbb{E}_{\mathcal{P}}(rC)}{p_r}$.
2. Let two error types cause the biases $b_1$ and $b_2$, let repairs leave the variance unchanged, and take the negative risk as the utility. The interaction of repairing both types, the joint gain minus the sum of the single gains, is $I=-2b_1b_2$.
3. Let the records come from $J$ independent runs with $k$ records each, and let record $i$ of run $j$ be observed as $\mu+b+A_j+\zeta_{ji}$, where the run effects $A_j$ and the record residuals $\zeta_{ji}$ are independent with mean zero and variances $\tau^2$ and $\sigma^2$. The excess risk of the equal-weight mean is $b^2+\tfrac{\tau^2}{J}+\tfrac{\sigma^2}{Jk}$.

*Proof.* (1) $\mathbb{E}\hat{\mu}-\mu=\mathbb{E}_{\tilde{\mathcal{P}}}(X+C)-\mu=\tfrac{\mathbb{E}_{\mathcal{P}}(rX)}{p_r}-\mathbb{E}_{\mathcal{P}}X+\tfrac{\mathbb{E}_{\mathcal{P}}(rC)}{p_r}=b_{\mathrm{sel}}+b_{\mathrm{err}}$. Under squared loss $\mathcal{R}(\hat{\mu})-\mathcal{R}(\mu)=(\hat{\mu}-\mu)^2$, whose expectation is the squared bias plus the variance $\omega^{\top}\Sigma\omega$ of $\hat{\mu}$. (2) Without the common variance term, the excess risks of the raw table, of repairing only the first type, of repairing only the second type and of repairing both are $(b_1+b_2)^2$, $b_2^2$, $b_1^2$ and 0. The interaction is then $(b_1+b_2)^2-\big((b_1+b_2)^2-b_2^2\big)-\big((b_1+b_2)^2-b_1^2\big)=-2b_1b_2$. (3) The error of the equal-weight mean is $b+\tfrac{1}{J}\sum_jA_j+\tfrac{1}{Jk}\sum_{j,i}\zeta_{ji}$, and the last two terms have the variances $\tfrac{\tau^2}{J}$ and $\tfrac{\sigma^2}{Jk}$.

The error rate therefore does not fix the task risk. A repair changes only the error bias, a deletion also changes the selection bias and the variance, and when the biases of two error types have opposite signs, repairing the type with the smaller absolute bias alone raises the risk. The example under [Non-additivity of the dimensions](#non-additivity-of-the-dimensions) is part (2) with $b_1=\tfrac{2}{11}$ and $b_2=-\tfrac{2}{11}$. By part (3), at a fixed number $Jk$ of records more independent runs lower the run term $\tfrac{\tau^2}{J}$, which a count of records does not show. The functions are `operation_biases`, `repair_interaction` and `run_mean_excess_risk`.

**Proposition 3, sample size of task tests.** The models trained on the candidate versions are fixed before the held-out records are used. The $n$ held-out records are independent and identically distributed and take no part in training or in building the candidates. The metric is the mean of single-record scores $s_j(x)\in[0,1]$, and the statements hold conditionally on the training results.

1. Let the paired difference $Z\in[-1,1]$ between each of $N$ candidates and the reference version on one record have variance at most $q$. For all $N$ gain estimates to lie within $\tfrac{\gamma}{2}$ of their true values at once with probability at least $1-\delta$, $n\ge\big(\tfrac{8q}{\gamma^2}+\tfrac{8}{3\gamma}\big)\ln\tfrac{2N}{\delta}$ suffices.
2. Let $j^{*}$ be the best of $N$ candidates, let $d_j$ be its expected score minus that of candidate $j$, let $q_j$ be the variance of the single-record score difference of the two, and let $\hat{j}$ be the candidate with the highest validation mean. For every $\gamma\ge 0$, $\Pr(d_{\hat{j}}>\gamma)\le\sum_{j:d_j>\gamma}\exp\big(-\tfrac{nd_j^2}{2q_j+\frac{4d_j}{3}}\big)$.

*Proof.* (1) Since $|Z-\mathbb{E}Z|\le 2$, the Bernstein inequality bounds the probability that one gain estimate deviates from its true value by more than $\tfrac{\gamma}{2}$ by $2\exp\big(-\tfrac{n\gamma^2}{8q+\frac{8\gamma}{3}}\big)$. Requiring this to be at most $\tfrac{\delta}{N}$ and taking the union bound gives the size. (2) When candidate $j$ is chosen, the sample mean of $W_i=s_{j^{*}}(x_i)-s_j(x_i)$ is at most 0, while $\mathbb{E}W_i=d_j$, $\mathrm{Var}W_i=q_j$ and $|W_i-d_j|\le 2$. By the one-sided Bernstein inequality this event has probability at most $\exp\big(-\tfrac{nd_j^2}{2q_j+\frac{4d_j}{3}}\big)$, and the union bound over the candidates with $d_j>\gamma$ gives the statement. The candidates are compared on the same records and need not be independent.

The variance bound $q$ can be the share of records on which two versions predict differently. With $q=q_j=0.1$, $N=20$, $\delta=0.05$ and $\gamma$ of 2 points, part (1) needs about 14300 records to tell apart accuracy gains that differ by $\gamma$. The Flights table has about 713 test records, which tell apart only gains that differ by about 10 points. Requiring each term of part (2) to be at most $\tfrac{\delta}{N-1}$ gives the sufficient size $n\ge\max_{j:d_j>\gamma}\big(\tfrac{2q_j}{d_j^2}+\tfrac{4}{3d_j}\big)\ln\tfrac{N-1}{\delta}$, which is at most $\big(\tfrac{2q}{\gamma^2}+\tfrac{4}{3\gamma}\big)\ln\tfrac{N-1}{\delta}$ when every $q_j\le q$ and comes to about 3400 records. Gains on benchmark test sets that differ by a few points thus need intervals, while the choice of a good enough version needs far fewer records. Macro-F1 and $R^2$ are not means of single-record scores, and the package gives their precision with `gain_precision_bootstrap`. The functions are `task_test_sample_size` and `resolution_for_sample_size` for part (1), and `selection_sample_size` and `wrong_choice_bound` for part (2).

**Theorem 2, coverage gaps and scene shift.** Let the target scene distribution $\mathcal{P}_S$ put probability $\alpha\in(0,1)$ on an input region $\mathcal{X}_0$, let all test records come from outside $\mathcal{X}_0$, and let the models trained on the candidate versions $D_a$ and $D_b$ predict the same outside $\mathcal{X}_0$ and differently everywhere on $\mathcal{X}_0$.

1. There are two target scenes with the same joint distribution outside $\mathcal{X}_0$ and the same input distribution on $\mathcal{X}_0$, in which the accuracy of the model trained on $D_a$ minus that of the model trained on $D_b$ is $\alpha$ and $-\alpha$. However many test records come from outside $\mathcal{X}_0$, the minimax loss of the choice between the two is $\tfrac{\alpha}{2}$, and with $n_0$ further labelled records drawn independently from $\mathcal{P}_S$ it is $\tfrac{\alpha}{2}(1-\alpha)^{n_0}$.
2. Let the test distribution be $Q$, let the single-record score difference of a candidate and the reference version lie in $[-1,1]$, and let the gain estimate $\hat{\Delta}_Q$ satisfy $|\hat{\Delta}_Q-\Delta_Q|\le\varepsilon$ with probability at least $1-\delta$. If the total variation distance between $\mathcal{P}_S$ and $Q$ on labelled records, the largest difference between the probabilities that the two distributions give to one event, is at most $\beta$, then $\Delta_{\mathcal{P}_S}\ge\hat{\Delta}_Q-\varepsilon-2\beta$ with the same probability, and the constant 2 cannot be improved.

*Proof.* (1) The first scene takes the predictions of the model trained on $D_a$ as the labels on $\mathcal{X}_0$, and the second scene takes those of the model trained on $D_b$. Since the two models disagree everywhere on $\mathcal{X}_0$, the model trained on $D_a$ is right everywhere on $\mathcal{X}_0$ in the first scene and wrong everywhere on it in the second, and the model trained on $D_b$ the reverse. Records outside $\mathcal{X}_0$ have the same distribution in the two scenes, so Theorem 1 (2) with $a_+=a_-=\alpha$ gives $\tfrac{\alpha}{2}$. All $n_0$ target records miss $\mathcal{X}_0$ with probability $(1-\alpha)^{n_0}$, the two scenes cannot then be told apart, and the worst-case conditional expected loss is at least $\tfrac{\alpha}{2}$. One record on $\mathcal{X}_0$ suffices to identify the scene and choose the right candidate. (2) Let $g$ be the single-record score difference and $\tilde{g}=\tfrac{g+1}{2}\in[0,1]$. Then $|\Delta_{\mathcal{P}_S}-\Delta_Q|=2|\mathbb{E}_{\mathcal{P}_S}\tilde{g}-\mathbb{E}_Q\tilde{g}|\le 2\beta$, and adding the estimation error gives the bound. On a space of two points, $g=(1,-1)$, $\mathcal{P}_S=(1-\beta,\beta)$ and $Q=(1,0)$ give $\Delta_{\mathcal{P}_S}=\Delta_Q-2\beta$, so with an exact estimate the bound holds with equality.

By part (1), more records from the distribution of the old test set cannot make up for a coverage gap. When TEP serves the choice of a model for deployment across operating modes, the modes outside the base mode form the uncovered region. In part (2) a larger test set shrinks only $\varepsilon$, and the term $2\beta$ from the change of scene needs data of the new scene. Stage 6 uses part (2) when the scene changes. The functions are `coverage_minimax_loss` and `shifted_gain_lower_bound`.
