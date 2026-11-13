"""Graded quality assessment of datasets.

The package implements the framework of the survey "A survey on task-oriented assessment
of multimodal high-quality datasets". It holds pure functions without file or network
access. Data loading and model training stay with the caller. Definitions, theorems and
propositions carry the numbers of the survey, and algorithms carry the numbers of
docs/framework.md.
"""
from .claim import AssessmentLevel, DatasetVersion, Estimate, QualityClaim, UsageContext, minimum_level, task_gain
from .interaction import (additivity_counterexample, benjamini_yekutieli, constant_predictor_risk, interaction_quantity,
                          interaction_test, one_sample_t, operation_biases, paired_flags, repair_interaction,
                          run_mean_excess_risk)
from .judge import ClaimRecord, Judgement, Requirement, judge
from .renewal import STAGES, renew, stages_to_redo, trigger_counts, violation_trigger
from .sampling import (bootstrap_metric_interval, calibrate_error_rate, calibrate_from_audit, estimator_variances,
                       jeffreys_interval, minimum_observed_rate, proportion_half_width, proportion_sample_size, rogan_gladen,
                       wilson_interval, zero_error_upper_bound)
from .scoring import (PASS_LINE, SUB_INDICATORS, data_quality_score, dimension_score, expected_performance_band,
                      model_application_score, model_fit_factor, model_fit_threshold, model_fit_value, passes, score_range,
                      single_item_floor, weighted_item_floor)
from .selection import (accuracy, coverage_minimax_loss, gain_interval, gain_precision_bootstrap, hindsight_best, macro_f1,
                        minimax_choice_probability, minimax_loss, r2, replay_rules, resolution_for_sample_size,
                        select_version, selection_sample_size, shifted_gain_lower_bound, split_holdout, static_rule,
                        static_rule_expected_loss, task_test_sample_size, two_version_gaps, wrong_choice_bound,
                        zero_loss_rule_exists)
from .split_check import (adjacent_window_overlap, check_split_independence, group_overlap, group_split,
                          sliding_window_starts, window_row_overlap)

__version__ = "1.0.0"
