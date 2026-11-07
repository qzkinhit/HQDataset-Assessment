# Tools for dataset quality

The table lists representative open-source tools with the version on the search date of 2026-11-06. The column Level gives the assessment level that the output of a tool reaches directly, as defined in [framework.md](framework.md). The column Family gives the method family, where Documents marks a check of metadata and Operators marks a library of processing operators. Licenses come from the project pages.

| Tool | Version | Data | Main functions | Family | Level | License |
|---|---|---|---|---|---|---|
| [Deequ](https://github.com/awslabs/deequ) [Schelter et al. 2018] | 2.1.0 (2026-09) | Tables on Spark | Constraints, metrics, anomaly history | Rules | L2, L3 | Apache-2.0 |
| [Great Expectations](https://github.com/fivetran/great_expectations) | 1.24.1 (2026-10) | Tables | Expectations, validation reports | Rules | L2, L3 | Apache-2.0 |
| [TFDV](https://github.com/tensorflow/data-validation) [Breck et al. 2019] | 1.21.0 (2026-06) | Tables | Schema checks, skew and anomalies | Rules | L2, L3 | Apache-2.0 |
| [Soda](https://github.com/sodadata/soda-core) | 4.30.0 (2026-11) | Data warehouses | Data contracts, declarative checks | Rules | L2 | See the project |
| [fg-data-profiling](https://github.com/Data-Centric-AI-Community/fg-data-profiling) | 4.20.0 (2026-09) | DataFrames | Profile reports and alerts | Sampling | L3 | MIT |
| [Evidently](https://github.com/evidentlyai/evidently) | 0.7.23 (2026-09) | Tables, text, LLM | Drift detection, test suites | Sampling | L3 | Apache-2.0 |
| [SDMetrics](https://github.com/sdv-dev/SDMetrics) | 0.33.0 (2026-10) | Synthetic tables | Fidelity and privacy reports | Sampling | L3 | MIT |
| [cleanlab](https://github.com/cleanlab/cleanlab) [Northcutt et al. 2021] | 2.9.0 (2026-01) | Multimodal | Label errors, outliers, near duplicates | Learned | L3 | Apache-2.0 |
| [mlcroissant](https://github.com/mlcommons/croissant) [Akhtar et al. 2024] | 1.1.1 (2026-10) | Metadata | Validation of metadata schemas | Documents | L2 | Apache-2.0 |
| [Data-Juicer](https://github.com/datajuicer/data-juicer) [Chen et al. 2024] | 1.6.2 (2026-11) | Multimodal | Over 200 operators, processing recipes | Operators | L3 | Apache-2.0 |
| [NeMo Curator](https://github.com/NVIDIA-NeMo/Curator) | 1.4.0 (2026-10) | Multimodal | Filtering and deduplication on GPU | Operators | L3 | Apache-2.0 |
| [DCLM](https://github.com/mlfoundations/dclm) [Li et al. 2024] | No tag (2025-09) | Web text | Filtering, fixed recipe, 53 evaluations | Task | L5 | MIT |

Most tools stay at L2 and L3. A constraint library reports which records break the given constraints, so its conclusion covers those constraints only. A profile or drift report describes the data and needs a reference or a task test before it supports a judgement. Operator libraries change the data, and a downstream experiment shows the effect of the change. DCLM fixes a training recipe and 53 evaluations, so a corpus filter can be compared at L5.

## References

- [Akhtar et al. 2024] Mubashara Akhtar, Omar Benjelloun, Costanza Conforti, et al. Croissant: a metadata format for ML-Ready datasets. Advances in Neural Information Processing Systems, pp. 1-6, 2024.
- [Breck et al. 2019] Eric Breck, Neoklis Polyzotis, Sudip Roy, et al. Data validation for machine learning. Second Conference on Machine Learning and Systems, pp. 334-347, 2019.
- [Chen et al. 2024] Daoyuan Chen, Yilun Huang, Zhijian Ma, et al. Data-Juicer: a One-Stop data processing system for large language models. Companion of the 2024 International Conference on Management of Data, pp. 120-134, 2024.
- [Li et al. 2024] Jeffrey Li, Alex Fang, Georgios Smyrnis, et al. DataComp-LM: in search of the next generation of training sets for language models. Advances in Neural Information Processing Systems, 2024.
- [Northcutt et al. 2021] Curtis G. Northcutt, Lu Jiang, Isaac L. Chuang. Confident learning: estimating uncertainty in dataset labels. J Artif Intell Res, vol. 70, pp. 1373-1411, 2021.
- [Schelter et al. 2018] Sebastian Schelter, Dustin Lange, Philipp Schmidt, et al. Automating large-scale data quality verification. Proc VLDB Endow, vol. 11, no. 12, pp. 1781-1794, 2018.
