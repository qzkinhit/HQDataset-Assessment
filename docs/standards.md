# Standards and the assessment levels of their indicators

## Comparison of the standards

Each standard answers a different question. The column Status gives the date in force or the date of publication. The sub-indicator codes, formulas and decision rules of the high-quality dataset series follow the public draft and the technical document TC609-5-2025-04.

| Standard | Object | Main content | Decision rule and sampling | Status |
|---|---|---|---|---|
| [GB/T 48562-2026] | High-quality dataset | Documentation, data quality, model application, 17 sub-indicators | 90 points per dimension, weighted sum, no sampling plan | 2027-05 |
| [TC609-5-2025-04] | High-quality dataset | Technical document with the three dimensions and the score formula | 90 points, any failed indicator fails the dataset | 2025-08 |
| [GB/T 48560-2026] | Data annotation | Annotation process and quality control, trial labeling of at least 2% | No threshold on κ, sample left to risk assessment | 2027-05 |
| [GB/T 48563-2026] | Dataset format | Metadata elements for each record, formats by modality | Conformity to the required format | 2027-05 |
| [GB/T 48561-2026] | Dataset type | General, industry general and industry specific datasets, 7 factors | Integrated judgment of the 7 factors | 2027-05 |
| [GB/T 45652-2025] | Training data safety | Requirements on data sources, content and annotation | 96% on 4000+ records by hand, 98% on 10%+ of records by tools | 2025-11 |
| [GB/T 25000.24-2017] | Data | Measures of data quality, adopts ISO/IEC 25024 | Measures without a rating range | Current |
| [ISO/IEC 25012:2008] | Stored data | 15 characteristics, inherent and system dependent | Characteristics without a rating | Current |
| [ISO/IEC 25024:2015] | Stored data | Measures of the 15 characteristics | Measures without a rating range | Current |
| ISO/IEC 5259-1 to 6 [ISO/IEC 5259-1:2024, ISO/IEC 5259-2:2024, ISO/IEC 5259-3:2024] | Analytics and ML data | Characteristics and measures, management, process, governance | Quality reports and quality claims | 2024 to 2026 |
| ISO 8000-1, 8000-61 [ISO 8000-1:2022, ISO 8000-61:2016] | Data | Principles of data quality, reference model of the process | Conformance of the process | Current |
| [ISO/IEC TR 24027:2021] | AI systems | Sources and assessment of bias | Technical report without a threshold | Current |
| [EU AI Act, Art. 10] | High-risk system data | Relevant, representative, free of errors and complete as far as possible | Qualitative requirements | 2027-12 |
| [NIST AI 100-1, NIST AI 600-1] | AI systems | Voluntary risk framework and generative AI profile | No tolerance set | 2023 to 2024 |

## Sub-indicators of GB/T 48562-2026 and their levels

The column Unit gives the statistical unit of a sub-indicator. The column Given gives the level that the calculation in the standard reaches. The column Needed gives the lowest level that supports the conclusion of the sub-indicator. The last column gives the work that reaches this level. The same mapping is available in code as `hqd_assess.scoring.SUB_INDICATORS`.

| Code | Sub-indicator | Unit | Given | Needed | Required to reach the needed level |
|---|---|---|---|---|---|
| 0101 to 0104 | Documentation completeness, four aspects | Aspects | L1 | L2 | Fields checked against data and lineage |
| 0201, 0204 | Format conformity, structural completeness | Records | L3 | L3 | Full scan, accepted missing range |
| 0202 | Safety conformity | Records | L3 | L3 | Sample size, veto threshold |
| 0203 | Annotation conformity | Records | L2 | L3 | Sample judged by hand, with an interval |
| 0205 | Content authenticity | Records | L1 | L2 | Lineage matched, sample traced back |
| 0206, 0207 | Content consistency, type consistency | Records | L3 | L3 | Detector stated, sample labeled |
| 0208 | Cleanliness (Annex A) | Records | L3 | L4 | Proxy metrics compared with a task score |
| 0301 | Content diversity | Dataset | L3 | L4 | Coverage by usage, subgroups tested |
| 0302 | Scale completeness | Dataset | L3 | L5 | Learning curves, independent units |
| 0303 | Timeliness | Records | L2 | L4 | Task test on a later period |
| 0304 | Annotation accuracy | Records | L3 | L3 | Sampled ground truth with an interval |
| 0305 | Model fit | Dataset | L4 | L4, L5 | Reference, model, split, intervention |

The table shows three links for further work. The documentation items stay at L1 and need a comparison of documents with the data. The ratio items have formulas without a sampling plan, a sample size or an interval, while GB/T 45652-2025 already gives a sample size and a threshold. Cleanliness, diversity, scale and timeliness do not read task results, and model fit stops at L4, so it reports a score and does not show which change of the data caused it.

## References

- [EU AI Act, Art. 10] European Parliament and Council. Regulation (EU) 2024/1689 (Artificial Intelligence Act), article 10: data and data governance. Official Journal of the European Union, 2024.
- [GB/T 25000.24-2017] SAC/TC 28. GB/T 25000.24-2017 Systems and software engineering: systems and software quality requirements and evaluation (SQuaRE), part 24: measurement of data quality. 2017. 全国信息技术标准化技术委员会. GB/T 25000.24-2017 系统与软件工程 系统与软件质量要求和评价 (SQuaRE) 第 24 部分: 数据质量测量. 2017.
- [GB/T 45652-2025] SAC/TC 260. GB/T 45652-2025 Cybersecurity technology: security specification for pre-training and optimization training data of generative artificial intelligence. 2025. 全国网络安全标准化技术委员会. GB/T 45652-2025 网络安全技术 生成式人工智能预训练和优化训练数据安全规范. 2025.
- [GB/T 48560-2026] SAC/TC 609. GB/T 48560-2026 High-quality dataset: data annotation requirements. 2026. 全国数据标准化技术委员会. GB/T 48560-2026 高质量数据集 数据标注要求. 2026.
- [GB/T 48561-2026] SAC/TC 609. GB/T 48561-2026 High-quality dataset: classification guide. 2026. 全国数据标准化技术委员会. GB/T 48561-2026 高质量数据集 分类指南. 2026.
- [GB/T 48562-2026] SAC/TC 609. GB/T 48562-2026 High-quality dataset: specifications for quality test and evaluation. 2026. 全国数据标准化技术委员会. GB/T 48562-2026 高质量数据集 质量测评规范. 2026.
- [GB/T 48563-2026] SAC/TC 609. GB/T 48563-2026 High-quality dataset: format requirements. 2026. 全国数据标准化技术委员会. GB/T 48563-2026 高质量数据集 格式要求. 2026.
- [ISO 8000-1:2022] ISO. ISO 8000-1:2022 Data quality, part 1: overview. 2022.
- [ISO 8000-61:2016] ISO. ISO 8000-61:2016 Data quality, part 61: data quality management: process reference model. 2016.
- [ISO/IEC 25012:2008] ISO/IEC. ISO/IEC 25012:2008 Software engineering: software product quality requirements and evaluation (SQuaRE), data quality model. 2008.
- [ISO/IEC 25024:2015] ISO/IEC. ISO/IEC 25024:2015 Systems and software engineering: systems and software quality requirements and evaluation (SQuaRE), measurement of data quality. 2015.
- [ISO/IEC 5259-1:2024] ISO/IEC. ISO/IEC 5259-1:2024 Artificial intelligence: data quality for analytics and machine learning (ML), part 1: overview, terminology, and examples. 2024.
- [ISO/IEC 5259-2:2024] ISO/IEC. ISO/IEC 5259-2:2024 Artificial intelligence: data quality for analytics and machine learning (ML), part 2: data quality measures. 2024.
- [ISO/IEC 5259-3:2024] ISO/IEC. ISO/IEC 5259-3:2024 Artificial intelligence: data quality for analytics and machine learning (ML), part 3: data quality management requirements and guidelines. 2024.
- [ISO/IEC TR 24027:2021] ISO/IEC. ISO/IEC TR 24027:2021 Information technology: artificial intelligence (AI), bias in AI systems and AI aided decision making. 2021.
- [NIST AI 100-1] National Institute of Standards and Technology. Artificial intelligence risk management framework (AI RMF 1.0). NIST AI 100-1. 2023.
- [NIST AI 600-1] National Institute of Standards and Technology. Artificial intelligence risk management framework: generative artificial intelligence profile. no. 600-1, 2024.
- [TC609-5-2025-04] SAC/TC 609. TC609-5-2025-04 High-quality dataset: specifications for quality evaluation. 2025. 全国数据标准化技术委员会. TC609-5-2025-04 高质量数据集 质量评测规范. 2025.
