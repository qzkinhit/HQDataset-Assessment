# Standards and the assessment levels of their indicators

## Comparison of the standards

Each standard answers a different question. The column Status gives the date in force or the date of publication. The sub-indicator codes, formulas and decision rules of the high-quality dataset series follow the public draft and the technical document TC609-5-2025-04.

| Standard | Object | Main content | Decision rule and sampling | Status |
|---|---|---|---|---|
| [GB/T 48562-2026] | High-quality dataset | Documentation, data quality, model application, 17 sub-indicators | 90 points per dimension, weighted sum, no sampling plan | 2027-05 |
| [TC609-5-2025-04] | High-quality dataset | Technical document with the three dimensions and the score formula | 90 points, any failed dimension fails the dataset | 2025-08 |
| [GB/T 48560-2026] | Data annotation | Annotation process and quality control, trial labeling of at least 2% | No threshold on κ, sample left to risk assessment | 2027-05 |
| [GB/T 48563-2026] | Dataset format | Metadata elements for each record, formats by modality | Conformity to the required format | 2027-05 |
| [GB/T 48561-2026] | Dataset type | General, industry general and industry specific datasets, 7 factors | Integrated judgment of the 7 factors | 2027-05 |
| [GB/T 45652-2025] | Training data safety | Requirements on data sources, content and annotation | 96% on 4000+ records by hand, 98% on 10%+ of records by tools | 2025-11 |
| [GB/T 25000.24-2017] | Data | Measures of data quality, adopts ISO/IEC 25024 | Measures without a rating range | Current |
| [ISO/IEC 25012:2008] | Stored data | 15 characteristics, inherent and system dependent | Characteristics without a rating | Current |
| [ISO/IEC 25024:2015] | Stored data | Measures of the 15 characteristics | Measures without a rating range | Current |
| ISO/IEC 5259-1 to 3 [ISO/IEC 5259-1:2024, ISO/IEC 5259-2:2024, ISO/IEC 5259-3:2024] | Analytics and ML data | Characteristics and measures, management | Quality reports and quality claims | 2024 |
| ISO 8000-1, 8000-61 [ISO 8000-1:2022, ISO 8000-61:2016] | Data | Principles of data quality, reference model of the process | Conformance of the process | Current |
| [ISO/IEC TR 24027:2021] | AI systems | Sources and assessment of bias | Technical report without a threshold | Current |
| [EU AI Act, Art. 10] | High-risk system data | Relevant, representative, free of errors and complete as far as possible | Qualitative requirements | 2027-12 |
| [NIST AI 100-1, NIST AI 600-1] | AI systems | Voluntary risk framework and generative AI profile | No tolerance set | 2023 to 2024 |

## Sub-indicators of GB/T 48562-2026 and their levels

The column Unit gives the statistical unit of a sub-indicator. The column Given gives the level that the calculation in the standard reaches. The column Needed gives the lowest level that supports the conclusion of the sub-indicator. The last column gives the work that reaches this level. The levels are those of the items G48562-0101 to G48562-0305 in [data/requirements/items.jsonl](../data/requirements/items.jsonl), described in the next section, and the same mapping is available in code as `hqd_assess.scoring.SUB_INDICATORS`.

| Code | Sub-indicator | Unit | Given | Needed | Required to reach the needed level |
|---|---|---|---|---|---|
| 0101 to 0104 | Documentation completeness, four aspects | Aspects | L1 | L2 | Fields checked against data and lineage |
| 0201, 0204 | Format conformity, structural completeness | Records | L3 | L3 | Full scan, accepted missing range |
| 0202 | Safety conformity | Records | L2 | L3 | Sample size, veto threshold |
| 0203 | Annotation conformity | Records | L2 | L3 | Sample judged by hand, with an interval |
| 0205 | Content authenticity | Records | L1 | L3 | Lineage matched on a sample traced back, with an interval |
| 0206, 0207 | Content consistency, type consistency | Records | L2 | L3 | Detector stated, sample labeled |
| 0208 | Cleanliness (Annex A) | Records | L2 | L3 | Proxy metrics checked on a labeled sample |
| 0301 | Content diversity | Dataset | L3 | L4 | Coverage by usage, subgroups tested |
| 0302 | Scale completeness | Dataset | L3 | L4 | Learning curves, independent units |
| 0303 | Timeliness | Records | L3 | L4 | Task test on a later period |
| 0304 | Annotation accuracy | Records | L2 | L3 | Sampled ground truth with an interval |
| 0305 | Model fit | Dataset | L4 | L5 | Reference, model, split, intervention |

The table shows three links for further work. The documentation items and content authenticity stay at L1 and need a comparison of documents with the data. The ratio items judged by people or detectors have formulas without a sampling plan, a sample size or an interval, while GB/T 45652-2025 already gives a sample size and a threshold. Diversity, scale and timeliness do not read task results, and model fit stops at L4, so it reports a score and does not show which change of the data caused it.

## Requirement items of the assessment documents

Section 3.2 of the survey assigns assessment levels to all quality requirements of 13 assessment documents. Three of them are technical documents with nearly the same content as three national standard drafts, so Table 3 counts 10 documents. [data/requirements/items.jsonl](../data/requirements/items.jsonl) holds one requirement item per line, and [analysis/requirement_levels.py](../analysis/requirement_levels.py) computes Table 3, the other numbers of Section 3.2 and the agreement of the two coding rounds from this file.

### Documents and extraction unit

The unit of extraction is a requirement that leads to a pass or fail decision on data, documents or the construction process, called an item below. Clauses that only describe workflows, staff or organizational management and lead to no pass or fail decision are not extracted. Nine documents are domestic and four are international.

| File | Document | Items | Extracted part |
|---|---|---|---|
| G48562 | GB/T 48562-2026, public draft, quality test and evaluation | 44 | The 17 sub-indicators, each joining the requirement of Chapter 4 with the calculation of the same code in Chapter 5, the scoring and overall decision rules S0 to S3, and the 23 cleanliness items of Annex A |
| T04 | TC609-5-2025-04, quality evaluation | 44 | Same structure as G48562 |
| G48560 | GB/T 48560-2026, public draft, data annotation requirements | 225 | Trial labeling (6.2.5), contents of the annotation rules (6.2.6), sampled and full checks (6.3.4 d)), acceptance and delivery (6.4), the general requirements of 7.1 and the 181 lettered requirements of the annotation tasks in 7.2 to 7.9, process documents (8.3), check methods, consistency measures and counting rules (8.4) and data compliance (8.5.1) |
| G48561 | GB/T 48561-2026, public draft, classification guide | 8 | One item for each of the 7 type factors of 4.1 and one for the type decision of 4.3. These items decide the type of a dataset, on which type consistency (0207) of G48562 relies |
| T03 | TC609-5-2025-03, classification guide | 8 | Same as G48561 |
| G48563 | GB/T 48563-2026, public draft, format requirements | 17 | One item for each metadata element |
| T02 | TC609-5-2025-02, format requirements | 16 | Same as G48563 without the element for the annotation tool |
| DPQ | Data products, general requirements for quality evaluation, public draft, plan number 20255404-T-907 | 74 | The 53 characteristics of Chapter 6, the 20 extension characteristics for machine learning of Annex A and the composite score of 7.3 |
| T01 | TC609-5-2025-01, construction guide | 6 | Subsets of the split and their statistics (5 a)), usability (5 b)), quality measurement at collection (7 c)), data validation (8 b)), model validation and root cause decision (Chapter 10) |
| I25012 | ISO/IEC 25012:2008, preview | 1 | Conformance clause (Chapter 2) |
| I25024 | ISO/IEC 25024:2015, preview | 3 | Conformance clause (Chapter 2) |
| NAI100 | NIST AI 100-1 | 6 | MAP 2.3, MAP 4.1, MEASURE 2.1, MEASURE 2.2, the test set requirement of Section 3.1 and the requirement on the sources and intellectual property of training data of Section 3.4 |
| NAI600 | NIST AI 600-1 | 27 | Suggested actions that read training data, test, evaluation, verification and validation (TEVV) data, dataset provenance or dataset documentation |

The texts of ISO/IEC 25012:2008 and ISO/IEC 25024:2015 were available only as previews. The preview of ISO/IEC 25012 ends at the definitions in 4.2 and the preview of ISO/IEC 25024 at the definitions in 4.5, so the characteristics of Chapter 5 of ISO/IEC 25012 and the measures of Chapter 8 of ISO/IEC 25024 are not visible. Their four items come from the conformance clauses, and Table 3 marks their row with a dagger. The scope of ISO/IEC 25024 states that it does not define rating ranges for the measures and does not address the statistical significance of random samples. The previews of ISO/IEC 5259-1, 5259-2, 5259-3 and 5259-5, ISO/IEC TR 5259-6, ISO/IEC TR 24027, ISO 8000-1, ISO 8000-2 and ISO 8000-61 end at their definitions, contents or process structure and contain no requirement of this kind, so they give no item.

### Fields

Each line of the file is one item with the following fields. The texts of `name`, `requirement` and `formula` are in Chinese for all documents, and `document` and `clause` follow the language of the document.

| Field | Content |
|---|---|
| `id` | Short name of the document and number of the item. G48562, T04 and DPQ keep the indicator codes of the document, S0 to S3 are the scoring and overall decision rules of G48562 and T04, Q is the composite score of DPQ, and the suffix -A marks the two extension characteristics of DPQ whose codes repeat codes of Chapter 6. The items of the other documents are numbered in the order of the text |
| `file`, `document` | Short name and full name of the document |
| `clause` | Clause, table or annex of the requirement |
| `name`, `requirement` | Name of the requirement and a summary of at most 80 characters. A long enumeration is cut, and the document gives it in full |
| `formula` | Numerator and denominator, or the pass condition, and `none` when the document gives no formula |
| `inputs` | Inputs that the calculation or the decision reads, from the eight categories below |
| `sampling` | `true` when the document sets a sampling plan, a sample size, a sampling proportion or a confidence interval. A mention of sampling without a plan or a number is `false` |
| `threshold` | `true` when the document gives an explicit pass condition, either a numeric bound such as 90 points or 2 percent, or a format condition such as a mandatory field, a value domain or a referenced format standard. A value given as an example does not count |
| `aggregation` | `true` when the item enters a weighted or equal-weight composite score or is such a score. S1 to S3 of G48562 and T04 and Q of DPQ are such scores. S0 requires each of the three dimensions to reach 90 points, so the dimensions do not compensate each other and S0 is `false` |
| `coverage` | `partial` when a requirement of a domestic document applies only to part of the data or of the users, such as a trial batch, a check sample, a questionnaire sample or an optional case, and for the four items of the two ISO/IEC previews, otherwise `full` |
| `given`, `needed` | Given level and needed level after adjudication, `L1` to `L5` |
| `form` | Form of computation, one of the six forms below |
| `round1`, `round2` | `given`, `needed` and `form` of the two independent coding rounds |

| Input | What the item reads |
|---|---|
| `documents` | Documents of the provider, such as the dataset description |
| `metadata` | Metadata of the dataset or of its records |
| `records_by_rule` | Records decided by a program from a definition, a rule, a code table or a reference distribution |
| `records_by_human_or_detector` | Records decided by people, a detector or a model score |
| `external_source_or_log` | Facts outside the records, such as the original source, operation logs, user questionnaires and user task records |
| `labeled_sample` | A labeled sample, such as a trial batch, a check sample or a test set |
| `trained_model` | A trained model with its predictions or its performance |
| `controlled_variant` | A version of the data in which only one aspect changes |

### Levels

Each item has two levels on the scale of [framework.md](framework.md), L1 Declared, L2 Matched, L3 Measured, L4 Task-tested and L5 Interventional. The given level is the level reached by the inputs the item reads. The needed level is the level of the question the item answers. The five questions are whether the documents state a property, whether the statements agree with the data, what share of the dataset has a property, whether the data serve a given task, and which change of the data causes a difference.

The given level follows five rules.

1. L1 when the item reads only statements in documents, metadata or logs and checks whether something is stated, without a comparison with the data. An item that lists what a log, a document or a delivery must contain and checks whether it is contained is also L1.
2. L2 when some part of the decision rests on people, a detector or facts outside the records, such as an external source, a synchronization log or the true time, and the item does not derive a sample size from a confidence level and a tolerated error. A fixed sampling share, as in the trial labeling of G48560, stays at L2.
3. L3 when a program decides from the fields of the records with an explicit rule, code table or reference distribution, over all records and with an exact result. An item that prescribes a sampling method and derives the sample size from a confidence level and a tolerated error is also L3.
4. L4 when the item reads the performance of a trained model on held-out data.
5. L5 when the item reads a version of the data in which only one aspect changes.

A composite score, and an overall decision that combines such scores (S0 of G48562 and T04), takes the lowest given level of its components. The type decision of G48561 rests on an integrated judgment by people and follows rule 2.

The needed level follows four rules.

1. L1 when the documents must state a rule, strategy, practice or disclosure chosen by the provider, so that the statement itself answers the question.
2. L2 when the documents must state a fact about the data or their production, such as scale, distribution, source, version, split, processing and check records or evaluation results, because the fact holds only if it agrees with the data. An item that judges a whole, such as a label system, a set of functions, a list of compliance items or a plan, by checking each part for agreement also needs L2.
3. L3 when the question is the share of records, time or events that meet a criterion, or the value of an index.
4. L4 when the requirement is judged by model development and training in the target scene, and L5 when it is judged by whether the data improve the model. A ratio to a planned or theoretical amount stays below L4 when the item names no task.

An item whose needed level is above its given level answers its question with inputs of a lower level. Column N>G of Table 3 counts these items.

### Forms of computation

| Form | Computation |
|---|---|
| `doc_presence` | Aspects or items that the documents state or meet, against those required, including a check of presence |
| `record_ratio` | Records that meet the requirement against all records |
| `scale_ratio` | Amount of the dataset against the amount that the use needs or the plan sets |
| `model_ratio` | Model performance against expected performance |
| `sampled_rate` | Pass rate on a prescribed number of sampled records |
| `other` | A pass condition only, a composite score, a type decision, another ratio, or no formula and no check of presence |

### Duplicate documents, coding rounds and adjudication

Three pairs each consist of a national standard draft and a technical document with nearly the same content, namely G48562 and T04, G48563 and T02, G48561 and T03. Their items correspond one to one by name except the metadata element for the annotation tool (G48563-0016), which T02 does not have. Within a pair the given level, the needed level and the form differ only for content authenticity (0205) and the data quality score (S2) that contains it. T04 requires the data to agree with the source of collection, which gives L2, and G48562 decides from the stated source and processing records, which gives L1. Table 3 and the counts that Section 3.2 derives from it keep the drafts and leave out T04, T02 and T03, which gives 411 items in 10 documents. The agreement of the two coding rounds uses all 479 items.

The given level, the needed level and the form of all 479 items were coded twice, in two independent rounds under the rules above. Items with a different code in at least one of the three fields were adjudicated one by one under the same rules, and a code on which both rounds agree was kept, except three given levels that the rules decide otherwise. G48562-S2 takes L1, the lowest given level of its components, and G48560-0013 and G48560-0015, which list deliverables, take L1 by rule 1. `round1` and `round2` hold the codes of the two rounds, and `given`, `needed` and `form` hold the decided codes. The agreement rates and Cohen's kappa compare `round1` with `round2` on all items, on the 442 items of the domestic documents and on the 37 items of the international documents.

### Running the script

`python analysis/requirement_levels.py` prints Table 3, the other numbers of Section 3.2 and the agreement of the two rounds, and `python analysis/requirement_levels.py --json` prints the same numbers as JSON. The script needs only the Python standard library. `pytest tests/test_requirement_levels.py` compares its results with the numbers of the survey.

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
