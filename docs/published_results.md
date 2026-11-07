# Published results on static quality and task utility

The table collects representative studies that measure a static quality indicator and a task score on the same data. In the column Relation, Conflict means that quality and utility move in opposite directions, and Cond. means that the direction depends on the condition in the last column.

| Study | Data, errors | Finding | Relation | Condition |
|---|---|---|---|---|
| [Li et al. 2021] | 14 tables, real | Share of positive effects, missing 49%, duplicates 11% | Cond. | Error type, method |
| [Neutatz et al. 2022] | CleanML, AutoML | Imputation gains at most 1% | Cond. | AutoML adapts |
| [Guha et al. 2023] | 5 tables, real | Accuracy worse 13%, fairness worse 37% | Conflict | Fairness metric |
| [Mohammed et al. 2025] | 10 tables, injected | Over 25 points against under 5 points | Cond. | Dimension, method |
| [Karlaš et al. 2020] | 4 tables, missing | HoloClean closes from minus 64% to 11% of the gap | Conflict | Task awareness |
| [Ni et al. 2024] | 12 tables, correlation | Error drop rate (EDR) correlates at 0.53 in classification, 0.12 in clustering | Cond. | Task type |
| [Recht et al. 2019] | CIFAR-10, ImageNet | Same procedure, down 3% to 15% | Conflict | Distribution gap |
| [Oakden-Rayner et al. 2020] | Medical images | Overall AUC 0.87, subclass 0.77 | Conflict | Hidden subclass |
| [Li et al. 2024] | 240T web tokens | Core about 28.5 at ROC-AUC 82% against human labels, above 31 at 73% | Conflict | Filter, task |
| [Engstrom et al. 2024] | C4 web text | Similarity selection not above random | Conflict | Quality definition |
| [Liang et al. 2026] | 6 domains, correlation | Baseline DAS correlates at 0.86 in math, 0.18 in finance | Cond. | Domain, model |
| [Gadre et al. 2023] | 12.8B image-text | Best filter keeps about 30% | Cond. | Criterion, model |
| [Mohri et al. 2026] | 670M web tokens | No filter wins at long training | Cond. | Compute, criterion |
| [Montesuma et al. 2026] | Multi-mode TEP | Above 80% in mode, down to 18% across | Conflict | Operating mode |

The criterion changes the direction of a filter in the corpus studies, and the task type and the method change the effect of a repair in the table studies. The distribution gap, a hidden subclass and the operating mode change the score while the labels and the collection procedure stay the same. The experiments of this repository put error truth, static scores and task scores on the same data.

## References

- [Engstrom et al. 2024] Logan Engstrom, Axel Feldmann, Aleksander Madry. DsDm: Model-Aware dataset selection with datamodels. 41st International Conference on Machine Learning, pp. 12491-12526, 2024.
- [Gadre et al. 2023] Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, et al. DataComp: in search of the next generation of multimodal datasets. Advances in Neural Information Processing Systems, 2023.
- [Guha et al. 2023] Shubha Guha, Falaah Arif Khan, Julia Stoyanovich, et al. Automated data cleaning can hurt fairness in machine learning-based decision making. 39th IEEE International Conference on Data Engineering, pp. 3747-3754, 2023.
- [Karlaš et al. 2020] Bojan Karlas, Peng Li, Renzhi Wu, et al. Nearest neighbor classifiers over incomplete information: from certain answers to certain predictions. Proc VLDB Endow, vol. 14, no. 3, pp. 255-267, 2020.
- [Li et al. 2021] Peng Li, Xi Rao, Jennifer Blase, et al. CleanML: a study for evaluating the impact of data cleaning on ML classification tasks. 37th IEEE International Conference on Data Engineering, pp. 13-24, 2021.
- [Li et al. 2024] Jeffrey Li, Alex Fang, Georgios Smyrnis, et al. DataComp-LM: in search of the next generation of training sets for language models. Advances in Neural Information Processing Systems, 2024.
- [Liang et al. 2026] Hao Liang, Qifeng Cai, Yibo Lin, et al. DataPrep-Bench: benchmarking LLMs as training data preparators. ArXiv:2607.20465, 2026.
- [Mohammed et al. 2025] Sedir Mohammed, Lukas Budach, Moritz Feuerpfeil, et al. The effects of data quality on machine learning performance on tabular data. Inf Syst, vol. 132, pp. 102549, 2025.
- [Mohri et al. 2026] Christopher Mohri, John C. Duchi, Tatsunori Hashimoto. A bitter lesson for data filtering. ArXiv:2605.19407, 2026.
- [Montesuma et al. 2026] Eduardo Fernandes Montesuma, Michela Mulas, Fred Ngolè Mboula, et al. Benchmarking domain adaptation for chemical processes on the Tennessee Eastman process. Machine Learning and Principles and Practice of Knowledge Discovery in Databases Workshops, pp. 307-322, 2026.
- [Neutatz et al. 2022] Felix Neutatz, Binger Chen, Yazan Alkhatib, et al. Data cleaning and AutoML: would an optimizer choose to clean?. Datenbank Spektrum, vol. 22, no. 2, pp. 121-130, 2022.
- [Ni et al. 2024] Wei Ni, Xiaoye Miao, Xiangyu Zhao, et al. Automatic data repair: are we ready to deploy?. Proc VLDB Endow, vol. 17, no. 10, pp. 2617-2630, 2024.
- [Oakden-Rayner et al. 2020] Luke Oakden-Rayner, Jared Dunnmon, Gustavo Carneiro, et al. Hidden stratification causes clinically meaningful failures in machine learning for medical imaging. ACM Conference on Health, Inference, and Learning, pp. 151-159, 2020.
- [Recht et al. 2019] Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, et al. Do ImageNet classifiers generalize to ImageNet?. 36th International Conference on Machine Learning, pp. 5389-5400, 2019.
