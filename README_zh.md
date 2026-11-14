# 高质量数据集测评

本仓库存放论文《面向任务的多模态高质量数据集测评研究》的代码、数据和结果。论文按五个测评层级梳理数据质量标准和测评方法，统计 10 份测评类文件中要求条目的测评层级，并对 54 个公开数据集做质量测评。仓库同时存放方法软件包 `hqd_assess` 及其测试，以及论文正文没有包含的文档，包括六阶段测评流程及其算法、全部数据集和参数的说明，以及工具、标准和已发表结果的比较。

英文版见 [README.md](README.md)。

## 工作量

| 项目 | 规模 |
|---|---|
| 文献 | 在 Semantic Scholar 上用 22 组检索式检索 2010 年至 2026 年 11 月的文献，命中 52213 篇，按被引次数筛选并去重后得到 21835 篇，论文引用 207 篇 |
| 标准和政策 | 国家标准、征求意见稿、TC609 技术文件、ISO/IEC 5259、ISO 8000、欧盟人工智能法案和 NIST AI RMF，有公开文本的逐篇通读 |
| 数据集 | 11 组共 54 个数据集，约 $3.39\times 10^8$ 条记录 |
| 下载数据 | 除 `data/` 中的文件外，`scripts/download_data.sh` 从官方来源下载 146 个文件，共约 17 GiB |
| 测评做法 | 论文五类方法中的 66 种具体做法 |
| 公开数据集测评 | 11 组数据集上的 6 个研究问题，44041 次模型训练（每个配置计一次），结果文件记录的运行时间累计约 760 小时，使用 64 核 CPU 服务器和 A6000 GPU |
| 代码 | 测评脚本和软件包 `hqd_assess` 共约 7200 行 Python 代码，163 项测试 |

| 数据集分组 | 数据集数 | 记录数 |
|---|---:|---:|
| 带基准错误的表格 | 11 | 299,122 |
| 带真实错误的表格 | 13 | 589,415 |
| 图像标签 | 2 | 120,000 |
| 大模型语料 | 2 | 1,564,345 |
| 多模态数据 | 2 | 530,671 |
| 合成数据 | 3 | 350,000 |
| 动态数据 | 1 | 259,287,888 |
| 评测协议 | 6 | 15,971,700 |
| 评测数据泄露 | 3 | 24,000 |
| 测试集标签 | 10 | 60,585,386 |
| 检测器估计 | 1 | 5,000 |
| 合计 | 54 | 339,327,527 |

对整张表做静态扫描时计入该表的全部记录。`analysis/big_tables.py` 从结果文件计算每个数据集的记录数。

## 基本概念和测评层级

论文第 2 节把一条关于数据集的质量结论写成六元组，依次为测评对象、质量维度、度量、带不确定性的估计值、测评层级和使用条件。测评层级记录结论的获得方式，论文按这五个层级梳理标准和方法。

| 层级 | 名称 | 做法 |
|---|---|---|
| L1 | 声明 | 提供方在文档中陈述 |
| L2 | 比对 | 把陈述或规则与数据本身、日志和血缘逐项对照 |
| L3 | 测量 | 按明确定义的估计量计算指标，并给出不确定性 |
| L4 | 任务 | 用数据集训练指定模型，在留出数据上测量任务效用 |
| L5 | 干预 | 只改变数据的某一方面，其他条件不变，比较下游结果 |

贯穿全文的例子是经典 Tennessee Eastman 故障诊断基准。它没有缺失值，重复记录极少，按记录统计的得分接近满分。在同一批文件上，同一个随机森林的 Macro-F1 仅因划分方式就相差 0.233，随机划分窗口时每个测试窗口都与训练窗口共享记录，按文件名标注时每个故障测试文件有 16.7% 的记录标错。

[docs/framework_zh.md](docs/framework_zh.md) 列出论文中的定义，并把六阶段测评流程写成算法 1 到算法 6。这一流程是本仓库的文档，不属于论文正文。

## 软件包 `hqd_assess`

软件包只含纯函数，不访问文件和网络，依赖 NumPy 和 SciPy。在仓库根目录运行下面的代码，也可以用 `pip install .` 安装。

| 模块 | 内容 |
|---|---|
| `claim` | `QualityClaim`、`Estimate`、`UsageContext`、`DatasetVersion`、`AssessmentLevel`、`minimum_level`、`task_gain` |
| `scoring` | GB/T 48562-2026 的得分、`SUB_INDICATORS`、单项下限、模型适配性 |
| `sampling` | 样本量、Wilson 和 Jeffreys 区间、`rogan_gladen`、`calibrate_error_rate`（算法 1） |
| `split_check` | 分组和窗口重叠、`check_split_independence`（算法 2） |
| `selection` | `select_version`（算法 3）、`static_rule`、`hindsight_best`、`replay_rules` |
| `interaction` | `interaction_test`（算法 4）、`paired_flags`、`additivity_counterexample` |
| `judge` | `judge`（算法 5） |
| `renewal` | `renew`（算法 6）、`stages_to_redo`、`violation_trigger`、`trigger_counts` |

```python
from hqd_assess import (AssessmentLevel, Estimate, QualityClaim, Requirement, UsageContext, judge, minimum_observed_rate,
                        model_application_score, model_fit_value, proportion_sample_size, single_item_floor,
                        wilson_interval)

print(proportion_sample_size(0.9, 0.02))                    # 半宽 2 个百分点需要 865 条记录
print(single_item_floor(8), single_item_floor(4))           # 90 分时单项下限为 0.2 和 0.6
print(minimum_observed_rate(0.96, 4000))                    # 观测合格率 0.96625 时下界达到 96%
print(model_application_score(model_fit_value(0.697, 0.8))) # 87.9 分，低于通过线

usage = UsageContext("fault diagnosis", "Macro-F1", "split by run", ("random forest",), "base mode")
low, high = wilson_interval(3865, 4000)
safety = QualityClaim("TEP test set", "safety conformity", "share of safe records", Estimate(3865 / 4000, low, high, "Wilson"),
                      AssessmentLevel.L3, usage)
fit = QualityClaim("TEP test set", "model fit", "Macro-F1 by run", Estimate(0.636), AssessmentLevel.L4, usage)
verdict = judge([Requirement(safety, AssessmentLevel.L3, 0.96)], [Requirement(fit, AssessmentLevel.L4, 0.80)])
print(verdict.passed, verdict.records[0].below_threshold)   # False True
print(fit)
```

示例和测试使用仓库内的数据。

```bash
pip install -r requirements.txt
python examples/tep_assessment.py                    # 静态扫描、标签检查、划分检查、质量结论
python examples/table_claims.py --table hospitals    # 错误率、抽样、校正、版本选择、综合判定
python analysis/print_tables.py                      # 按数据集输出主要结果
pytest
```

## 仓库内容

| 路径 | 内容 |
|---|---|
| `hqd_assess/` | 方法软件包 |
| `tests/` | 复现论文中全部计算得到的数字，并把软件包的结果与运行记录相比较 |
| `examples/` | 经典 TEP 上的 `tep_assessment.py` 和单张表上的 `table_claims.py` |
| `docs/framework.md`，`docs/framework_zh.md` | 论文中的定义和测评层级，以及仓库给出的六阶段流程和六个算法 |
| `docs/datasets.md` | 每个数据集的官方来源、来源声明的许可、用途、规模、错误来源、下载命令和对应脚本 |
| `docs/parameters.md` | 全部工程参数及其所在文件，以及每个结果文件存放的运行数 |
| `docs/release_assets.md` | 每个发布归档包的内容 |
| `docs/tools.md`，`docs/published_results.md`，`docs/standards.md` | 工具比较、静态质量和任务效用的已发表结果、标准比较和子指标层级 |
| `data/` | 小体积数据文件，见 `data/README.md` |
| `case_tep/`，`big/`，`study/`，`native/`，`corpus/`，`llm/` | 各组测评的脚本 |
| `analysis/` | 运行汇总、论文中的数字和表格 |
| `figures/` | 测评结果图 |
| `results/` | 运行结果记录 |
| `scripts/download_data.sh` | 从来源下载数据集的命令 |

## 各组测评

| 测评 | 数据 | 错误来源 | 问题 | 脚本 |
|---|---|---|---|---|
| 表格 | 11 张有脏版本和干净版本的表 | 基准自带 | 标准的得分能否确定任务分数，修复是否可加，修复和删除比例的影响 | `study/run_real_tables.py`，`study/run_tables_fine.py` |
| 版本选择 | 同样的 11 张表 | 基准自带 | $n$ 条样本上的任务测试相对静态规则的增益 | `study/run_decision.py`，`study/run_decision_fine.py` |
| 清洗方法 | 13 张 CleanML 表 | 真实错误 | 任务测试会选中基准中的哪种清洗方法 | `native/clean_cleanml.py`，`native/run_cleanml.py` |
| 检测器 | 同样的 11 张表 | 基准自带 | 检测器和抽样估计的错误率与真实错误率相差多少 | `study/run_detectors.py` |
| 贯穿全文的例子 | 经典 TEP | 评测协议 | 划分、窗口长度和标签能使分数变化多少 | `case_tep/run_case.py`，`case_tep/run_windows.py` |
| 独立运行数 | Rieth TEP，1533 万条记录 | 评测协议 | 记录数相同时独立运行数的影响 | `big/run_rieth.py`，`big/run_rieth_models.py` |
| 时间 | 纽约市出租车记录，72 个月，2.59 亿条 | 采集过程 | 规则违反率能否反映任务效用的变化，过滤和扩大规模的作用 | `big/run_taxi.py` |
| 人工标签噪声 | CIFAR-10N，CIFAR-100N | 采集过程 | 修复和删除人工标签错误的效果 | `study/run_vision.py` |
| 覆盖 | CIFAR-100 超类 | 覆盖 | 子群体缺失与同等规模损失的比较 | `study/run_vision.py` |
| 近重复 | CIFAR 测试集的 ciFAIR 清单 | 基准自带 | 测试分数中有多少来自训练集中有近重复的图像 | `study/fair.py` |
| 分布偏移 | WILDS Camelyon17，CivilComments | 真实偏移 | 医院和子群体的偏移使分数变化多少 | `native/prep_camelyon.py`，`native/run_camelyon.py`，`native/run_civilcomments.py` |
| 网页语料 | C4 分片和 DCLM 语料，以 Wikipedia 为参照 | 采集过程 | 困惑度最低的文本是否更好，取决于哪些指标和规模 | `corpus/run_corpus.py`，`native/prep_dclm.py` |
| 泄漏和语料缺陷 | 五领域文本池 | 按文献构造 | 发生测试数据泄露的基准会报告什么结果 | `study/run_text.py` |
| 图文 | COCO val2017 | 按文献构造 | 单一缺陷下的检索效果，相似度代理的误差 | `study/run_multimodal.py` |
| 图文数据池 | DataComp 分片，MIRFLICKR-25000 | 采集过程 | 图文对上的筛选如何影响检索 | `native/run_datacomp.py`，`native/embed_mirflickr.py`，`native/run_mirflickr.py` |
| 合成图像 | EDM、DDPM、CIFAKE 图像和 CIFAR-10.1 | 真实生成模型 | 合成训练数据在真实测试集上的表现，保真度指标能否反映任务分数 | `study/run_synth.py`，`native/fidelity.py` |
| 测试集标签错误 | Northcutt 等的 10 个测试集 | 采集过程 | 重新评审后的标签使测试分数变化多少 | `native/label_errors.py` |
| 大模型基线 | 上述表格、C4 和 3 个测试集 | 同上 | 基于大模型的修复、语料评分和标注能达到什么效果 | `llm/`，见 `llm/README.md` |

## 数据集

[docs/datasets.md](docs/datasets.md) 为每个数据集列出官方来源、来源声明的许可、用途和下载命令。带基准错误的表格和 44 个经典 TEP 文件在 `data/`，ciFAIR 清单和测试集标签错误的评审投票也在其中。`scripts/download_data.sh DIR` 从公开地址把其余数据集下载到 `DIR`，`scripts/download_data.sh DIR rieth taxi` 可以只下载其中几个。脚本中的全部链接已于 2026 年 11 月 7 日逐个测试可用。

| 分组 | 数据集 |
|---|---|
| 基准自带错误的表，来自 [REIN](https://github.com/mohamedyd/rein-benchmark) 和 UniClean | [Adult](https://archive.ics.uci.edu/dataset/2/adult)，[Beers](https://www.kaggle.com/datasets/nickhould/craft-cans)，[Breast Cancer](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original)，[Flights](http://lunadong.com/fusionDataSets.htm)，[Hospitals](https://github.com/BigDaMa/raha)，[SmartFactory](https://www.kaggle.com/datasets/inIT-OWL/high-storage-system-data-for-energy-optimization)，[Soccer](https://db.unibas.it/projects/bart/)，[Bike](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)，[Mercedes](https://www.kaggle.com/competitions/mercedes-benz-greener-manufacturing/data)，[NASA](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise)，[Soil Moisture](https://github.com/felixriese/hyperspectral-soilmoisture-dataset) |
| 带真实错误的表，来自 [CleanML](https://github.com/chu-data-lab/CleanML) | Airbnb，Citation，Company，Credit，EEG，KDD，Marketing，Movie，Restaurant，Sensor，Titanic，University，USCensus |
| 图像标签 | [CIFAR-10N，CIFAR-100N](https://github.com/UCSC-REAL/cifar-10-100n)，图像来自 [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html) |
| 大模型语料 | [C4](https://huggingface.co/datasets/allenai/c4)，[DCLM 语料池](https://huggingface.co/datasets/mlfoundations/dclm-pool-1b-1x)，[DCLM 基线](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0)，作为参照的 [Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia)，由 [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) 等 5 个来源构成的五领域文本池 |
| 多模态数据 | [DataComp](https://huggingface.co/datasets/mlfoundations/datacomp_small)，[MIRFLICKR-25000](https://press.liacs.nl/mirflickr/)，[COCO val2017](https://cocodataset.org/#download) |
| 合成数据 | [EDM 图像](https://github.com/wzekai99/DM-Improves-AT)，[DDPM 图像](https://github.com/google-deepmind/deepmind-research/tree/master/adversarial_robustness)，[CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) |
| 动态数据 | [纽约市 TLC 黄色出租车记录](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) |
| 评测协议 | [经典 TEP](http://web.mit.edu/braatzgroup/links.html)，[Rieth TEP](https://doi.org/10.7910/DVN/6C3JR1)，[Camelyon17 和 CivilComments](https://wilds.stanford.edu/datasets/)，[CIFAR-10.1](https://github.com/modestyachts/CIFAR-10.1)，CIFAR-100 超类 |
| 近重复 | [ciFAIR-10，ciFAIR-100](https://cvjena.github.io/cifair)，C4 的近重复组 |
| 测试集标签 | [Northcutt 等](https://github.com/cleanlab/label-errors) 整理的 MNIST、CIFAR-10、CIFAR-100、Caltech-256、ImageNet、QuickDraw、20 News、IMDB、Amazon、AudioSet |

## 复现步骤

每个脚本的路径来自命令行参数或环境变量 `HQD_ROOT`、`HQD_POOL` 和 `HQD_CORPUS`，默认值相对于仓库根目录，下载数据在 `downloads/`，结果文件在 `results/`。作业脚本把结果追加到结果文件并跳过已完成的作业，因此可以中断后继续。`--procs`、`--threads`、`--gpus` 和 `--per-gpu` 控制硬件用量。`docs/parameters.md` 列出了全部设置。下面的命令在仓库根目录运行。

```bash
pip install -r requirements.txt
DIR=downloads
scripts/download_data.sh $DIR

# 表格和版本选择
python study/run_real_tables.py
python study/run_tables_fine.py
python study/run_decision.py
python study/run_decision_fine.py
python analysis/tables_summary.py results/results_real_tables.jsonl > results/tables_summary.json
python analysis/table_sizes.py data/tables results/table_sizes.json

# 同样的表上的检测器
python study/run_detectors.py --root data/tables --meta data/tables/table_meta.json --out $DIR/detectors --stage detect
python study/run_detectors.py --root data/tables --meta data/tables/table_meta.json --out $DIR/detectors --stage raha    # 需要 raha 软件包
python study/run_detectors.py --root data/tables --meta data/tables/table_meta.json --out $DIR/detectors --stage measure
python study/run_detectors.py --root data/tables --meta data/tables/table_meta.json --out $DIR/detectors --stage check
cp $DIR/detectors/results_detectors.jsonl $DIR/detectors/detectors_summary.json $DIR/detectors/sanity_detectors.json results/
python analysis/validate_detector_intervals.py --flags-dir $DIR/detectors/flags --previous-results results/results_detectors.jsonl --skip-enumeration --output-dir $DIR/intervals
cp $DIR/intervals/results_finite_population.jsonl results/

# 经典 TEP
python case_tep/run_case.py
python case_tep/run_windows.py
python case_tep/make_traces.py data/tep results/tep_traces.json

# Rieth TEP 和出租车记录
python big/run_rieth.py --seeds 5
python big/run_rieth_models.py
python big/run_taxi.py
python big/run_taxi.py --sizes 100000 --out results/results_taxi_100k.json
python big/run_taxi.py --sizes 10000,30000,300000,3000000 --out results/results_taxi_sizes.json

# GPU 测评任务的作业清单，由 study/gpu_pool.py 分配到各张卡
export HQD_ROOT=$DIR HQD_POOL=$DIR/processed HQD_CORPUS=$DIR/corpus
mkdir -p jobs
python study/make_jobs.py jobs
python corpus/make_jobs_corpus.py jobs/corpus.jsonl
python native/make_jobs_native.py jobs/synth.jsonl jobs/dclm.jsonl

# 人工标签噪声、覆盖和近重复
python study/prep_vision.py $DIR $DIR/cifar_n $DIR/cifar-10-python.tar.gz
python study/gpu_pool.py --jobs jobs/vision.jsonl --script study/run_vision.py --out results/results_vision.jsonl
python study/gpu_pool.py --jobs jobs/vision_fine.jsonl --script study/run_vision.py --out results/results_vision_fine.jsonl
python study/coverage_reference.py --root $DIR --results results/results_vision.jsonl --out results/results_vision_coverage_ref.json   # 全覆盖模型在较小保留比例未见的细类上的准确率
python study/fair.py $DIR results/results_fair.json

# 合成图像、保真度和图像版本的回放
python native/prep_native.py $DIR $DIR/cifar10_real.npz
python study/gpu_pool.py --jobs jobs/synth.jsonl --script study/run_synth.py --out results/results_synth.jsonl
python study/embed_cifar.py --npz $DIR/cifar10_real.npz --out $DIR/cifar10_clip.npz
python native/fidelity.py --out results/results_fidelity.json
python native/replay_versions.py $DIR results/results_versions.jsonl

# 网页语料和 DCLM
python corpus/prep_corpus.py --dir $DIR/corpus --out $DIR/corpus
cp $DIR/corpus/corpus_stats.json results/corpus_stats.json
python corpus/static_corpus.py $DIR/corpus results/corpus_static.json
python native/prep_dclm.py $DIR/corpus/tok/tokenizer.json $DIR $DIR/corpus
python corpus/run_corpus.py --job '{"mode": "ppl"}' --out /dev/null
python study/gpu_pool.py --jobs jobs/corpus.jsonl --script corpus/run_corpus.py --out results/results_corpus.jsonl
python study/gpu_pool.py --jobs jobs/dclm.jsonl --script corpus/run_corpus.py --out results/results_corpus.jsonl

# 文本池
python study/run_text.py --job '{"mode": "ppl"}' --out /dev/null
python study/gpu_pool.py --jobs jobs/text.jsonl --script study/run_text.py --out results/results_text.jsonl
python study/gpu_pool.py --jobs jobs/text_fine.jsonl --script study/run_text.py --out results/results_text_more.jsonl

# COCO 图文对和图文数据池
python study/embed_coco.py $DIR
python study/gpu_pool.py --jobs jobs/multimodal.jsonl --script study/run_multimodal.py --out results/results_multimodal.jsonl
python study/gpu_pool.py --jobs jobs/multimodal_fine.jsonl --script study/run_multimodal.py --out results/results_multimodal_fine.jsonl
python native/run_datacomp.py --native $DIR --out results/results_datacomp.jsonl
python native/embed_mirflickr.py --native $DIR --out $DIR/mirflickr_clip.npz --device cuda
python native/run_mirflickr.py --native $DIR --feats $DIR/mirflickr_clip.npz --out results/results_mirflickr.jsonl

# CleanML 表，每张表在基准仓库内部完成清洗
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

# 测试集标签错误
python native/label_errors.py $DIR/label_errors $DIR/preds results/label_errors.json
python native/label_errors_audioset.py $DIR/label_errors $DIR/label_errors/class_labels_indices.csv results/results_audioset_labels.json

# 原生数据上各组测评的静态质量得分，脚本从数据目录读取结果文件
cp results/results_datacomp.jsonl results/results_mirflickr.jsonl results/results_fidelity.json $DIR/
python native/static_native.py --native $DIR --rieth $DIR/rieth --out results

# 数字、表格和图
python analysis/summaries.py results
python analysis/make_numbers.py results                 # 正文中的全部数字，输出 JSON
python analysis/print_tables.py                         # 按数据集输出的表格
python analysis/print_tables.py --latex                 # 论文表 3 和表 4 的 LaTeX
python figures/plot_results.py results figures/plots    # 单个子图 res_*.pdf 组成图 4 到图 8
python native/audit_native.py results                   # 原生数据上各组测评的完成情况与作业清单的比较
python analysis/stored_runs.py --update docs/parameters.md
```

大模型基线的命令见 [llm/README.md](llm/README.md)。`results/dataset_docs.json` 中的说明文档得分来自阅读该文件列出的来源页面，每一项都附有原文。

## 论文中的图表和结果文件的对应关系

图 1 到图 3 和归纳标准与方法的表格随论文正文排版，下表列出其内容的来源。最后一列写上面命令中对应的分组或具体命令。

| 论文中的内容 | 结果文件 | 脚本 | 对应命令 |
|---|---|---|---|
| 图 1，TEP 的静态检查和任务分数 | `results/case_tep.json`，`results/tep_traces.json` | `case_tep/run_case.py`，`case_tep/make_traces.py` | 经典 TEP |
| 图 2，研究演进 | | | 论文引用的文献 |
| 图 3，基本概念与测评层级 | `docs/framework_zh.md` | `hqd_assess/claim.py`，`hqd_assess/scoring.py` | 依据定义绘制 |
| 图 4(a)，版本选择 | `results/results_decision.jsonl`，`results/results_decision_fine.jsonl` | `study/run_decision.py`，`study/run_decision_fine.py`，`figures/plot_results.py` | 表格和版本选择，数字、表和图 |
| 图 4(b)，部分修复增益占全部修复增益的比例 | `results/results_real_tables.jsonl`，`results/results_tables_fine.jsonl` | `study/run_tables_fine.py`，`figures/plot_results.py` | 表格和版本选择，数字、表和图 |
| 图 4(c)，CleanML 表上的清洗方法 | `results/results_cleanml.jsonl` | `native/clean_cleanml.py`，`native/run_cleanml.py`，`figures/plot_results.py` | CleanML 表，数字、表和图 |
| 图 4(d)，修复和删除人工标签错误 | `results/results_vision.jsonl`，`results/results_vision_fine.jsonl` | `study/run_vision.py`，`figures/plot_results.py` | 人工标签噪声、覆盖和近重复，数字、表和图 |
| 图 5(a) 到 (d)，修复和删除的增益、等量随机删除、沿修复路径和删除路径的 Spearman 相关、联合修复和单项修复 | `results/tables_summary.json`，`results/results_real_tables.jsonl`，`results/results_random_delete.jsonl` | `study/run_real_tables.py`，`study/run_random_delete.py`，`analysis/tables_summary.py`，`figures/plot_results.py` | 表格和版本选择，数字、表和图 |
| 图 6(a)，评测协议和数据错误 | `results/tables_summary.json`，`results/case_tep.json` | `case_tep/run_case.py`，`figures/plot_results.py` | 经典 TEP，数字、表和图 |
| 图 6(b)，独立运行数 | `results/results_rieth.json`，`results/results_rieth_models.jsonl` | `big/run_rieth.py`，`big/run_rieth_models.py`，`figures/plot_results.py` | Rieth TEP 和出租车记录，数字、表和图 |
| 图 6(c)，错误率估计和区间 | `results/results_detectors.jsonl`，`results/results_finite_population.jsonl` | `study/run_detectors.py`，`analysis/validate_detector_intervals.py`，`figures/plot_results.py` | 同样的表上的检测器，数字、表和图 |
| 图 6(d)，Qwen 修复和破坏的单元格 | `results/results_llm_cells.json`，`results/results_llm_tables.jsonl` | `llm/repair_tables.py`，`llm/score_tables.py`，`figures/plot_results.py` | [llm/README.md](llm/README.md)，数字、表和图 |
| 图 7(a)，细类覆盖 | `results/results_vision.jsonl`，`results/results_vision_coverage_ref.json` | `study/run_vision.py`，`study/coverage_reference.py`，`figures/plot_results.py` | 人工标签噪声、覆盖和近重复，数字、表和图 |
| 图 7(b), (c)，$R^2$ 随训练月份的变化和规则违反率 | `results/results_taxi.json`，`results/results_taxi_monthly.json` | `big/run_taxi.py`，`figures/plot_results.py` | Rieth TEP 和出租车记录，数字、表和图 |
| 图 7(d)，训练规模和时间 | `results/results_taxi.json`，`results/results_taxi_100k.json`，`results/results_taxi_sizes.json` | `big/run_taxi.py`，`figures/plot_results.py` | Rieth TEP 和出租车记录，数字、表和图 |
| 图 8(a), (b)，C4 上的筛选和模型规模 | `results/results_corpus.jsonl` | `corpus/run_corpus.py`，`figures/plot_results.py` | 网页语料和 DCLM，数字、表和图 |
| 图 8(c)，合成图像 | `results/results_synth.jsonl` | `study/run_synth.py`，`figures/plot_results.py` | 合成图像、保真度和图像版本的回放，数字、表和图 |
| 图 8(d)，图文数据的筛选 | `results/results_datacomp.jsonl`，`results/results_mirflickr.jsonl` | `native/run_datacomp.py`，`native/run_mirflickr.py`，`figures/plot_results.py` | COCO 图文对和图文数据池，数字、表和图 |
| 表 1，主要数据质量标准和规范 | `docs/standards.md` | | |
| 表 2，五类测评方法的比较 | `docs/framework_zh.md` | | |
| 表 3，按数据类别汇总的公开数据集 | `results/table_sizes.json`，`results/results_sizes_native.json`，`results/label_errors.json`，`results/results_audioset_labels.json`，`results/case_tep.json`，`results/results_rieth.json`，`results/results_taxi.json`，`results/corpus_stats.json`，`results/results_cleanml.jsonl`，`results/results_datacomp.jsonl`，`results/results_wilds.jsonl`，`results/results_multimodal.jsonl` | `analysis/datasets.py`，`analysis/big_tables.py` | `python analysis/print_tables.py --latex` |
| 表 4，全部数据集的测评结果 | `results/tables_summary.json`，`results/static_native.json`，`results/corpus_static.json`，`results/dataset_docs.json`，`results/results_decision_fine.jsonl`，`results/results_versions.jsonl`，`results/results_cleanml.jsonl`，`results/results_corpus.jsonl`，`results/results_taxi.json`，`results/results_datacomp.jsonl`，`results/results_mirflickr.jsonl`，`results/case_tep.json`，`results/results_rieth.json`，`results/results_wilds.jsonl`，`results/results_synth.jsonl`，`results/results_vision.jsonl`，`results/results_fair.json`，`results/results_multimodal.jsonl`，`results/label_errors.json` | `analysis/datasets.py`，`analysis/big_tables.py` | `python analysis/print_tables.py --latex` |
| 表 5，测评类文件中要求条目的测评层级 | `data/requirements/items.jsonl` | `analysis/requirement_levels.py` | `python analysis/requirement_levels.py` |
| 正文中的数字 | `results/numbers_*.json`，`results/tables_summary.json`，`results/insights_summary.json`，`results/results_random_delete.jsonl` | `analysis/summaries.py`，`analysis/insights.py`，`study/run_random_delete.py`，`analysis/make_numbers.py` | 数字、表格和图 |
| 论文中计算得到的数字 | | `hqd_assess/`，`tests/test_paper_numbers.py` | `pytest tests/test_paper_numbers.py` |

## 结果

`results/` 存放每组测评的结果记录，JSONL 文件每行一次运行，JSON 文件中为运行列表。`docs/parameters.md` 给出每个文件的已计分运行数。`results/numbers_*.json` 存放正文中的数字，`results/tables_summary.json` 存放表格测评的汇总，`results/audit_native.json` 存放原生数据上各组测评与作业清单的比较，`results/audit_cleanml_published.json` 存放 CleanML 运行的准确率与已发表数值的对照，`results/results_detectors.jsonl`、`results/detectors_summary.json` 和 `results/sanity_detectors.json` 存放检测器估计及其汇总和检查结果，`results/dataset_docs.json` 存放说明文档得分及其原文。`results/results_civilcomments.jsonl` 与 `results/results_wilds.jsonl` 中的 CivilComments 行相同。带 `error` 字段的行是未计分的运行，例如删除后不足 30 条记录。

图像测评保存的测试预测和语料文档的评分需要 GPU 时间重新计算，它们放在 GitHub Release [v1.0](https://github.com/qzkinhit/HQDataset-Assessment/releases/tag/v1.0) 中。`scripts/download_data.sh DIR release` 下载并解包这些文件，[docs/release_assets.md](docs/release_assets.md) 列出其内容。

## 许可

代码、文档和结果记录采用 `LICENSE` 中的 MIT 许可。数据遵循各自来源的条款，`docs/datasets.md` 逐个列出。`data/tep/LICENSE` 是经典 TEP 分发文件的许可文本。

## 引用

```bibtex
@article{hqd_assessment,
  title   = {面向任务的多模态高质量数据集测评研究论文},
  author  = {Zekai Qian and Xiaoou Ding and Hongzhi Wang and Chen Wang},
  journal = {Scientia Sinica Informationis},
  year    = {2026}
}
```

## 基准的参考文献

- M. Abdelaal, C. Hammacher, H. Schöning. REIN: a comprehensive benchmark framework for data cleaning methods in ML pipelines. The 26th International Conference on Extending Database Technology, pp. 499-511, 2023.
- X. Ding, Z. Qian, H. Wang, et al. UniClean: a scalable data cleaning solution for mixed errors based on unified cleaners and optimized cleaning workflow. Proc. VLDB Endow., vol. 18, no. 11, pp. 4117-4130, 2025. 基准见 [UniClean-bench-Result](https://github.com/qzkinhit/UniClean-bench-Result)。
- P. Li, X. Rao, J. Blase, et al. CleanML: a study for evaluating the impact of data cleaning on ML classification tasks. 37th IEEE International Conference on Data Engineering, pp. 13-24, 2021.
