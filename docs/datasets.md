# Datasets of the experiments

The experiments prefer errors from the collection process or from the benchmark itself, with a clean reference or an accepted split. The text pool and the COCO image-text pairs carry constructed defects, because their experiments need a known dose or a truth per pair.

Small files are in `data/`, namely the 11 tables, the classic TEP files, the ciFAIR lists and the reviewer votes of the test-set label errors. `scripts/download_data.sh DIR` downloads the other files into the layout that the scripts read, and `scripts/download_data.sh DIR NAME` selects one set. The large files are also packed as release assets, see [release_assets.md](release_assets.md).

The license column repeats what the source states. "None stated" means that the source page, repository or archive holds no license statement for the data. It is the reader's task to check the terms before reuse.

## Index

| Dataset | Official source | License as stated by the source | Use | How to obtain |
|---|---|---|---|---|
| Adult | [UCI](https://archive.ics.uci.edu/dataset/2/adult), through REIN | CC BY 4.0 | Classification of `income`, label flips | `data/tables/adult` |
| Beers | [Kaggle craft-cans](https://www.kaggle.com/datasets/nickhould/craft-cans), through REIN | "Database: Open Database, Contents: Database Contents" | Classification of `style`, format and value errors | `data/tables/beers` |
| Breast Cancer | [UCI](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original), through REIN | CC BY 4.0 | Classification of `class`, format and value errors | `data/tables/breast_cancer` |
| Flights | [Dong et al.](http://lunadong.com/fusionDataSets.htm), through UniClean | None stated | Classification of `arrival_delay_bucket`, wrong web sources | `data/tables/flights` |
| Hospitals | [Raha](https://github.com/BigDaMa/raha), through UniClean | None stated for the data, the repository license names the code | Classification of `Condition`, replaced characters | `data/tables/hospitals` |
| SmartFactory | [Kaggle](https://www.kaggle.com/datasets/inIT-OWL/high-storage-system-data-for-energy-optimization), through REIN | CC BY-NC-SA 4.0 | Classification of `labels`, missing values | `data/tables/smartfactory` |
| Soccer | [BART](https://db.unibas.it/projects/bart/), through UniClean | None stated for the data, the repository license names the code | Classification of `manager`, wrong values | `data/tables/soccer_full` |
| Bike | [UCI](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), through REIN | CC BY 4.0 | Regression of `cnt`, out-of-domain values and substitutions | `data/tables/bike` |
| Mercedes | [Kaggle competition](https://www.kaggle.com/competitions/mercedes-benz-greener-manufacturing/data), through REIN | Competition rules, "Participants must use the Data solely for the purpose and duration of the Competition" | Regression of `y`, flips and empty cells | `data/tables/mercedes` |
| NASA | [UCI](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise), through REIN | CC BY 4.0 | Regression of `sound_pressure_level`, format errors | `data/tables/nasa` |
| Soil Moisture | [Riese and Keller](https://github.com/felixriese/hyperspectral-soilmoisture-dataset), through REIN | CC BY 4.0 in the README and LICENSE, BSD-3-Clause in CITATION.cff, other-open on Zenodo | Regression of `soil_moisture`, out-of-domain values | `data/tables/soilmoisture` |
| Airbnb, Citation, Company, Credit, EEG, KDD, Marketing, Movie, Restaurant, Sensor, Titanic, University, USCensus | [CleanML](https://github.com/chu-data-lab/CleanML) | None stated for the code or the archive | Classification on each cleaned version of the benchmark | `download_data.sh DIR cleanml` |
| CIFAR-10N, CIFAR-100N | [CIFAR-N](https://github.com/UCSC-REAL/cifar-10-100n), images from [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html) | Labels CC BY-NC 4.0, images none stated | Repair and deletion of human label errors | `download_data.sh DIR cifar` |
| C4 | [Dodge et al.](https://huggingface.co/datasets/allenai/c4), [release note](https://github.com/allenai/allennlp/discussions/5056) | ODC-BY | Corpus filters at a fixed token budget | `download_data.sh DIR corpus` |
| DCLM | [Pool](https://huggingface.co/datasets/mlfoundations/dclm-pool-1b-1x), [baseline](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) | CC BY 4.0 | The filtered baseline against the unfiltered pool on the C4 recipe | `download_data.sh DIR dclm` |
| DataComp | [Small pool](https://huggingface.co/datasets/mlfoundations/datacomp_small) | CC BY 4.0 for the samples and metadata, "The individual images are under their own copyrights" | Filters on image-text pairs, retrieval on COCO | `download_data.sh DIR datacomp` |
| MIRFLICKR | [MIRFLICKR-25000](https://press.liacs.nl/mirflickr/) | The paper states that the collection holds only images with Creative Commons attribution licenses | Filters on image-tag pairs, retrieval on COCO | `download_data.sh DIR mirflickr` |
| CIFAR-10, EDM | [Wang et al.](https://github.com/wzekai99/DM-Improves-AT) | MIT on the [Hugging Face repository](https://huggingface.co/wzekai99/DM-Improves-AT) that the script reads, Apache-2.0 on the [copy](https://huggingface.co/datasets/P2333/DM-Improves-AT) | Synthetic images replace or extend the training set | `download_data.sh DIR synthetic` |
| CIFAR-10, DDPM | [Gowal et al.](https://github.com/google-deepmind/deepmind-research/tree/master/adversarial_robustness) | None stated for the data, the repository license names the code | Synthetic images replace or extend the training set | `download_data.sh DIR synthetic` |
| CIFAKE | [Bird and Lotfi](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images), with the [copy](https://huggingface.co/datasets/yanbax/CIFAKE_autotrain_compatible) that the script reads | "published under the same MIT license as CIFAR-10", MIT tag on the copy | Synthetic images replace or extend the training set | `download_data.sh DIR synthetic` |
| NYC taxi | [TLC trip records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) | None stated on the TLC page, NYC Open Data writes "The license for this dataset is unspecified" | Rule violations and $R^2$ over 72 months | `download_data.sh DIR taxi` |
| TEP, classic | [Braatz group](http://web.mit.edu/braatzgroup/links.html), archive [TE_process.zip](http://web.mit.edu/braatzgroup/TE_process.zip) | The license text of the distribution names the software and does not name the data files | Split, window length and labels | `data/tep` |
| TEP, large | [Rieth et al.](https://doi.org/10.7910/DVN/6C3JR1) | Public domain dedication in the terms of the Dataverse page, "dedicated the work to the public domain" | Independent runs at a fixed number of records | `download_data.sh DIR rieth` |
| Camelyon17 | [WILDS](https://wilds.stanford.edu/datasets/), with the [copy](https://huggingface.co/datasets/wltjr1007/Camelyon17-WILDS) that the script reads | "This dataset is in the public domain and is distributed under CC0", tag `other` on the copy | Score on the training hospitals against the test hospital | `download_data.sh DIR wilds` |
| CivilComments | [WILDS](https://wilds.stanford.edu/datasets/), with the [copy](https://huggingface.co/datasets/pietrolesci/civilcomments-wilds) that the script reads | "This dataset is in the public domain and is distributed under CC0", none stated on the copy | Average accuracy against worst-group accuracy | `download_data.sh DIR wilds` |
| CIFAR-10.1 | [Recht et al.](https://github.com/modestyachts/CIFAR-10.1) | "The image data is part of the Tiny Images dataset and can be used the same way as the Tiny Images dataset" | Second test set of the synthetic and vision runs | `download_data.sh DIR synthetic` |
| CIFAR-100 coverage | [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html) | None stated | Missing fine classes against the same loss of scale | `study/prep_vision.py` |
| ciFAIR-10, ciFAIR-100 | [Barz and Denzler](https://cvjena.github.io/cifair) | None stated | Accuracy with and without near duplicates in training | `data/cifair`, `download_data.sh DIR cifair` |
| C4 held-out set | [Dodge et al.](https://huggingface.co/datasets/allenai/c4) | ODC-BY | Loss on documents without a near duplicate in the pool | `download_data.sh DIR corpus` |
| MNIST, CIFAR-10, CIFAR-100, Caltech-256, ImageNet, QuickDraw, 20 News, IMDB, Amazon, AudioSet | [label-errors](https://github.com/cleanlab/label-errors) of Northcutt et al. | Apache-2.0 on the repository, which holds labels, predictions and votes | Accuracy against given and corrected test labels | `data/label_errors`, `download_data.sh DIR labelerrors` |
| COCO | [COCO](https://cocodataset.org/#download) | Annotations under a Creative Commons Attribution 4.0 License, "Use of the images must abide by the Flickr Terms of Use" | Constructed mismatches for the detector estimates, clean retrieval test of the image-text pools | `download_data.sh DIR coco` |

Other files that the scripts read are listed in [Reference data and models](#reference-data-and-models).

## The 11 tables

Each table has a dirty version and a clean reference with the same `index` column, in `data/tables/NAME/dirty_index.csv` and `clean_index.csv`, with a rule library in `rules.txt`. `data/tables/table_meta.json` gives the task, the target and the categorical columns. No error is injected by this repository, and repairs copy values from the clean reference. Missing values are written as `empty`.

| Table | Task | Target | Rows | Features | Source benchmark | Underlying data | Errors in the dirty table |
|---|---|---|---|---|---|---|---|
| `adult` | Classification | `income` | 45222 | 14 | REIN | UCI Adult | Label flips in both directions |
| `beers` | Classification | `style` | 2410 | 9 | REIN | Beers table of the cleaning literature | Unit text in `ounces`, format errors in `abv` and `city`, missing `state` |
| `breast_cancer` | Classification | `class` | 699 | 9 | REIN | UCI Breast Cancer Wisconsin | Doubled digits, letters inside numbers, missing cells, label flips |
| `flights` | Classification | `arrival_delay_bucket` | 2376 | 6 | UniClean | Flights table of the cleaning literature | Native errors, flight times that are missing, malformed or wrong across web sources |
| `hospitals` | Classification | `Condition` | 1000 | 18 | UniClean | Hospital table of the cleaning literature | Native errors, characters replaced by `x` in 16 columns |
| `smartfactory` | Classification | `labels` | 23645 | 18 | REIN | Industrial smart factory sensors | Missing values in 30 percent of the rows |
| `soccer_full` | Classification | `manager` | 200000 | 9 | UniClean | Soccer table of the cleaning literature | Wrong values in `surname`, `position` and `city`, out-of-domain `season` |
| `bike` | Regression | `cnt` | 17379 | 15 | REIN | UCI Bike Sharing | Out-of-domain values, float injection, in-domain substitution |
| `mercedes` | Regression | `y` | 4209 | 376 | REIN | Mercedes-Benz Greener Manufacturing | Flips of binary columns, empty categorical cells |
| `nasa` | Regression | `sound_pressure_level` | 1503 | 5 | REIN | UCI Airfoil Self-Noise | Format errors, digit concatenation, empty values |
| `soilmoisture` | Regression | `soil_moisture` | 679 | 127 | REIN | Hyperspectral soil moisture | Out-of-domain values and substitutions across spectral bands |

The next table gives the state of the raw tables as means over the training records of 20 seeds, from `analysis/print_tables.py --only tables`. The rates are shares of records in percent, except for wrong cells, which is a share of cells. The static score follows the formula of GB/T 48562-2026.

| Table | Wrong cells | Wrong records | Missing | Out of domain | In-domain value | Label | Static score |
|---|---|---|---|---|---|---|---|
| `adult` | 0.0 | 3.8 | 0.0 | 0.0 | 0.0 | 3.8 | 100.0 |
| `beers` | 15.5 | 100.0 | 5.3 | 100.0 | 0.0 | 0.1 | 86.8 |
| `breast_cancer` | 8.1 | 57.0 | 29.6 | 20.6 | 22.0 | 2.1 | 91.0 |
| `flights` | 34.6 | 80.2 | 48.8 | 60.2 | 12.4 | 0.0 | 84.8 |
| `hospitals` | 2.6 | 38.8 | 0.0 | 38.8 | 0.0 | 0.0 | 95.2 |
| `smartfactory` | 1.7 | 30.0 | 30.0 | 0.0 | 0.0 | 0.0 | 96.2 |
| `soccer_full` | 1.7 | 15.7 | 0.0 | 2.4 | 13.2 | 0.0 | 98.0 |
| `bike` | 12.8 | 94.0 | 0.0 | 74.2 | 54.6 | 0.0 | 83.9 |
| `mercedes` | 19.0 | 100.0 | 36.0 | 100.0 | 100.0 | 0.0 | 70.5 |
| `nasa` | 9.7 | 48.6 | 2.0 | 43.2 | 3.5 | 0.0 | 93.9 |
| `soilmoisture` | 30.1 | 100.0 | 29.2 | 100.0 | 100.0 | 0.0 | 71.4 |

- Official source. The REIN benchmark (Abdelaal et al., 2023) at https://github.com/mohamedyd/rein-benchmark, with its data at https://www.researchgate.net/publication/382496079_Datasets_of_REIN_Benchmark. The UniClean benchmark (Ding et al., 2025), with its link in the references of `README.md`. The underlying sources are in the index above.
- License. The REIN and UniClean repositories state no license. The terms of the underlying sources are in the index above.
- How to obtain. The files are in `data/tables/`.
- Scripts. `study/run_real_tables.py`, `study/run_tables_fine.py`, `study/run_decision.py`, `study/run_decision_fine.py`, `study/run_detectors.py`, `analysis/tables_summary.py` and `examples/table_claims.py`.

## The 13 CleanML tables

Tables with real errors and the cleaned versions of the CleanML benchmark (Li et al., 2021). The variants with injected wrong labels are not used.

| Table | Records | Error types | Underlying source named by the benchmark |
|---|---|---|---|
| Airbnb | 33145 | Duplicates, outliers, missing values | Collected by the authors of the benchmark, in the archive |
| Citation | 5005 | Duplicates | https://sites.google.com/site/anhaidgroup/useful-stuff/data |
| Company | 128889 | Inconsistency | https://www.kaggle.com/jacksapper/company-sentiment-by-location |
| Credit | 150000 | Outliers, missing values | https://www.kaggle.com/c/GiveMeSomeCredit/data |
| EEG | 14980 | Outliers | https://archive.ics.uci.edu/ml/datasets/EEG+Eye+State |
| KDD | 131329 | Missing values, outliers | KDD Cup 2014, in the archive |
| Marketing | 8993 | Missing values | https://sites.google.com/site/anhaidgroup/useful-stuff/data |
| Movie | 9329 | Duplicates, inconsistency | https://data.world/popculture/imdb-5000-movie-dataset and https://www.kaggle.com/tmdb/tmdb-movie-metadata |
| Restaurant | 12007 | Duplicates, inconsistency | https://sites.google.com/site/anhaidgroup/useful-stuff/data |
| Sensor | 62076 | Outliers | http://db.csail.mit.edu/labdata/labdata.html |
| Titanic | 891 | Missing values | https://www.kaggle.com/upendr/titanic-machine-learning-from-disaster/data |
| University | 210 | Inconsistency | https://archive.ics.uci.edu/ml/datasets/University |
| USCensus | 32561 | Missing values | https://archive.ics.uci.edu/ml/datasets/US+Census+Data+%281990%29 |

- Official source. https://github.com/chu-data-lab/CleanML at commit `92bcb77`, with the archive `CleanML-datasets-2020.zip` linked from its README. The list of sources is `DatasetDescriptions.pdf` of that repository.
- License. The repository and the archive state no license. The underlying sources keep their own terms.
- How errors arise. The errors are the native errors of the tables, as the benchmark defines them. `native/clean_cleanml.py` runs the cleaning code of the benchmark and writes the versions. Nothing is injected.
- How to obtain. `scripts/download_data.sh DIR cleanml`.
- Scripts. `native/clean_cleanml.py`, `native/run_cleanml.py`, `native/audit_cleanml_acc.py`, `native/static_native.py` and `native/audit_native.py`.

## Classic TEP

The Tennessee Eastman process simulation in the distribution of the Braatz group (Downs and Vogel, 1993, Russell et al., 2000).

- Official source. The archive http://web.mit.edu/braatzgroup/TE_process.zip on the group page http://web.mit.edu/braatzgroup/links.html. The repository holds the 44 files of this archive, as `data/tep/SHA256SUMS.txt` records. A copy is at https://github.com/camaramm/tennessee-eastman-profBraatz.
- License. `data/tep/LICENSE` is the license text of the distribution. It names the software and does not name the data files.
- Scale. 44 files, `d00` to `d21` for training and `d00_te` to `d21_te` for testing, 31700 records with 52 variables and 22 classes. A training file holds 480 records, or 500 for the normal class. A test file holds 960 records.
- How errors arise. Records are simulated and the files have no injected defect. The file `d00.dat` is stored with rows and columns swapped. A test fault file holds the fault from record 161, and labels taken from file names mark the 160 records before as faulty, which is 16.7 percent of the file. Faults 3, 9 and 15 are hard to detect, as Russell et al. report. The files carry no field for the operating mode, while the process defines six modes. The experiments change only the evaluation protocol.
- How to obtain. The files are in `data/tep/`.
- Scripts. `case_tep/run_case.py`, `case_tep/run_windows.py`, `case_tep/make_traces.py` and `examples/tep_assessment.py`.

## Rieth TEP

The extended simulation of Rieth et al. (2017), with 500 independent runs per class.

- Official source. https://doi.org/10.7910/DVN/6C3JR1 on Harvard Dataverse.
- License. The terms of the Dataverse page are a public domain dedication, "The person who owns, created, or contributed a work to the data or work provided here dedicated the work to the public domain".
- Scale. 15.33 million records in four R data files of about 1.4 GB. The 21 classes of the fault-present records hold 5.05 million training records and 8.48 million test records. A faulty training run holds 500 records with the fault after record 20, and a faulty test run holds 960 records with the fault after record 160.
- How errors arise. The records, labels and runs are as published. The experiments fix the records per class and change how many independent runs they come from, and they compare a split by record with a split by run.
- How to obtain. `scripts/download_data.sh DIR rieth`.
- Scripts. `big/run_rieth.py` and `big/run_rieth_models.py`.

## NYC TLC yellow taxi records

Trip records of the New York City Taxi and Limousine Commission.

- Official source. https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page.
- License. None stated on the TLC page. NYC Open Data writes "The license for this dataset is unspecified" for the same records.
- Scale. 72 monthly Parquet files of 2019 to 2024, with 259287888 records.
- How errors arise. The errors come from the collection process. A full scan checks each month against the data dictionary and upper limits on duration, distance, fare and speed. The share of records that break any rule moves between 4.2 and 23.7 percent per month. The relation between trip features and duration also changes over the months, which shows as a change of $R^2$.
- How to obtain. `scripts/download_data.sh DIR taxi`.
- Script. `big/run_taxi.py`.

## CIFAR-10N and CIFAR-100N

Human labels of the CIFAR training images (Wei et al., 2022).

- Official source. http://noisylabels.com and https://github.com/UCSC-REAL/cifar-10-100n for the labels, https://www.cs.toronto.edu/~kriz/cifar.html for the images.
- License. The labels are under Creative Commons Attribution-NonCommercial 4.0 in `LICENSE.md` of the repository. The CIFAR page states no license for the images.
- Scale. Two sets of 50000 training images with 10000 clean test images each. The experiments use the label sets `aggre_label`, `random_label1` and `worse_label` of CIFAR-10N with noise rates of 9.0, 17.2 and 40.2 percent, and `noisy_label` of CIFAR-100N with a noise rate of 40.2 percent.
- How errors arise. Crowd workers labelled the images, and the wrong labels are their real mistakes. A repair copies the true label and a deletion removes the image.
- How to obtain. `scripts/download_data.sh DIR cifar`, then `python study/prep_vision.py DIR DIR/cifar_n DIR/cifar-10-python.tar.gz`. CIFAR-100 comes through the Hugging Face dataset `uoft-cs/cifar100`.
- Scripts. `study/prep_vision.py`, `study/make_jobs.py`, `study/run_vision.py` and `native/replay_versions.py`.

## ciFAIR duplicate lists

Near duplicates of the CIFAR test images in the training sets (Barz and Denzler, 2020).

- Official source. https://cvjena.github.io/cifair and https://github.com/cvjena/cifair.
- License. None stated on the page or in the repository.
- Scale. 286 test images of CIFAR-10 and 891 test images of CIFAR-100, which are 2.9 and 8.9 percent of the test sets.
- How errors arise. The test sets of the benchmark hold these images. Nothing is injected.
- How to obtain. The two lists are in `data/cifair/`, and `scripts/download_data.sh DIR cifair` writes them to `DIR/cifair`.
- Script. `study/fair.py`, which reads the saved predictions of `study/run_vision.py`.

## CIFAR-100 superclasses

CIFAR-100 with its 20 superclasses of 5 fine classes each.

- Official source. https://www.cs.toronto.edu/~kriz/cifar.html.
- License. None stated on the page.
- Scale. 50000 training images and 10000 test images.
- How errors arise. All labels are correct. Coverage jobs train on 5 to 1 fine classes per superclass and test on all fine classes, following the BREEDS idea. Scale jobs train on a random subset of the same size, so that coverage and scale can be compared.
- How to obtain. `study/prep_vision.py` writes the superclass labels with the arrays of CIFAR-100.
- Scripts. `study/make_jobs.py` and `study/run_vision.py`.

## C4 shards with Wikipedia as reference

Real web text of the C4 corpus (Dodge et al., 2021), with one Wikipedia shard as a curated reference.

- Official source. https://huggingface.co/datasets/allenai/c4 and https://huggingface.co/datasets/wikimedia/wikipedia.
- License. C4 under ODC-BY, as the release note https://github.com/allenai/allennlp/discussions/5056 states. Wikipedia under CC BY-SA 3.0 and GFDL on its dataset card.
- Scale. The first three shards of 1024, `c4-train.00000` to `00002`, with 1068952 documents and 310576817 tokens after keeping the first 512 tokens of each document. The reference is the shard `train-00001-of-00041` of the `20231101.en` configuration. Preparation keeps its first 20000 articles over 500 characters, and a run scores 4000 of them.
- How errors arise. The text is as collected from the web. Nothing is injected. Filters by perplexity, heuristic rules and near-duplicate removal select the training text at a fixed token budget.
- How to obtain. `scripts/download_data.sh DIR corpus`, then `python corpus/prep_corpus.py --dir DIR/corpus --out DIR/corpus`. The tokenizer and the pretrained scorer are those of `HuggingFaceTB/SmolLM2-135M`.
- Scripts. `corpus/prep_corpus.py`, `corpus/static_corpus.py`, `corpus/make_jobs_corpus.py` and `corpus/run_corpus.py`.

## DCLM corpora

The unfiltered pool and the filtered baseline of the DataComp-LM benchmark (Li et al., 2024) on the training recipe of the corpus track.

- Official source. https://huggingface.co/datasets/mlfoundations/dclm-pool-1b-1x and https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0.
- License. CC BY 4.0 on both dataset cards.
- Scale. 3 shards of each corpus with 495393 documents of at least 200 characters together, tokenized up to 130 million tokens per corpus.
- How errors arise. The corpora are as published, and the baseline is the result of the filters of the benchmark.
- How to obtain. `scripts/download_data.sh DIR dclm`, then `python native/prep_dclm.py DIR/corpus/tok/tokenizer.json DIR DIR/corpus`.
- Scripts. `native/prep_dclm.py`, `native/make_jobs_native.py`, `corpus/run_corpus.py` and `native/static_native.py`.

## DataComp image-text pool

One shard of the small pool of DataComp (Gadre et al., 2023), with its CLIP ViT-B/32 features.

- Official source. https://huggingface.co/datasets/mlfoundations/datacomp_small and https://www.datacomp.ai.
- License. The dataset card writes "We distribute the image url-text samples and metadata under a standard Creative Common CC-BY-4.0 license. The individual images are under their own copyrights." The experiments read the table and the features and do not download images.
- Scale. The shard `0020f0cbd157d470aa56bea08e304b90` with 505671 pairs.
- How errors arise. The pairs are as crawled from the web. Nothing is injected. A filter keeps a share of the shard by CLIP similarity, a text rule or near-duplicate removal, and retrieval is scored on COCO.
- How to obtain. `scripts/download_data.sh DIR datacomp`.
- Script. `native/run_datacomp.py`.

## MIRFLICKR-25000

Images with their native Flickr user tags and manual concept annotations (Huiskes and Lew, 2008).

- Official source. https://press.liacs.nl/mirflickr/.
- License. The paper https://press.liacs.nl/mirflickr/mirflickr.pdf writes "The image collection presented here consists only of images with one of the Creative Commons attribution licenses." The license of each image is in the metadata of the archive.
- Scale. 25000 images, a pool of 20000 pairs and 5000 held-out images.
- How errors arise. The tags are as given by the users. Nothing is injected. The filters of the DataComp track keep a share of the pool, and retrieval is scored on COCO.
- How to obtain. `scripts/download_data.sh DIR mirflickr`, then `python native/embed_mirflickr.py --native DIR --out DIR/mirflickr_clip.npz --device cuda`.
- Scripts. `native/embed_mirflickr.py` and `native/run_mirflickr.py`.

## WILDS Camelyon17 and CivilComments

Real shifts between hospitals and between subgroups (Koh et al., 2021).

- Official source. https://wilds.stanford.edu/datasets/. The scripts read the copies https://huggingface.co/datasets/wltjr1007/Camelyon17-WILDS and https://huggingface.co/datasets/pietrolesci/civilcomments-wilds.
- License. The WILDS page writes "This dataset is in the public domain and is distributed under CC0" for both. The Camelyon17 copy carries the tag `other`, and the CivilComments copy states nothing.
- Scale. Camelyon17 has 455954 patches, of which `native/prep_camelyon.py` draws 80000 training patches and 10000 held-out patches of the training hospitals and 10000 patches of the test hospital. CivilComments has 448000 comments on the official split of 269038, 45180 and 133782.
- How errors arise. The shifts are real. Nothing is injected.
- How to obtain. `scripts/download_data.sh DIR wilds`, then `python native/run_civilcomments.py --mode features --native DIR`.
- Scripts. `native/prep_camelyon.py`, `native/run_camelyon.py` and `native/run_civilcomments.py`.

## Published synthetic CIFAR-10 images and CIFAR-10.1

Synthetic images replace or extend the real CIFAR-10 training images. Nothing is generated or injected by this repository.

- Official source. EDM images of Wang et al. (2023) at https://github.com/wzekai99/DM-Improves-AT, DDPM images of Gowal et al. (2021) at https://github.com/google-deepmind/deepmind-research/tree/master/adversarial_robustness, CIFAKE of Bird and Lotfi (2024) at https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images, with the copy https://huggingface.co/datasets/yanbax/CIFAKE_autotrain_compatible that the download script reads, and CIFAR-10.1 of Recht et al. (2019) at https://github.com/modestyachts/CIFAR-10.1.
- License. EDM, MIT on https://huggingface.co/wzekai99/DM-Improves-AT and Apache-2.0 on https://huggingface.co/datasets/P2333/DM-Improves-AT. DDPM, none stated for the images, and the Apache-2.0 license of the repository names the code. CIFAKE, the Kaggle page writes "This dataset is published under the same MIT license as CIFAR-10". CIFAR-10.1, the README writes "The image data is part of the Tiny Images dataset and can be used the same way as the Tiny Images dataset".
- Scale. 100000 images each for EDM and DDPM and 50000 for CIFAKE enter the arrays. CIFAR-10.1 v6 is a second test set of 2000 images collected by the original procedure of CIFAR-10.
- How errors arise. The generators define the distribution of the synthetic images. A run replaces a share of the real images or adds synthetic images, and it is scored on the real test set and on CIFAR-10.1.
- How to obtain. `scripts/download_data.sh DIR synthetic`, then `python native/prep_native.py DIR DIR/cifar10_real.npz`.
- Scripts. `native/prep_native.py`, `native/make_jobs_native.py`, `study/run_synth.py`, `native/fidelity.py` and `native/replay_versions.py`.

## Label errors of ten test sets

The test-set label errors of Northcutt et al. (2021) on MNIST, CIFAR-10, CIFAR-100, Caltech-256, ImageNet, QuickDraw, 20 News, IMDB, Amazon and AudioSet.

- Official source. https://github.com/cleanlab/label-errors.
- License. Apache-2.0 on the repository. The repository holds label arrays, predictions and reviewer votes, and the images and texts of the ten sets keep their own terms. The vision and audio files of these sets are not downloaded.
- Scale. The published predictions, original labels and reviewer votes of the ten sets, from 7532 records of 20 News to 50426266 records of QuickDraw.
- How errors arise. Test labels hold real mistakes. A flagged label counts as wrong when at least three of the five reviewers chose the suggested label and fewer than three kept the given one.
- How to obtain. The votes and the two index files are in `data/label_errors/`, and `scripts/download_data.sh DIR labelerrors` writes all files to `DIR/label_errors`. `scripts/download_data.sh DIR llmtexts` fetches the texts of IMDB, 20 News and Amazon for `llm/prep_labels.py`.
- Scripts. `native/label_errors.py`, `native/label_errors_audioset.py`, `llm/prep_labels.py` and `llm/annotate_labels.py`. The AudioSet row also reads the class list `class_labels_indices.csv` of https://research.google.com/audioset/download.html. That page places the dataset under CC BY 4.0 and the ontology under CC BY-SA 4.0.

## Five-domain text pool

A text pool over five domains for the fine-tuning track.

- Source. Pinned revisions of https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu for general text, https://huggingface.co/datasets/HuggingFaceTB/finemath for mathematics, https://huggingface.co/datasets/codeparrot/codeparrot-clean-valid for code, https://huggingface.co/datasets/yerevann/coco-karpathy for image captions, and https://huggingface.co/datasets/mstz/adult for table rows written as text.
- License. FineWeb-Edu and FineMath, ODC-BY on their dataset cards. Adult copy, the tag `cc` on its card. The code and caption copies state nothing on their cards.
- Scale. 25000 training records with 5000 per domain, and 2000 held-out records with 400 per domain. A run trains on 6000 records, so a run reads 8000 documents with the held-out set.
- How errors arise. Each domain holds 3000 high-quality records and 500 records each of four kinds. Truncation cuts a record to 80 to 149 characters. Template repeats a few records with a changed tail. Cross-domain labels text of the next domain as this domain. Low-tier takes the `finemath-3plus` tier for mathematics and a cut to 120 characters for the other domains. Leakage moves held-out documents into training, following Oren et al. (2024).
- How to obtain. `scripts/download_data.sh DIR textpool` runs `study/build_text_pool.py`, which compares the SHA-256 of the two files with the recorded values.
- Scripts. `study/build_text_pool.py`, `study/make_jobs.py` and `study/run_text.py`.

## COCO val2017

Image-text pairs of the COCO 2017 validation set, encoded once with CLIP ViT-B/32.

- Official source. https://cocodataset.org/#download.
- License. The terms of use at https://cocodataset.org/#termsofuse place the annotations under a Creative Commons Attribution 4.0 License and write "Use of the images must abide by the Flickr Terms of Use".
- Scale. 5000 images with the first caption of each, split into 4000 training pairs and 1000 test pairs by seed.
- How errors arise. Real mismatches have no truth per pair, so the experiment constructs defects by the protocol of the noisy correspondence literature (Huang et al., 2021). A defect is one of mismatch, missing modality, duplicate, coverage, scale and test-pair leakage. COCO is also the clean retrieval test of the DataComp and MIRFLICKR tracks.
- How to obtain. `scripts/download_data.sh DIR coco`, then `python study/embed_coco.py DIR`.
- Scripts. `study/embed_coco.py`, `study/make_jobs.py` and `study/run_multimodal.py`.

## Data of the auxiliary runner

`study/run_study.py` injects defects at known rates into the clean tables, classic TEP, CIFAR-10 CLIP features from `study/embed_cifar.py` and AG News from `study/prepare_data.py`. No number of the survey comes from this runner. AG News comes from https://huggingface.co/datasets/fancyzhx/ag_news, whose card carries the tag `unknown` for the license, through `scripts/download_data.sh DIR agnews`.

## Reference data and models

| Item | Source | License as stated by the source | Read by |
|---|---|---|---|
| Wikipedia shard | https://huggingface.co/datasets/wikimedia/wikipedia | CC BY-SA 3.0 and GFDL | `corpus/prep_corpus.py` |
| SmolLM2-135M | https://huggingface.co/HuggingFaceTB/SmolLM2-135M | Apache-2.0 | `corpus/run_corpus.py`, `study/run_text.py` |
| CLIP ViT-B/32 | https://huggingface.co/openai/clip-vit-base-patch32 | None stated on the model card | `study/embed_coco.py`, `study/embed_cifar.py`, `native/embed_mirflickr.py`, `native/fidelity.py` |
| Qwen 2.5 7B Instruct | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct | Apache-2.0 | `llm/qwen.py` |
