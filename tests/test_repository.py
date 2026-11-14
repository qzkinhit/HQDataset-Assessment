"""The download script, the table printers and the documents of the repository."""
import csv
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(*parts):
    return open(os.path.join(ROOT, *parts), encoding="utf-8").read()


def test_download_script_lists_every_dataset():
    if shutil.which("bash") is None:
        pytest.skip("bash is absent")
    out = subprocess.run(["bash", os.path.join(ROOT, "scripts", "download_data.sh"), "dl"], capture_output=True, text=True, env=dict(os.environ, DRY_RUN="1"), timeout=60)
    assert out.returncode == 0, out.stderr
    for piece in ("TEP_Faulty_Training.RData", "yellow_tripdata_2024-12.parquet", "CIFAR-100_human.pt", "duplicates_cifar100.csv", "val2017.zip", "c4-00002.json.gz", "wiki-00001.parquet", "build_text_pool.py",
                  "imdb_index.json", "imagenet_index.json", "ddpm_head.bin", "CC_shard_00000002", "CleanML-datasets-2020.zip", "b32_part.bin", "mirflickr25k.zip", "civilcomments_raw.parquet", "prep_camelyon.py", "amazon5core.tar.gz-partab"):
        assert piece in out.stdout, piece
    assert "CC_shard_00000003" not in out.stdout


def test_download_script_rejects_an_unknown_set():
    if shutil.which("bash") is None:
        pytest.skip("bash is absent")
    out = subprocess.run(["bash", os.path.join(ROOT, "scripts", "download_data.sh"), "dl", "nothing"], capture_output=True, text=True, env=dict(os.environ, DRY_RUN="1"), timeout=60)
    assert out.returncode == 2


def test_table_printer_skips_absent_files(tmp_path):
    script = os.path.join(ROOT, "analysis", "print_tables.py")
    out = subprocess.run([sys.executable, script, "--results", str(tmp_path)], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ""


def test_table_printer_reads_the_stored_results():
    results = os.path.join(ROOT, "results")
    if not os.path.exists(os.path.join(results, "case_tep.json")):
        pytest.skip("results are absent")
    script = os.path.join(ROOT, "analysis", "print_tables.py")
    out = subprocess.run([sys.executable, script, "--results", results, "--only", "tep"], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "Classic TEP" in out.stdout


def test_table_printer_writes_the_latex_tables():
    results = os.path.join(ROOT, "results")
    if not os.path.exists(os.path.join(results, "case_tep.json")):
        pytest.skip("results are absent")
    script = os.path.join(ROOT, "analysis", "print_tables.py")
    out = subprocess.run([sys.executable, script, "--results", results, "--latex"], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.count("\\begin{table}") == 2


def test_stored_runs_table(tmp_path):
    (tmp_path / "results_a.jsonl").write_text(json.dumps(dict(seed=0)) + "\n" + json.dumps(dict(seed=1)) + "\n" + json.dumps(dict(seed=2)) + "\n" + json.dumps(dict(seed=3, error="x")) + "\n")
    (tmp_path / "results_b.json").write_text(json.dumps(dict(runs=[dict(seed=4)])))
    doc = tmp_path / "doc.md"
    doc.write_text("text\n<!-- STORED RUNS BEGIN -->\nold\n<!-- STORED RUNS END -->\nend\n")
    out = subprocess.run([sys.executable, os.path.join(ROOT, "analysis", "stored_runs.py"), "--results", str(tmp_path), "--update", str(doc)], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert "| `results_a.jsonl` | 3 | 0 to 2 |" in out.stdout
    assert "| `results_b.json` | 1 | 4 |" in out.stdout
    assert "old" not in doc.read_text() and doc.read_text().endswith("<!-- STORED RUNS END -->\nend\n")


def survey_datasets():
    spec = importlib.util.spec_from_file_location("survey_datasets", os.path.join(ROOT, "analysis", "datasets.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.all_datasets()


def test_every_dataset_of_the_survey_has_an_entry():
    index = read("docs", "datasets.md").split("## Index")[1].split("\n## ")[0]
    rows = [line.split("|")[1] for line in index.splitlines() if line.startswith("| ")]
    names = {name.strip() for row in rows for name in re.split(r", (?=[A-Z0-9])", row)}
    names |= {"CIFAR-10, EDM", "CIFAR-10, DDPM", "TEP, classic", "TEP, large"}
    missing = [d["name"] for d in survey_datasets() if d["name"] not in names and d["name"] not in index]
    assert not missing, missing


def test_each_row_of_the_dataset_index_has_a_source_a_license_and_a_way_to_obtain():
    index = read("docs", "datasets.md").split("## Index")[1].split("\n## ")[0]
    rows = [line for line in index.splitlines() if line.startswith("| ") and not line.startswith("| Dataset")]
    assert len(rows) >= 30
    for row in rows:
        cells = [c.strip() for c in row.split("|")[1:-1]]
        assert len(cells) == 5, row
        assert "http" in cells[1], row
        assert cells[2], row
        assert cells[4].startswith(("`data/", "`download_data.sh", "`study/")), row


def test_every_local_link_of_the_documents_exists():
    for name in ("README.md", "README_zh.md", "data/README.md", "docs/datasets.md", "docs/framework.md", "docs/framework_zh.md", "docs/parameters.md", "docs/published_results.md", "docs/release_assets.md", "docs/standards.md", "docs/tools.md"):
        folder = os.path.dirname(os.path.join(ROOT, name))
        for target in re.findall(r"\]\((?!https?:|#)([^)#]+)", read(name)):
            assert os.path.exists(os.path.join(folder, target)), (name, target)


def test_every_script_named_in_the_readme_exists():
    for name in ("README.md", "README_zh.md"):
        for path in set(re.findall(r"\b(?:analysis|big|case_tep|corpus|examples|figures|hqd_assess|native|scripts|study|tests)/[\w./]+\.(?:py|sh)\b", read(name))):
            assert os.path.exists(os.path.join(ROOT, path)), (name, path)


def test_the_readmes_map_each_figure_and_table():
    for name, figure, table in (("README.md", "Figure {}", "Table {}"), ("README_zh.md", "图 {}", "表 {}")):
        text = read(name)
        for i in range(1, 9):
            assert figure.format(i) in text, (name, i)
        for i in range(1, 6):
            assert table.format(i) in text, (name, i)
        assert figure.format(9) not in text and table.format(6) not in text, name


def test_plain_text_rules_of_the_readmes():
    for name in ("README.md", "README_zh.md"):
        prose = re.sub(r"```.*?```", "", read(name), flags=re.S)
        assert ";" not in prose and "；" not in prose and "—" not in prose and "–" not in prose, name


def test_no_local_paths_in_the_code_and_the_documents():
    pattern = re.compile(r"/(?:home|Users)/\w|/backup\d+/\w")
    skip = {"results", "data", "__pycache__", ".git", ".pytest_cache"}
    for folder, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in skip]
        for name in files:
            if name.endswith((".py", ".md", ".sh", ".toml", ".txt")):
                assert not pattern.search(open(os.path.join(folder, name), encoding="utf-8").read()), name


def test_cifair_lists_match_the_stored_shares():
    results = os.path.join(ROOT, "results", "results_fair.json")
    if not os.path.exists(results):
        pytest.skip("results are absent")
    stored = json.load(open(results))
    for name in ("cifar10", "cifar100"):
        ids = {row["TestID"] for row in csv.DictReader(open(os.path.join(ROOT, "data", "cifair", f"duplicates_{name}.csv")))}
        assert len(ids) == stored[name]["n"]


def test_reviewer_votes_match_the_stored_counts():
    results = os.path.join(ROOT, "results", "label_errors.json")
    if not os.path.exists(results):
        pytest.skip("results are absent")
    stored = json.load(open(results))
    for name, entry in stored.items():
        assert len(json.load(open(os.path.join(ROOT, "data", "label_errors", f"{name}_mturk.json")))) == entry["flagged"], name


def test_static_rates_of_the_corpus_on_small_arrays(tmp_path):
    np = pytest.importorskip("numpy")
    np.savez(tmp_path / "c4.npz", alpha=np.array([.9, .9, .1, .9]), rep=np.array([.1, .1, .1, .9]), stop=np.array([.5, .5, .5, .5]), words=np.array([60, 60, 60, 60]))
    np.save(tmp_path / "c4_group.npy", np.array([0, 0, 2, 3]))
    out = tmp_path / "out.json"
    done = subprocess.run([sys.executable, os.path.join(ROOT, "corpus", "static_corpus.py"), str(tmp_path), str(out)], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    stored = json.load(open(out))
    assert stored == dict(rule_fail=0.5, near_duplicate=0.5, docs=4)

