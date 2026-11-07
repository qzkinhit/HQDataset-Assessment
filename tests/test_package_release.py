"""The release packer on a small tree of fake files. Parts stay under the limit, checksums hold, logs lose their machine details."""
import hashlib
import json
import os
import subprocess
import sys
import tarfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "package_release.py")
# fake paths of a machine, built at run time
HOME, MAC, STORE = "/" + "home/someuser", "/" + "Users/someuser", "/" + "backup01/someuser"


def run(*args):
    return subprocess.run([sys.executable, SCRIPT, *map(str, args)], capture_output=True, text=True, timeout=120)


def join_parts(out, stem):
    names = sorted(n for n in os.listdir(out) if n.startswith(stem))
    return names, b"".join(open(os.path.join(out, n), "rb").read() for n in names)


@pytest.fixture
def tree(tmp_path):
    data = tmp_path / "work" / "data"
    (data / "big" / "rieth").mkdir(parents=True)
    (data / "old").mkdir()
    (data / "agnews").mkdir()
    payload = os.urandom(5000)
    (data / "big" / "rieth" / "TEP_Faulty_Training.RData").write_bytes(payload)
    (data / "old" / "cifar-10-python.tar.gz").write_bytes(b"not this one")
    (data / "agnews" / "train.csv").write_text("text,label\n")
    logs = tmp_path / "work" / "runs"
    (logs / "logs_vision").mkdir(parents=True)
    (logs / "attempts" / "20260102_0304_gpu" / "logs").mkdir(parents=True)
    (logs / "skipped").mkdir()
    private = tmp_path / "terms.txt"
    private.write_text("# names\nsomeuser=<user>\nmyproject=<project>\n")
    return dict(data=data, logs=logs, private=private, payload=payload, out=tmp_path / "out")


def test_parts_stay_under_the_limit_and_join_to_the_files(tree):
    done = run("--data", tree["data"], "--only", "rieth,agnews", "--out", tree["out"], "--max-bytes", 1000, "--label", "t")
    assert done.returncode == 0, done.stderr
    names, joined = join_parts(tree["out"], "hqd-data-rieth")
    assert len(names) > 1 and all(n.startswith("hqd-data-rieth.tar.part-") for n in names)
    assert all(os.path.getsize(os.path.join(tree["out"], n)) <= 1000 for n in names)
    target = tree["out"] / "joined.tar"
    target.write_bytes(joined)
    with tarfile.open(target) as tar:
        member = tar.getmember("rieth/TEP_Faulty_Training.RData")
        assert tar.extractfile(member).read() == tree["payload"]
        assert member.uname == "" and member.gname == "" and member.uid == 0 and member.mtime == 0


def test_a_group_that_fits_keeps_the_plain_name_and_all_checksums_hold(tree):
    done = run("--data", tree["data"], "--only", "agnews,rieth", "--out", tree["out"], "--max-gib", 1, "--label", "t")
    assert done.returncode == 0, done.stderr
    assert (tree["out"] / "hqd-data-agnews.tar").exists()
    lines = (tree["out"] / "SHA256SUMS-t.txt").read_text().splitlines()
    assert len(lines) == 2
    for line in lines:
        digest, name = line.split()
        assert hashlib.sha256((tree["out"] / name).read_bytes()).hexdigest() == digest
    manifest = json.loads((tree["out"] / "MANIFEST-t.json").read_text())
    member = manifest["assets"]["hqd-data-rieth"]["members"][0]
    assert member["sha256"] == hashlib.sha256(tree["payload"]).hexdigest()


def test_a_second_batch_with_the_same_label_extends_the_checksums(tree):
    assert run("--data", tree["data"], "--only", "agnews", "--out", tree["out"], "--label", "t").returncode == 0
    assert run("--data", tree["data"], "--only", "rieth", "--out", tree["out"], "--label", "t").returncode == 0
    assert len((tree["out"] / "SHA256SUMS-t.txt").read_text().splitlines()) == 2


def test_excluded_folders_are_not_searched(tree):
    done = run("--data", tree["data"], "--only", "cifar", "--out", tree["out"], "--dry-run")
    assert done.returncode == 0, done.stderr
    assert "cifar: 0 files" in done.stdout


def test_dry_run_writes_nothing(tree):
    done = run("--data", tree["data"], "--logs", tree["logs"], "--out", tree["out"], "--dry-run")
    assert done.returncode == 0, done.stderr
    assert "rieth: 1 files" in done.stdout
    assert not tree["out"].exists()


def test_logs_lose_paths_names_and_tokens(tree):
    text = "\n".join((
        f"reading {tree['logs']}/preds/a.npy",
        f"{HOME}/miniconda3/envs/myenv/lib/python3.8/site-packages/x.py:1: Warning",
        f"{STORE}/envs/other/bin/python run.py",
        f"{STORE}/hf_cache/hub/models",
        f"{HOME}/work/run.py and {MAC}/run.py and /mnt/disk2/run.py",
        "myproject/data/x and someuser@host.org and https_proxy=http://127.0.0.1:8888",
        "token hf_abcdefghijklmnopqrstuvwxyz",
        "relative data/tables and https://huggingface.co/datasets/data/x stay",
    ))
    (tree["logs"] / "logs_vision" / "v_cifar10_resnet18_s0_clean.log").write_text(text)
    done = run("--only", "logs", "--logs", tree["logs"], "--private", tree["private"], "--out", tree["out"], "--label", "t")
    assert done.returncode == 0, done.stderr
    with tarfile.open(tree["out"] / "hqd-logs-t.tar.gz") as tar:
        member = tar.getmember("logs/vision/v_cifar10_resnet18_s0_clean.log")
        out = tar.extractfile(member).read().decode()
        assert member.uname == "" and member.mtime == 0
    assert "$WORK/preds/a.npy" in out
    assert "$PYTHON_ENV/lib/python3.8/site-packages/x.py" in out
    assert "$PYTHON_ENV/bin/python" in out
    assert "$HF_HOME/hub/models" in out
    assert "$HOME/work/run.py and $HOME/run.py and $STORE/run.py" in out
    assert "<project>/data/x" in out and "<proxy>" in out and "<token>" in out
    assert "relative data/tables and https://huggingface.co/datasets/data/x stay" in out
    for word in ("someuser", "myproject", "/" + "home/", "/" + "Users/", "/" + "backup01", "miniconda", "8888", "hf_abc"):
        assert word not in out, word
    report = json.loads((tree["out"] / "scrub_report-t.json").read_text())
    assert report["logs"] == 1 and report["replacements"]["python_env"] == 2


def test_log_names_are_neutral_and_attempts_are_numbered(tree):
    logs = tree["logs"]
    (logs / "logs_vision" / "v_a_s0.log").write_text("first\n")
    os.utime(logs / "logs_vision" / "v_a_s0.log", (1, 1))
    (logs / "attempts" / "20260102_0304_gpu" / "logs" / "v_a_s0.log").write_text("second\n")
    (logs / "attempts" / "x_run.log").write_text(f"{logs}/attempts/20260102_0304_gpu/run.py\n")
    (logs / "pool_vision10_fix.log").write_text("pool\n")
    (logs / "threshold.log").write_text("name stays\n")
    (logs / "skipped" / "c_s0_all.log").write_text("left out\n")
    (logs / "pool_failed.log").write_text("left out\n")
    (logs / "empty.log").write_text("")
    done = run("--only", "logs", "--logs", logs, "--private", tree["private"], "--out", tree["out"], "--label", "t", "--exclude", "skipped", "--drop-folder", "attempts")
    assert done.returncode == 0, done.stderr
    with tarfile.open(tree["out"] / "hqd-logs-t.tar.gz") as tar:
        names = sorted(tar.getnames())
        assert tar.extractfile("logs/vision/v_a_s0.1.log").read() == b"first\n"
        assert tar.extractfile("logs/vision/v_a_s0.2.log").read() == b"second\n"
        assert tar.extractfile("logs/other/x_run.log").read() == b"$WORK/run.py\n"
    assert names == ["logs/other/threshold.log", "logs/other/x_run.log", "logs/pools/pool_vision10.log", "logs/vision/v_a_s0.1.log", "logs/vision/v_a_s0.2.log"]


def test_a_private_term_that_stays_stops_the_batch(tree):
    (tree["logs"] / "y.log").write_text("user someuser\n")
    tree["private"].write_text("someuser=someuser\n")
    done = run("--only", "logs", "--logs", tree["logs"], "--private", tree["private"], "--out", tree["out"], "--label", "t")
    assert done.returncode == 3
    assert not (tree["out"] / "hqd-logs-t.tar.gz").exists()


def test_the_limit_of_github_is_kept():
    assert run("--max-gib", 2, "--dry-run", "--only", "agnews").returncode == 2


def test_unknown_group_is_rejected():
    assert run("--only", "nothing", "--dry-run").returncode == 2


def test_groups_can_be_listed():
    done = run("--list")
    assert done.returncode == 0 and "rieth" in done.stdout and "logs" in done.stdout
