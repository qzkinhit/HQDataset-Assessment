"""The examples run on the data of the repository."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(script, *args):
    out = subprocess.run([sys.executable, os.path.join(ROOT, "examples", script), *args], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_tep_example_reports_the_scan_and_the_claims():
    text = run("tep_assessment.py", "--no-task")
    assert "31700" in text and "d00.dat" in text and "0.841" in text
    assert "Quality claims" in text and "L3" in text


def test_table_example_reports_the_rates_and_the_claims():
    text = run("table_claims.py", "--no-task")
    assert "38.8%" in text and "95.2" in text
    assert "Quality claims" in text and "Algorithm 1" in text


def test_table_example_handles_a_table_without_clean_records():
    text = run("table_claims.py", "--table", "beers", "--model", "linear")
    assert "100.0%" in text and "delete" not in text.split("Quality claims")[0].split("Choice of a training version")[1]
