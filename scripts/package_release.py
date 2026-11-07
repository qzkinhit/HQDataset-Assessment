#!/usr/bin/env python3
"""Pack the large dataset files and the run logs into release archives below the 2 GB limit per asset, with checksums.

The script runs on the machine that holds the files and needs only the standard library. It searches the folders given by
--data for the files of each group in GROUPS and stores them under the layout of scripts/download_data.sh. Each group becomes
one tar stream, cut into parts below the limit. Logs from the folders given by --logs are rewritten before packing, so that
paths, user names, host names, addresses and tokens become neutral placeholders. A rewritten log that still holds a private
term stops the run. Tar headers carry no owner and no time.

Output in --out:
  hqd-data-GROUP.tar or hqd-data-GROUP.tar.part-NNN    dataset files
  hqd-logs-LABEL.tar.gz or .tar.gz.part-NNN            rewritten logs
  SHA256SUMS-LABEL.txt                                 checksums of all assets of this label
  MANIFEST-LABEL.json                                  parts and members of each asset with their SHA-256
  scrub_report-LABEL.json                              replacements per rule, without the replaced text

An archive is unpacked with cat hqd-data-GROUP.tar.part-* | tar -x -C downloads

Usage: package_release.py [--data DIR ...] [--logs DIR ...] [--out DIR] [--only NAME,...] [--skip NAME,...] [--private FILE]
                          [--label NAME] [--max-gib 1.8] [--exclude NAME ...] [--drop-folder NAME ...] [--dry-run] [--list]
A line of the --private file is a term or term=placeholder, for names that the rules cannot derive. The file stays outside
the repository. The current user name, the home folder and the host name are added to it.
"""
import argparse
import fnmatch
import getpass
import hashlib
import io
import json
import os
import re
import shutil
import socket
import sys
import tarfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GIB = 2 ** 30

# group -> (folder in the archive, tail of the source path as a glob), or ("tree", tail of a source folder, folder in the archive)
GROUPS = {
    "rieth": [("rieth", "rieth/TEP_*.RData")],
    "taxi": [("tlc", "tlc/yellow_tripdata_*.parquet")],
    "cifar": [("", "cifar-10-python.tar.gz"), ("cifar_n", "cifar_n/CIFAR-1*_human.pt"), ("", "cifar10_real.npz"), ("", "cifar100_real.npz"), ("", "cifar10_clip.npz")],
    "coco": [("coco", "coco/annotations.zip"), ("coco", "coco/val2017.zip"), ("", "coco_clip.npz")],
    "corpus": [("corpus", "corpus/c4-0000?.json.gz"), ("corpus", "corpus/wiki-00001.parquet"), ("corpus/tok", "tok/tokenizer.json"), ("corpus/tok", "tok/tokenizer_config.json"), ("corpus/tok", "tok/special_tokens_map.json"), ("corpus/tok", "tok/config.json")],
    "corpus-arrays": [("corpus", "corpus/c4.npz"), ("corpus", "corpus/c4_group.npy"), ("corpus", "corpus/c4_ppl.npy"), ("corpus", "corpus/c4_llm_score.npy"), ("corpus", "corpus/wiki.npz"), ("corpus", "corpus/dclm_pool.npz"), ("corpus", "corpus/dclm_baseline.npz")],
    "textpool": [("processed", "processed/qpool_train.jsonl"), ("processed", "processed/qpool_heldout.jsonl"), ("processed", "processed/pool_manifest.json"), ("", "text_ppl.npz")],
    "synthetic": [("", "edm_200k.npz"), ("", "ddpm_head.bin"), ("", "ddpm_tail.bin"), ("", "cifake_train.zip"), ("", "cifar10.1_v6_*.npy"), ("", "cifar10_1.npz"), ("", "synth_edm.npz"), ("", "synth_ddpm.npz"), ("", "synth_cifake.npz")],
    "labelerrors": [("label_errors", "*_cv_pred.npy"), ("label_errors", "*_orig.npy"), ("label_errors", "*_mturk.json"), ("label_errors", "imdb_index.json"), ("label_errors", "imagenet_index.json"), ("label_errors", "class_labels_indices.csv")],
    "dclm": [("dclm_pool", "dclm_pool/CC_shard_*.jsonl.zst"), ("dclm_base", "dclm_base/shard_*.jsonl.zst")],
    "cleanml": [("cleanml", "cleanml/CleanML-datasets-2020.zip")],
    "cleanml-versions": [("tree", "cleanml/data", "cleanml/repo/data")],
    "datacomp": [("datacomp", "datacomp/*.parquet"), ("datacomp", "datacomp/b32_part.bin")],
    "mirflickr": [("", "mirflickr25k.zip"), ("", "mirflickr25k_annotations_v080.zip"), ("", "mirflickr_clip.npz")],
    "wilds": [("wilds", "wilds/civilcomments_raw.parquet"), ("", "camelyon_x.npy"), ("", "camelyon_meta.npz")],
    "llmtexts": [("llm_texts", "imdb.tgz"), ("llm_texts", "20news-bydate.tar.gz"), ("llm_texts", "amazon5core.tar.gz-part*")],
    "agnews": [("agnews", "agnews/train.csv"), ("agnews", "agnews/test.csv")],
    "predictions": [("preds", "preds/*.npy"), ("preds", "preds_clean/*.npy")],
}
EXCLUDE_DIRS = {"__pycache__", ".git", "old", "wandb", ".cache"}
EXCLUDE_LOG_NAME = re.compile(r"(?i)failed|\.bad|\.bak|before")
TRACKS = [(re.compile(r"^v_"), "vision"), (re.compile(r"^t_s\d"), "text"), (re.compile(r"^m_s\d"), "multimodal"), (re.compile(r"^c_s\d"), "corpus"), (re.compile(r"^sy_"), "synthetic"),
          (re.compile(r"^cam_"), "camelyon"), (re.compile(r"^civil_"), "civilcomments"), (re.compile(r"^clean_"), "cleanml"), (re.compile(r"^llm_"), "llm")]
NAME_NOTE = re.compile(r"(?i)(?:^|[._-])(?:[a-z0-9]*fix|old|restart\d*|retry)(?=[._-]|$)")
PATH_CHAR = r"[^\s:\"'=,()\[\]{}<>]"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class PartWriter:
    """A binary stream cut into files of at most limit bytes, with the SHA-256 of each file."""

    def __init__(self, folder, name, limit):
        self.folder, self.name, self.limit = folder, name, limit
        self.parts, self.f, self.h, self.size = [], None, None, 0

    def _open(self):
        path = os.path.join(self.folder, f"{self.name}.part-{len(self.parts) + 1:03d}")
        self.f, self.h, self.size = open(path, "wb"), hashlib.sha256(), 0
        self.parts.append(dict(file=os.path.basename(path)))

    def _close(self):
        self.f.close()
        self.parts[-1].update(bytes=self.size, sha256=self.h.hexdigest())
        self.f = None

    def write(self, data):
        view = memoryview(data)
        while len(view):
            if self.f is None:
                self._open()
            chunk = view[:self.limit - self.size]
            self.f.write(chunk)
            self.h.update(chunk)
            self.size += len(chunk)
            view = view[len(chunk):]
            if self.size >= self.limit:
                self._close()
        return len(data)

    def close(self):
        """Close the last part. A stream that fits one part keeps the plain name."""
        if self.f is not None:
            self._close()
        if len(self.parts) == 1:
            os.replace(os.path.join(self.folder, self.parts[0]["file"]), os.path.join(self.folder, self.name))
            self.parts[0]["file"] = self.name
        return self.parts


class HashingReader:
    """Reads a file and computes its SHA-256 on the way."""

    def __init__(self, f):
        self.f, self.h = f, hashlib.sha256()

    def read(self, size=-1):
        data = self.f.read(size)
        self.h.update(data)
        return data


def header(name, size):
    """A tar header without owner, time or machine details."""
    info = tarfile.TarInfo(name)
    info.size, info.mode, info.mtime, info.uid, info.gid, info.uname, info.gname = size, 0o644, 0, 0, 0, "", ""
    return info


def walk(roots, exclude, with_root=False):
    """Files under the roots as (parts of the path relative to the root, full path), in the order of the roots."""
    for root in roots:
        top = [os.path.basename(os.path.abspath(root))] if with_root else []
        for folder, dirs, files in os.walk(root):
            dirs[:] = sorted(d for d in dirs if d not in exclude)
            rel = os.path.relpath(folder, root)
            parts = top + ([] if rel == "." else rel.split(os.sep))
            for name in sorted(files):
                yield parts + [name], os.path.join(folder, name)


def collect(specs, files):
    """Destination path to source path for one group. The first root that holds a destination wins. Also returns the specs without a file."""
    chosen, missing = {}, []
    for spec in specs:
        found = False
        if spec[0] == "tree":
            tail = spec[1].split("/")
            for parts, path in files:
                at = [i for i in range(len(parts) - len(tail)) if parts[i:i + len(tail)] == tail]
                if at:
                    chosen.setdefault("/".join([spec[2]] + parts[at[0] + len(tail):]), path)
                    found = True
        else:
            folder, pattern = spec
            tail = pattern.split("/")
            for parts, path in files:
                if len(parts) >= len(tail) and all(fnmatch.fnmatchcase(a, b) for a, b in zip(parts[-len(tail):], tail)):
                    chosen.setdefault("/".join(x for x in (folder, parts[-1]) if x), path)
                    found = True
        if not found:
            missing.append(spec[1])
    return chosen, missing


def make_rules(roots, private, drop_folders=()):
    """Rewriting rules in the order of application, as (name, regex, replacement)."""
    literal = sorted({os.path.realpath(r) for r in roots} | {os.path.abspath(r) for r in roots}, key=len, reverse=True)
    literal = [r for r in literal if r != os.sep]
    rules = [
        ("python_env", re.compile(rf"(?<![\w.\-])/(?:{PATH_CHAR}+?/)??(?:miniconda3|anaconda3|miniforge3|mambaforge|envs/{PATH_CHAR}+?)(?:/envs/[^/\s:\"'=,()\[\]{{}}<>]+)?(?=/|[\s:\"'=,()\[\]{{}}<>]|$)", re.M), "$PYTHON_ENV"),
        ("model_cache", re.compile(rf"(?<![\w.\-])/(?:{PATH_CHAR}+?/)??(?:hf_cache|\.cache/huggingface)(?=/|[\s:\"'=,()\[\]{{}}<>]|$)", re.M), "$HF_HOME"),
    ]
    if literal:
        rules.append(("work_folder", re.compile("|".join(re.escape(r) for r in literal)), "$WORK"))
    rules += [
        ("attempt_folder", re.compile("/(?:" + "|".join([re.escape(d) + r"(?:/\d{8}_\d+\w*)?" for d in drop_folders] + [r"\d{8}_\d+\w*"]) + ")(?=/)"), ""),
        ("home", re.compile(rf"(?<![\w.\-/:])/(?:home|Users)(?:/{PATH_CHAR}[^/\s:\"'=,()\[\]{{}}<>]*)?"), "$HOME"),
        ("home", re.compile(r"(?<![\w.\-/:])/root(?=/|\s|$)", re.M), "$HOME"),
        ("storage", re.compile(rf"(?<![\w.\-/:])/(?:backup|data|disk|mnt|media|scratch|nfs|work|storage|srv|Volumes)\d*(?:/{PATH_CHAR}[^/\s:\"'=,()\[\]{{}}<>]*)?"), "$STORE"),
    ]
    for term, placeholder in private:
        rules.append(("private_term", re.compile(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", re.I), placeholder))
    rules += [
        ("address", re.compile(r"\b(?:127\.0\.0\.1|localhost):\d+"), "<proxy>"),
        ("address", re.compile(r"(?<=://)\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?"), "<host>"),
        ("address", re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}:\d+\b"), "<host>"),
        ("email", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"), "<email>"),
        ("token", re.compile(r"\b(?:hf|ghp|gho|github_pat|sk)[_-][A-Za-z0-9_-]{20,}\b"), "<token>"),
    ]
    return rules


def residual_regex(private):
    """Text that must not remain after the rewriting."""
    terms = [rf"(?<![A-Za-z0-9]){re.escape(t)}(?![A-Za-z0-9])" for t, _ in private]
    return re.compile("|".join(terms + [r"(?<![\w.\-/:$])/(?:home|Users)/\w", r"(?<![\w.\-/:$])/backup\d*/\w"]), re.I)


def rewrite(text, rules, counts):
    for name, regex, replacement in rules:
        text, n = regex.subn(lambda _m, r=replacement: r, text)
        if n:
            counts[name] = counts.get(name, 0) + n
    return text


def read_private(path):
    """Terms from the file, then the user name, the host name and the name of the home folder of this machine."""
    out = []
    if path:
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#"):
                term, _, placeholder = line.partition("=")
                out.append((term.strip(), placeholder.strip() or "<private>"))
    user = getpass.getuser()
    out.append((user, "<user>"))
    home = os.path.basename(os.path.expanduser("~"))
    if home:
        out.append((home, "<user>"))
    host = socket.gethostname().split(".")[0]
    if len(host) >= 4:
        out.append((host, "<host>"))
    seen, unique = set(), []
    for term, placeholder in sorted(out, key=lambda x: -len(x[0])):
        if term and term.lower() not in seen:
            seen.add(term.lower())
            unique.append((term, placeholder))
    return unique


def log_target(parts, drop_folders=()):
    """The path of a log in the archive, by its track, without folder names of the machine."""
    stem = parts[-1][:-len(os.path.splitext(parts[-1])[1])]
    for regex, track in TRACKS:
        if regex.match(stem):
            return f"logs/{track}/{stem}.log"
    keep = [p for p in parts[:-1] if p not in drop_folders and not re.match(r"(?i)^(logs(_\w+)?|\d{8}_\d+\w*)$", p)]
    clean = NAME_NOTE.sub("", stem) or stem
    name = "_".join(keep + [clean])
    return f"logs/{'pools' if clean.startswith(('pool', 'ppl')) else 'other'}/{name}.log"


def collect_logs(roots, exclude, drop_folders=()):
    """Archive path to the sources of that path, oldest first. Several sources become numbered attempts."""
    targets = {}
    for parts, path in walk(roots, exclude):
        if not parts[-1].endswith((".log", ".out")) or EXCLUDE_LOG_NAME.search(parts[-1]) or os.path.getsize(path) == 0:
            continue
        targets.setdefault(log_target(parts, drop_folders), []).append(path)
    out = {}
    for target, paths in targets.items():
        paths.sort(key=os.path.getmtime)
        if len(paths) == 1:
            out[target] = paths[0]
        else:
            for i, path in enumerate(paths, 1):
                out[f"{target[:-4]}.{i}.log"] = path
    return out


def pack_data(name, chosen, out, limit):
    writer = PartWriter(out, f"{name}.tar", limit)
    members = []
    with tarfile.open(fileobj=writer, mode="w|", format=tarfile.GNU_FORMAT) as tar:
        for dest in sorted(chosen):
            size = os.path.getsize(chosen[dest])
            with open(chosen[dest], "rb") as f:
                reader = HashingReader(f)
                tar.addfile(header(dest, size), reader)
            members.append(dict(path=dest, bytes=size, sha256=reader.h.hexdigest()))
            print(f"  {dest} {size}", flush=True)
    return dict(parts=writer.close(), members=members, unpack=f"cat {name}.tar* | tar -x -C downloads")


def pack_logs(name, logs, rules, residual, out, limit, dry_run):
    counts, problems, members = {}, [], []
    writer = None if dry_run else PartWriter(out, f"{name}.tar.gz", limit)
    tar = None if dry_run else tarfile.open(fileobj=writer, mode="w|gz", format=tarfile.GNU_FORMAT)
    for dest in sorted(logs):
        text = open(logs[dest], encoding="utf-8", errors="replace").read()
        text = rewrite(text, rules, counts)
        left = residual.findall(re.sub(r"<\w+>", "", text))
        if left:
            problems.append(dict(path=dest, count=len(left)))
        data = text.encode("utf-8")
        members.append(dict(path=dest, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
        if tar is not None:
            tar.addfile(header(dest, len(data)), io.BytesIO(data))
    if tar is not None:
        tar.close()
    entry = dict(members=members, unpack=f"cat {name}.tar.gz* | tar -xz -C .")
    if writer is not None:
        entry["parts"] = writer.close()
    return entry, counts, problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", action="append", default=[], help="folder to search for dataset files, several allowed, default downloads in the repository")
    ap.add_argument("--logs", action="append", default=[], help="folder to search for logs, several allowed, default pool_logs in the repository")
    ap.add_argument("--out", default=os.path.join(REPO, "release_assets"))
    ap.add_argument("--only", default="", help="comma separated groups, the name logs selects the logs")
    ap.add_argument("--skip", default="", help="comma separated groups to leave out")
    ap.add_argument("--private", help="file with one private term per line, or term=placeholder")
    ap.add_argument("--label", default="main", help="neutral name of this batch, used in the names of the log archive and the checksum files")
    ap.add_argument("--max-gib", type=float, default=1.8, help="largest asset in GiB, below the limit of 2 GB per file of GitHub Releases")
    ap.add_argument("--max-bytes", type=int, help="largest asset in bytes, replaces --max-gib")
    ap.add_argument("--exclude", action="append", default=[], help="folder name to leave out, in addition to the built-in list")
    ap.add_argument("--drop-folder", action="append", default=[], help="folder name to remove from paths inside logs and from the names of logs, with a dated folder below it")
    ap.add_argument("--dry-run", action="store_true", help="print what would be packed and write nothing")
    ap.add_argument("--list", action="store_true", help="print the groups and exit")
    A = ap.parse_args(argv)
    if A.list:
        for name, specs in GROUPS.items():
            print(name, " ".join(s[1] for s in specs))
        print("logs *.log and *.out under --logs")
        return 0
    limit = A.max_bytes or int(A.max_gib * GIB)
    if limit > 2 * 10 ** 9 and A.max_bytes is None:
        ap.error("--max-gib must stay below the limit of 2 GB per asset")
    names = list(GROUPS) + ["logs"]
    only = [x for x in A.only.split(",") if x] or names
    skip = {x for x in A.skip.split(",") if x}
    unknown = (set(only) | skip) - set(names)
    if unknown:
        ap.error(f"unknown groups {sorted(unknown)}, see --list")
    selected = [n for n in names if n in only and n not in skip]
    data_roots = A.data or [os.path.join(REPO, "downloads")]
    log_roots = A.logs or [os.path.join(REPO, "pool_logs")]
    exclude = EXCLUDE_DIRS | set(A.exclude)
    groups = [n for n in selected if n != "logs"]
    files = list(walk(data_roots, exclude, with_root=True)) if groups else []
    plan = {}
    for name in groups:
        chosen, missing = collect(GROUPS[name], files)
        total = sum(os.path.getsize(p) for p in chosen.values())
        plan[name] = (chosen, total)
        parts = total // limit + 1 if total else 0
        print(f"{name}: {len(chosen)} files, {total / GIB:.2f} GiB, about {parts} assets" + (f", no file for {missing}" if missing else ""), flush=True)
    if not A.dry_run:
        os.makedirs(A.out, exist_ok=True)
    total_all = sum(t for _, t in plan.values())
    probe = os.path.abspath(A.out)
    while not os.path.isdir(probe):
        probe = os.path.dirname(probe)
    free = shutil.disk_usage(probe).free
    print(f"total {total_all / GIB:.2f} GiB, free space at the output folder {free / GIB:.2f} GiB", flush=True)
    if not A.dry_run and total_all * 1.01 > free:
        print("Not enough free space at the output folder", file=sys.stderr)
        return 2
    manifest_path = os.path.join(A.out, f"MANIFEST-{A.label}.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else dict(limit_bytes=limit, assets={})
    for name in groups:
        chosen, _ = plan[name]
        if chosen and not A.dry_run:
            print(f"packing {name}", flush=True)
            manifest["assets"][f"hqd-data-{name}"] = pack_data(f"hqd-data-{name}", chosen, A.out, limit)
    status = 0
    if "logs" in selected:
        private = read_private(A.private)
        rules = make_rules(data_roots + log_roots, private, A.drop_folder)
        logs = collect_logs(log_roots, exclude, A.drop_folder)
        name = f"hqd-logs-{A.label}"
        entry, counts, problems = pack_logs(name, logs, rules, residual_regex(private), A.out, limit, A.dry_run)
        print(f"logs: {len(logs)} files, {sum(m['bytes'] for m in entry['members']) / 2 ** 20:.1f} MiB after rewriting, replacements {counts}", flush=True)
        if problems:
            print(f"{len(problems)} logs still hold a private term: {[p['path'] for p in problems][:20]}", file=sys.stderr)
            status = 3
        if not A.dry_run:
            if status:
                for part in entry["parts"]:
                    os.remove(os.path.join(A.out, part["file"]))
            else:
                manifest["assets"][name] = entry
                json.dump(dict(replacements=counts, logs=len(logs)), open(os.path.join(A.out, f"scrub_report-{A.label}.json"), "w"), indent=1)
    if not A.dry_run:
        json.dump(manifest, open(manifest_path, "w"), indent=0)
        with open(os.path.join(A.out, f"SHA256SUMS-{A.label}.txt"), "w") as f:
            for asset in sorted(manifest["assets"]):
                for part in manifest["assets"][asset]["parts"]:
                    f.write(f"{part['sha256']}  {part['file']}\n")
        print(f"written to {A.out}, sha256sum -c SHA256SUMS-{A.label}.txt checks the assets", flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
