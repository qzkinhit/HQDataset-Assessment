#!/usr/bin/env python3
"""The language model as quality judge of web documents, in the style of the judge filters of the survey. Each document of the C4 training pool of corpus/run_corpus.py gets a rating from 1 to 5 from one fixed prompt, read off the next-token probabilities as the expected rating, one forward pass per document, first 400 tokens.
The documents are in random order with a fixed seed, in units of 2048. The merge writes c4_llm_score.npy with NaN for documents without a score, and a summary with the agreement of the rating with the static signals."""
import glob, json, os, zlib
import numpy as np
from qwen import chat, ids, next_probs
ROOT = os.environ.get("HQD_CORPUS", "corpus"); UNIT = 2048; MAX_TOKENS = 400
SYSTEM = "You rate web documents as training data for a general language model."
HEAD = """Rate the quality of the document below on a scale from 1 to 5.
1: spam, boilerplate, gibberish or lists without readable content.
2: mostly low value, such as fragments, navigation text or heavy repetition.
3: readable, ordinary text with limited information.
4: clear, coherent and informative text.
5: excellent, well written, accurate and rich in knowledge.

Document:
"""
TAIL = "\n\nAnswer with one digit from 1 to 5.<|im_end|>\n<|im_start|>assistant\n"
_c = {}
def data():
    if not _c:
        z = np.load(f"{ROOT}/c4.npz"); TOK, OFF = z["tokens"], z["offsets"]; LEN = np.diff(OFF); H = np.array([zlib.crc32(str(i).encode()) % 100 for i in range(len(LEN))]); pool = np.where((H >= 2) & (H < 42) & (LEN >= 32))[0]
        _c.update(TOK=TOK, OFF=OFF, n=len(LEN), pool=pool, order=np.random.default_rng(0).permutation(pool))
    return _c
def units(n_docs): return [(k, min(n_docs, len(data()["order"]))) for k in range(0, min(n_docs, len(data()["order"])), UNIT)]
def source_tok():
    from tokenizers import Tokenizer
    fn = sorted(glob.glob(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")) + "/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots/*/tokenizer.json"))[0]; return Tokenizer.from_file(fn)
def run(model, tok, src, start, stop, size=UNIT):
    """Document numbers, expected ratings and the probability mass on the five digits for one unit."""
    d = data(); docs = d["order"][start:min(start + size, stop)]; text = src.decode_batch([d["TOK"][d["OFF"][i]:d["OFF"][i + 1]].tolist() for i in docs], skip_special_tokens=False); tail = ids(tok, TAIL); prefix = ids(tok, chat(SYSTEM) + "<|im_start|>user\n" + HEAD); opt = [ids(tok, str(r))[0] for r in range(1, 6)]
    sufs = [ids(tok, t, special=False, limit=MAX_TOKENS) + tail for t in text]; p = np.array(next_probs(model, tok, prefix, sufs, opt)); mass = p.sum(1); return docs, (p @ np.arange(1, 6)) / np.maximum(mass, 1e-12), mass, np.array([len(s) - len(tail) for s in sufs])
def merge(root, stop):
    """Writes the score file when every unit is finished, atomically, so that the training jobs read a whole file."""
    out = f"{ROOT}/c4_llm_score.npy"; files = [f"{root}/units/judge/{k}.npz" for k, _ in units(stop)]
    if os.path.exists(out) or not all(os.path.exists(f) for f in files): return False
    d = data(); s = np.full(d["n"], np.nan, np.float32); m = s.copy(); toks = 0
    for f in files: z = np.load(f); s[z["doc"]] = z["score"]; m[z["doc"]] = z["mass"]; toks += int(z["tokens"].sum())
    tmp = f"{out}.{os.getpid()}.npy"; np.save(tmp, s); os.replace(tmp, out); ok = np.isfinite(s); z = np.load(f"{ROOT}/c4.npz"); ppl = np.load(f"{ROOT}/c4_ppl.npy"); rank = lambda x: np.argsort(np.argsort(x))
    cor = {k: float(np.corrcoef(rank(s[ok]), rank(v[ok]))[0, 1]) for k, v in dict(ppl=ppl, alpha=z["alpha"], rep=z["rep"], stop=z["stop"], words=z["words"]).items()}
    json.dump(dict(scored=int(ok.sum()), pool=len(d["pool"]), tokens_read=toks, mean=float(s[ok].mean()), quantiles={str(q): float(np.quantile(s[ok], q)) for q in (.01, .1, .25, .5, .75, .9, .99)}, mass_mean=float(m[ok].mean()), mass_low=float(np.quantile(m[ok], .01)), spearman=cor), open(f"{root}/results_llm_judge.json", "w"), indent=0); return True
