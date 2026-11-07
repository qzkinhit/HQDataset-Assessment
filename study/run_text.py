#!/usr/bin/env python3
"""Language-model track of the controlled study. A small pretrained model is fine-tuned on a five-domain pool under one controlled corpus defect or a
quality filter, and scored by per-token held-out negative log-likelihood per domain. Low-quality documents come from the pool itself (truncated,
template, cross-domain and low-tier web text). Document-level PPL under the untouched model is computed once, because the draft standard names PPL in its Annex A."""
import argparse, ast, json, os, time, fcntl, zlib
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
ap = argparse.ArgumentParser(); ap.add_argument("--job", required=True); ap.add_argument("--out", required=True); A = ap.parse_args(); J = json.loads(A.job)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); ROOT = os.environ.get("HQD_ROOT", os.path.join(REPO, "downloads")); POOL = os.environ.get("HQD_POOL", os.path.join(REPO, "downloads", "processed")); NAME = "HuggingFaceTB/SmolLM2-135M"
def rows(fn):
    out = []
    for l in open(f"{POOL}/{fn}"):
        r = json.loads(l); m = r["meta"] if isinstance(r["meta"], dict) else ast.literal_eval(r["meta"]); r["quality"] = m.get("quality"); r["noise"] = m.get("noise"); out.append(r)
    return out
tok = AutoTokenizer.from_pretrained(NAME); tok.pad_token = tok.pad_token or tok.eos_token
device = "cuda"; MAXLEN = 192; t0 = time.time()
def batches(texts, bs, order=None):
    order = np.arange(len(texts)) if order is None else order
    for i in range(0, len(order), bs):
        e = tok([texts[j] for j in order[i:i + bs]], return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN); yield {k: v.to(device) for k, v in e.items()}
@torch.no_grad()
def nll_per_doc(model, texts, bs=32):
    model.eval(); out = []
    for e in batches(texts, bs):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            lg = model(**e).logits[:, :-1]
        y = e["input_ids"][:, 1:]; m = e["attention_mask"][:, 1:].bool(); ll = torch.nn.functional.cross_entropy(lg.float().reshape(-1, lg.size(-1)), y.reshape(-1), reduction="none").view(y.shape)
        out.append(((ll * m).sum(1) / m.sum(1).clamp_min(1)).cpu().numpy())
    return np.concatenate(out)
train, held = rows("qpool_train.jsonl"), rows("qpool_heldout.jsonl")
if J.get("mode") == "ppl":
    model = AutoModelForCausalLM.from_pretrained(NAME, torch_dtype=torch.bfloat16).to(device)
    np.savez(f"{ROOT}/text_ppl.npz", ppl=nll_per_doc(model, [r["text"] for r in train]), ids=np.array([r["id"] for r in train])); print("ppl done", time.time() - t0); raise SystemExit
ppl = np.load(f"{ROOT}/text_ppl.npz")["ppl"]
quality = np.array([r["quality"] for r in train]); noise = np.array([r["noise"] for r in train]); dom = np.array([r["domain"] for r in train]); texts = np.array([r["text"] for r in train], dtype=object)
seed = J["seed"]; defects = J["defects"]; rng = np.random.default_rng(zlib.crc32(json.dumps([seed, defects], sort_keys=True).encode()) % (2 ** 32)); torch.manual_seed(seed)
hi = np.where(quality == "high")[0]; NBASE = J.get("n", 6000); base = rng.choice(hi, NBASE, replace=False); cur = list(base); heldout_text = np.array([r["text"] for r in held], dtype=object); held_dom = np.array([r["domain"] for r in held]); leak = None; cur_text = list(texts[base])
for d, r in defects.items():
    k = int(round(r * NBASE))
    if d in ("truncation", "template", "crossdomain", "lowtier"):
        src = rng.choice(np.where(noise == d)[0], k, replace=False); pos = rng.choice(len(cur_text), k, replace=False)
        for p, s in zip(pos, src): cur_text[p] = texts[s]; cur[p] = s
    elif d == "duplicate":
        pos = rng.choice(len(cur_text), k, replace=False); rest = np.setdiff1d(np.arange(len(cur_text)), pos); src = rng.choice(rest, k)
        for p, s in zip(pos, src): cur_text[p] = cur_text[s]; cur[p] = cur[s]
    elif d == "coverage":      # remove one domain first, the stand-in for a scene the corpus does not cover
        order = rng.permutation(np.unique(dom)); removable = [i for i, c in enumerate(cur) if dom[c] == order[0]]
        take = rng.permutation(removable)[:k]; keep = np.setdiff1d(np.arange(len(cur)), take); fill = rng.choice(keep, len(take)); cur = [cur[i] for i in keep] + [cur[i] for i in fill]; cur_text = [cur_text[i] for i in keep] + [cur_text[i] for i in fill]
    elif d == "scale":
        keep = rng.permutation(len(cur))[:len(cur) - k]; cur = [cur[i] for i in keep]; cur_text = [cur_text[i] for i in keep]
    elif d == "contamination":
        li = rng.choice(len(heldout_text), min(k, len(heldout_text)), replace=False); pos = rng.choice(len(cur_text), len(li), replace=False)
        for p, s in zip(pos, li): cur_text[p] = heldout_text[s]
        leak = li
if "filter" in J:              # the pool contains low-quality text at J["low"], a filter keeps NBASE documents out of it
    low = rng.choice(np.where(quality == "low")[0], int(J["low"] * NBASE), replace=False); pool = np.r_[rng.choice(hi, NBASE, replace=False), low]; f = J["filter"]
    if f == "none": sel = rng.permutation(pool)[:]            # no filter, larger corpus
    elif f == "random": sel = rng.choice(pool, NBASE, replace=False)
    elif f == "ppl": sel = pool[np.argsort(ppl[pool])[:NBASE]]            # lowest PPL, the Annex A reading
    elif f == "ppl_mid": o = np.argsort(ppl[pool]); drop = int((len(pool) - NBASE) / 2); sel = pool[o[drop:drop + NBASE]]
    elif f == "oracle": sel = pool[quality[pool] == "high"]
    cur = list(sel); cur_text = list(texts[sel])
model = AutoModelForCausalLM.from_pretrained(NAME, torch_dtype=torch.float32).to(device)
EPOCHS, BS, LR = J.get("epochs", 2), 16, 2e-5; opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01); total = EPOCHS * (len(cur_text) // BS); sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1.0, (t + 1) / 50) * max(0.0, 1 - t / max(1, total)))
gen = np.random.default_rng(seed); model.train(); cur_text = list(map(str, cur_text))
for ep in range(EPOCHS):
    for e in batches(cur_text, BS, gen.permutation(len(cur_text))):
        lab = e["input_ids"].masked_fill(e["attention_mask"] == 0, -100)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16): loss = model(**e, labels=lab).loss
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
nl = nll_per_doc(model, list(map(str, heldout_text)))
r = dict(J, group="text", task="lm", n_train=len(cur_text), n_test=len(nl), nll=float(nl.mean()), nll_by_domain={d: float(nl[held_dom == d].mean()) for d in np.unique(held_dom)}, corpus_ppl=float(np.mean(ppl[np.array(cur, dtype=int)])) if len(cur) else None, sec=round(time.time() - t0, 1))
if leak is not None:
    m = np.ones(len(nl), bool); m[leak] = False; r["nll_unseen"] = float(nl[m].mean()) if m.any() else None; r["nll_leaked"] = float(nl[~m].mean())
with open(A.out, "a") as f:
    fcntl.flock(f, fcntl.LOCK_EX); f.write(json.dumps(r) + "\n"); fcntl.flock(f, fcntl.LOCK_UN)
