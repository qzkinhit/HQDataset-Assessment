#!/usr/bin/env python3
"""Web corpus track. A small language model is trained from scratch on real web text (C4) kept by a quality filter, with the token budget fixed, and scored by per-token held-out loss
on three references: unseen C4 documents, C4 documents with no near duplicate in the pool, and Wikipedia. The filters are the ones named in standards and papers: lowest perplexity (the reading of the draft standard's Annex A),
middle perplexity, heuristic rules, near-duplicate removal, and random selection. Nothing is injected. Mode "ppl" scores every document once with a pretrained model."""
import argparse, json, os, time, fcntl, zlib
import numpy as np, torch
from transformers import AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel
ap = argparse.ArgumentParser(); ap.add_argument("--job", required=True); ap.add_argument("--out", required=True); A = ap.parse_args(); J = json.loads(A.job)
ROOT = os.environ.get("HQD_CORPUS", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "downloads", "corpus")); device = "cuda"; t0 = time.time()
z = np.load(f"{ROOT}/c4.npz"); TOK, OFF = z["tokens"], z["offsets"]; n_docs = len(OFF) - 1; EOS = 0
LEN = np.diff(OFF); group = np.load(f"{ROOT}/c4_group.npy")
# held-out split by document hash, fixed for all arms, the unique reference has no document of its near-duplicate group in the pool
H_ = np.array([zlib.crc32(str(i).encode()) % 100 for i in range(n_docs)]); hold = H_ < 2; sub = (H_ >= 2) & (H_ < 42)   # 2% held out, 40% of the documents form the training pool
gs, gc = np.unique(group, return_counts=True); gsize = dict(zip(gs.tolist(), gc.tolist())); uniq = np.array([gsize[g] == 1 for g in group.tolist()])
def doc(i, tokens=TOK, off=OFF): return tokens[off[i]:off[i + 1]]
if J.get("mode") == "ppl":
    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", torch_dtype=torch.bfloat16).to(device).eval(); idx_ = np.where(sub)[0]; order = idx_[np.argsort(LEN[idx_])]; out = np.full(n_docs, np.nan, np.float32); BS = 128; L = 128
    for b in range(0, len(order), BS):
        ids = order[b:b + BS]; seqs = [doc(i)[:L] for i in ids]; m = max(max(len(s) for s in seqs), 2); x = np.zeros((len(ids), m), np.int64); a = np.zeros((len(ids), m), np.int64)
        for k, s in enumerate(seqs): x[k, :len(s)] = s; a[k, :len(s)] = 1
        x = torch.from_numpy(x).to(device); a = torch.from_numpy(a).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16): lg = model(input_ids=x, attention_mask=a).logits[:, :-1]
        y = x[:, 1:]; mk = a[:, 1:].bool(); ll = torch.nn.functional.cross_entropy(lg.float().reshape(-1, lg.size(-1)), y.reshape(-1), reduction="none").view(y.shape); out[ids] = ((ll * mk).sum(1) / mk.sum(1).clamp_min(1)).cpu().numpy()
        if b % (BS * 500) == 0: print(b, len(order), round(time.time() - t0), flush=True)
    np.save(f"{ROOT}/c4_ppl.npy", out); print("ppl done", time.time() - t0); raise SystemExit
ppl = np.load(f"{ROOT}/c4_ppl.npy"); alpha, rep, stop = z["alpha"], z["rep"], z["stop"]; words = z["words"]
seed, arm, budget, size = J["seed"], J["arm"], J.get("budget"), J["size"]; rng = np.random.default_rng(zlib.crc32(json.dumps([seed, arm, budget, size]).encode()) % (2 ** 32)); torch.manual_seed(seed)
pool = np.where(sub & (LEN >= 32))[0]
def take(ids):
    """ids in priority order, keep documents until the token budget is met"""
    if budget is None: return ids
    c = np.cumsum(LEN[ids]); k = int(np.searchsorted(c, budget)) + 1; return ids[:k]
lo, hi = np.quantile(ppl[pool], [.1, .9])
rule = (alpha >= .6) & (rep <= .3) & (stop >= .1) & (words >= 50)
first = np.zeros(n_docs, bool); first[np.unique(group, return_index=True)[1]] = True
if arm.startswith("dclm_"):       # a published corpus, tokenized in the same way: the unfiltered pool or the filtered baseline of DCLM
    y2 = np.load(f"{ROOT}/{arm}.npz"); T2, O2 = y2["tokens"], y2["offsets"]; ids2 = rng.permutation(len(O2) - 1); c2 = np.cumsum(np.diff(O2)[ids2]); k2 = int(np.searchsorted(c2, budget)) + 1; sel2 = ids2[:k2]
elif arm == "all": sel = rng.permutation(pool)
elif arm == "random": sel = take(rng.permutation(pool))
elif arm == "ppl_low": sel = take(pool[np.argsort(ppl[pool])])
elif arm == "ppl_high": sel = take(pool[np.argsort(-ppl[pool])])
elif arm == "ppl_mid": c = pool[(ppl[pool] >= lo) & (ppl[pool] <= hi)]; sel = take(rng.permutation(c))
elif arm == "rule": c = pool[rule[pool]]; sel = take(rng.permutation(c))
elif arm == "dedup": c = pool[first[pool]]; sel = take(rng.permutation(c))
elif arm == "rule_dedup": c = pool[rule[pool] & first[pool]]; sel = take(rng.permutation(c))
elif arm in ("llm_high", "llm_mid"):     # rating of the language model judge, llm/judge_corpus.py, documents without a rating are left out
    llm = np.load(f"{ROOT}/c4_llm_score.npy"); c = pool[np.isfinite(llm[pool])]
    if arm == "llm_high": c = rng.permutation(c); sel = take(c[np.argsort(-llm[c], kind="stable")])     # equal ratings in random order
    else: q1, q9 = np.quantile(llm[c], [.1, .9]); sel = take(rng.permutation(c[(llm[c] >= q1) & (llm[c] <= q9)]))
else: raise ValueError(arm)
stream = np.concatenate([np.r_[T2[O2[i]:O2[i + 1]], EOS] for i in sel2]).astype(np.int64) if arm.startswith("dclm_") else np.concatenate([np.r_[doc(i), EOS] for i in sel]).astype(np.int64); data = torch.from_numpy(stream).to(device)
CFG = dict(XS=(2, 4, 64), T=(4, 4, 128), S=(4, 4, 192), SM=(6, 8, 256), MS=(6, 8, 384), M=(8, 8, 512), ML=(10, 10, 640), L=(12, 12, 768))[size]; net = GPT2LMHeadModel(GPT2Config(vocab_size=int(max(TOK.max(), stream.max())) + 1, n_positions=256, n_layer=CFG[0], n_head=CFG[1], n_embd=CFG[2], resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)).to(device); net.gradient_checkpointing_disable()
BS, W, EPOCHS = 48, 256, J.get("epochs", 1); steps = max(100, EPOCHS * len(stream) // (BS * W)); opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=.01, betas=(.9, .99)); sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1.0, (t + 1) / 500) * (0.5 * (1 + np.cos(np.pi * min(1.0, t / steps)))))
net.train()
for s in range(steps):
    ix = torch.randint(0, len(stream) - W - 1, (BS,), device=device); x = torch.stack([data[i:i + W] for i in ix])
    with torch.amp.autocast("cuda", dtype=torch.bfloat16): loss = net(input_ids=x, labels=x).loss
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step()
net.eval()
def score(tokens, off, ids, bs=64):
    tot, cnt = 0.0, 0
    for b in range(0, len(ids), bs):
        seqs = [tokens[off[i]:off[i + 1]][:W] for i in ids[b:b + bs]]; seqs = [s for s in seqs if len(s) >= 2]
        if not seqs: continue
        m = max(len(s) for s in seqs); x = np.zeros((len(seqs), m), np.int64); a = np.zeros((len(seqs), m), np.int64)
        for k, s in enumerate(seqs): x[k, :len(s)] = s; a[k, :len(s)] = 1
        x = torch.from_numpy(x).to(device); a = torch.from_numpy(a).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16): lg = net(input_ids=x, attention_mask=a).logits[:, :-1]
        y = x[:, 1:]; mk = a[:, 1:].bool(); ll = torch.nn.functional.cross_entropy(lg.float().reshape(-1, lg.size(-1)), y.reshape(-1), reduction="none").view(y.shape); tot += float((ll * mk).sum()); cnt += int(mk.sum())
    return tot / max(1, cnt)
er = np.random.default_rng(99); H = np.where(hold & (LEN >= 32))[0]; Hu = H[uniq[H]]; w = np.load(f"{ROOT}/wiki.npz"); nw = len(w["offsets"]) - 1
stats = dict(n_docs=int(len(sel2)), corpus_ppl=None, corpus_rep=None, corpus_stop=None, corpus_group_share=None) if arm.startswith("dclm_") else dict(n_docs=int(len(sel)), corpus_ppl=float(np.average(ppl[sel], weights=LEN[sel])), corpus_rep=float(np.average(rep[sel], weights=LEN[sel])), corpus_stop=float(np.average(stop[sel], weights=LEN[sel])), corpus_group_share=float((~uniq[sel]).mean()))     # these fields are read off the C4 documents, they do not exist for a DCLM document
r = dict(J, group="corpus", n_tokens=int(len(stream)), steps=int(steps), params=int(sum(p.numel() for p in net.parameters())), **stats, nll_c4=score(TOK, OFF, er.choice(H, min(4000, len(H)), replace=False)), nll_c4_unique=score(TOK, OFF, er.choice(Hu, min(4000, len(Hu)), replace=False)), nll_wiki=score(w["tokens"], w["offsets"], er.choice(nw, min(4000, nw), replace=False)), sec=round(time.time() - t0, 1))
with open(A.out, "a") as f:
    fcntl.flock(f, fcntl.LOCK_EX); f.write(json.dumps(r) + "\n"); fcntl.flock(f, fcntl.LOCK_UN)
