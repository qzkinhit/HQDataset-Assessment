#!/usr/bin/env python3
"""Calls to the local Qwen 2.5 7B Instruct model at temperature zero, shared by the three language model baselines.
A prompt is a shared prefix plus a part per item. The prefix is computed once and its cache is copied per batch, greedy decoding returns the new tokens, one forward pass returns the next-token probabilities."""
import copy, os, subprocess, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
PATH = os.environ.get("QWEN_PATH", "Qwen/Qwen2.5-7B-Instruct")
def free_mib():
    """Free memory of the card of this process."""
    q = subprocess.run(["nvidia-smi", "-i", os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0], "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    return int(q) if q.isdigit() else 0
def fresh_start(within=120):
    """Whether a process of this user started on the card of this process in the last two minutes. Such a process may not have allocated its memory yet, so the free memory overstates what is left."""
    card = "CUDA_VISIBLE_DEVICES=" + os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    for line in subprocess.run(["ps", "-eo", "pid=,etimes="], capture_output=True, text=True).stdout.splitlines():
        pid, sec = map(int, line.split())
        if sec < within and pid != os.getpid():
            try:
                if card in open(f"/proc/{pid}/environ", "rb").read().decode("latin1").split("\0"): return True
            except OSError: pass
    return False
def load(need_mib=17000, margin_mib=3000, poll=3):
    """Model in bfloat16 on the card of this process. The process waits until the card has need + margin free, the same condition as gpu_pool3.py, polled every few seconds because the pools on the card poll every 20, and until no job has just started on the card. The memory is then held at once, so that other pools see it while the weights are read from disk."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        while True:
            while free_mib() < need_mib + margin_mib or fresh_start(): time.sleep(poll)
            try: h = torch.empty((need_mib - 500) * 2 ** 20, dtype=torch.uint8, device=dev); del h; break
            except torch.OutOfMemoryError: time.sleep(poll)
    tok = AutoTokenizer.from_pretrained(PATH); model = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.bfloat16, device_map={"": dev}).eval()
    return model, tok
def ids(tok, text, special=True, limit=None):
    """Token ids of one text, with special=False a marker such as <|im_end|> inside a document stays plain text."""
    x = tok(text, add_special_tokens=False, split_special_tokens=not special).input_ids
    return x[:limit] if limit else x
def chat(system, turns=()):
    return f"<|im_start|>system\n{system}<|im_end|>\n" + "".join(f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n" for u, a in turns)
def turn(user): return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
def _batches(lens, plen, budget, extra):
    order = sorted(range(len(lens)), key=lens.__getitem__); i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and (j - i + 1) * (plen + lens[order[j]] + extra) <= budget: j += 1
        yield [order[k] for k in range(i, j)]; i = j
def _split(fn, idx, tries=0):
    """A batch that does not fit into the shared card is split, a single item waits for memory."""
    try: return fn(idx)
    except torch.OutOfMemoryError:
        time.sleep(20)
        if len(idx) > 1: h = len(idx) // 2; return _split(fn, idx[:h], tries) + _split(fn, idx[h:], tries)
        if tries > 20: raise
        time.sleep(120); return _split(fn, idx, tries + 1)
def _first(model, c0, plen, sufs, pad):
    B, S = len(sufs), max(map(len, sufs)); x = torch.full((B, S), pad, dtype=torch.long); m = torch.zeros((B, plen + S), dtype=torch.long); m[:, :plen] = 1
    for k, s in enumerate(sufs): x[k, S - len(s):] = torch.tensor(s, dtype=torch.long); m[k, plen + S - len(s):] = 1
    x, m = x.to(model.device), m.to(model.device); pos = (m.cumsum(1) - 1)[:, plen:]; c = copy.deepcopy(c0); c.batch_repeat_interleave(B)
    return model(input_ids=x, attention_mask=m, position_ids=pos, past_key_values=c, use_cache=True, logits_to_keep=1).logits[:, -1], c, m, pos[:, -1]
def _prefix(model, prefix): return _split(lambda idx: [model(input_ids=torch.tensor([prefix], dtype=torch.long, device=model.device), use_cache=True, logits_to_keep=1).past_key_values], [0])[0]
@torch.inference_mode()
def generate(model, tok, prefix, sufs, max_new, budget=24000):
    """New token ids per item, greedy. An item leaves the batch at its end token, so a long reply does not hold the short ones."""
    pad = tok.pad_token_id; eos = torch.tensor([tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")], device=model.device); c0 = _prefix(model, prefix); res = [None] * len(sufs)
    def one(idx):
        lg, c, m, last = _first(model, c0, len(prefix), [sufs[i] for i in idx], pad); out = [[] for _ in idx]; live = torch.arange(len(idx), device=model.device)
        for _ in range(max_new):
            t = lg.argmax(-1); fin = torch.isin(t, eos)
            for k, v, f in zip(live.tolist(), t.tolist(), fin.tolist()):
                if not f: out[k].append(v)
            if fin.all(): break
            if fin.any(): keep = (~fin).nonzero().squeeze(1); c.batch_select_indices(keep); m, live, t, last = m[keep], live[keep], t[keep], last[keep]
            m = torch.cat([m, m.new_ones((len(live), 1))], 1); last = last + 1
            lg = model(input_ids=t[:, None], attention_mask=m, position_ids=last[:, None], past_key_values=c, use_cache=True, logits_to_keep=1).logits[:, -1]
        return out
    for b in _batches([len(s) for s in sufs], len(prefix), budget, 64):
        for i, o in zip(b, _split(one, b)): res[i] = o
    return res
@torch.inference_mode()
def next_probs(model, tok, prefix, sufs, options, budget=14000):
    """Next-token probabilities of the option tokens per item, from one forward pass, in float32."""
    pad = tok.pad_token_id; c0 = _prefix(model, prefix); res = [None] * len(sufs); opt = torch.tensor(options, device=model.device)
    def one(idx): return list(torch.softmax(_first(model, c0, len(prefix), [sufs[i] for i in idx], pad)[0].float(), -1)[:, opt].cpu().numpy())
    for b in _batches([len(s) for s in sufs], len(prefix), budget, 0):
        for i, o in zip(b, _split(one, b)): res[i] = o
    return res
def claim(path):
    """Atomic claim of a work unit whose result goes to path, a claim of a process that has died can be taken over. Returns False when the unit is finished or held."""
    if os.path.exists(path): return False
    os.makedirs(os.path.dirname(path), exist_ok=True); lock = path + ".claim"
    for _ in range(2):
        try: fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, str(os.getpid()).encode()); os.close(fd); return True
        except FileExistsError:
            try:
                pid = int(open(lock).read() or 0)
                if pid: os.kill(pid, 0); return False
            except (OSError, ValueError): pass
            try: os.remove(lock)
            except OSError: pass
    return False
def release(path):
    try: os.remove(path + ".claim")
    except OSError: pass
