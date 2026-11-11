#!/usr/bin/env python
"""PREREG-25 re-adjudication v2 driver (runs on sk2): the lane-A harness (prereg25_adjudication_run.py)
unchanged except: author-calibrated batches (prereg25_readjudication_v2_build.py), PREREG-11 items
(kind "p11") are real items, Claude <= 3 concurrent, Claude tokens stanford -> anon only.

Original docstring follows.
PREREG-25 Step 1 driver (runs on sk2): two judge families over the adjudication batches.

GPT-5.6: codex exec --model gpt-5.6-sol, web search disabled, read-only sandbox, fresh
ephemeral session per call, all content inline (<=3 concurrent).
Claude: Claude Code CLI -p --model claude-sonnet-5 (fallback claude-sonnet-5-5), token
preference order stanford -> anon -> anon (a token is parked on a
limit reply), stdout to a file (<=4 concurrent).

A batch passes when the reply parses, covers every id with a valid code, and >= 8/10
anchors are right. A failing batch is re-run in a fresh session (max 3 attempts); every
attempt is kept on disk. Usage:  python prereg25_adjudication_run.py <workdir> [gpt|claude|both]
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HOME = "/remote/anon"
W = sys.argv[1]
WHICH = sys.argv[2] if len(sys.argv) > 2 else "both"
BATCHES = json.load(open(f"{W}/batches.json"))
KEY = json.load(open(f"{W}_private/key_private.json"))["key"]  # kept outside the judges' cwd tree
PAIR_CODES = {"same", "overlap", "different"}
FORM_CODES = {"better", "worse", "unclear"}
MAX_ATTEMPTS = 3
GATE = 0.8
LOCK = threading.Lock()
LOG = open(f"{W}/run.log", "a")
TOKENS = ["stanford", "anon"]
PARKED: set[str] = set()


def log(msg: str) -> None:
    with LOCK:
        LOG.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
        LOG.flush()


def parse(text: str):
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j < 0:
        return None
    try:
        return json.loads(t[i:j + 1])
    except Exception:
        return None


def check(bid: str, obj) -> tuple[bool, str, float]:
    kb = KEY[bid]
    if not isinstance(obj, dict) or not isinstance(obj.get("pairs"), dict) or not isinstance(obj.get("forms"), dict):
        return False, "unparsed", 0.0
    P = {k: str(v).strip().lower() for k, v in obj["pairs"].items()}
    F = {k: str(v).strip().lower() for k, v in obj["forms"].items()}
    miss = [k for k in kb["pairs"] if P.get(k) not in PAIR_CODES] + \
           [k for k in kb["forms"] if F.get(k) not in FORM_CODES]
    if miss:
        return False, f"missing/invalid {len(miss)}: {miss[:5]}", 0.0
    n = ok = 0
    for k, e in kb["pairs"].items():
        if e["kind"].startswith("anchor"):
            n += 1
            ok += P[k] == e["answer"]
    for k, e in kb["forms"].items():
        if e["kind"].startswith("anchor"):
            n += 1
            ok += F[k] == e["answer"]
    acc = ok / n
    return acc >= GATE, f"anchors {ok}/{n}", acc


def call_gpt(prompt: str, d: str) -> str:
    open(f"{d}/prompt.txt", "w").write(prompt)
    env = dict(os.environ, HOME=HOME, PATH=f"{HOME}/.npm-global/bin:" + os.environ.get("PATH", ""))
    with open(f"{d}/prompt.txt") as fin, open(f"{d}/codex.log", "w") as flog:
        subprocess.run(["codex", "exec", "--model", "gpt-5.6-sol", "-c", 'web_search="disabled"',
                        "--sandbox", "read-only", "--skip-git-repo-check", "--ephemeral", "-C", d,
                        "-o", f"{d}/last.txt", "-"], stdin=fin, stdout=flog, stderr=subprocess.STDOUT,
                       env=env, timeout=1500)
    txt = open(f"{d}/last.txt").read() if os.path.exists(f"{d}/last.txt") else ""
    if not txt and "hit your usage limit" in open(f"{d}/codex.log").read():
        return "__LIMIT__"
    return txt


def call_claude(prompt: str, d: str) -> str:
    open(f"{d}/prompt.txt", "w").write(prompt)
    for model in ("claude-sonnet-5", "claude-sonnet-5-5"):
        for tok in TOKENS:
            if tok in PARKED:
                continue
            token = open(f"{HOME}/.claude-{tok}-token.txt").read()
            token = "".join(token.split())
            env = dict(os.environ, HOME=HOME, CLAUDE_CODE_OAUTH_TOKEN=token)
            out = f"{d}/out_{tok}_{model}.json"
            with open(f"{d}/prompt.txt") as fin, open(out, "w") as fout, open(f"{d}/err.txt", "w") as ferr:
                subprocess.run([f"{HOME}/.npm-global/bin/claude", "-p", "--model", model,
                                "--output-format", "json"], stdin=fin, stdout=fout, stderr=ferr,
                               env=env, cwd=d, timeout=1500)
            try:
                res = json.load(open(out))
            except Exception:
                res = {"result": open(out).read()}
            r = str(res.get("result", ""))
            if res.get("is_error") and re.search(r"hit your (session )?limit|session limit", r):
                with LOCK:
                    PARKED.add(tok)
                log(f"claude token {tok} parked: {r[:80]}")
                continue
            if res.get("is_error") and re.search(r"model|not found|invalid|400", r, re.I):
                log(f"claude model {model} refused: {r[:120]}")
                break
            if res.get("is_error"):
                log(f"claude error {tok} {model}: {r[:160]}")
                return ""
            with open(f"{d}/meta.json", "w") as fm:
                json.dump({"token": tok, "model": model, "modelUsage": list(res.get("modelUsage", {}))}, fm)
            return r
    return "__LIMIT__"


def run(fam: str, b: dict) -> dict:
    bid = b["batch"]
    final = f"{W}/{fam}/{bid}.json"
    if os.path.exists(final):
        return json.load(open(final))
    for att in range(1, MAX_ATTEMPTS + 1):
        d = f"{W}/{fam}/{bid}_a{att}"
        os.makedirs(d, exist_ok=True)
        t0 = time.time()
        try:
            txt = (call_gpt if fam == "gpt" else call_claude)(b["prompt"], d)
        except subprocess.TimeoutExpired:
            txt = ""
        if txt == "__LIMIT__":
            log(f"{fam} {bid} a{att} LIMIT; stopping this family")
            return {"batch": bid, "status": "limit"}
        obj = parse(txt)
        ok, msg, acc = check(bid, obj)
        log(f"{fam} {bid} a{att} {'PASS' if ok else 'FAIL'} {msg} {time.time() - t0:.0f}s")
        if ok:
            rec = {"batch": bid, "status": "pass", "attempt": att, "anchor_acc": acc, "reply": obj}
            json.dump(rec, open(final, "w"), indent=1)
            return rec
        with open(f"{d}/fail.json", "w") as f:
            json.dump({"msg": msg, "acc": acc, "reply": obj}, f)
    rec = {"batch": bid, "status": "failed_gate"}
    json.dump(rec, open(final, "w"), indent=1)
    return rec


def main() -> None:
    fams = ["gpt", "claude"] if WHICH == "both" else [WHICH]
    for f in fams:
        os.makedirs(f"{W}/{f}", exist_ok=True)
    pools = {"gpt": ThreadPoolExecutor(3), "claude": ThreadPoolExecutor(3)}
    futs = [pools[f].submit(run, f, b) for f in fams for b in BATCHES]
    for fu in futs:
        fu.result()
    log("ALL DONE")


if __name__ == "__main__":
    main()
