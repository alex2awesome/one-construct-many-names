#!/usr/bin/env python3
"""PREREG-29 judge runner = prereg26_run.py with a polling queue across domains (one family per process).

usage: prereg29_run.py <BASE> <gpt56|claude> <workers>
Queue: BASE/queue/<prio>__<domain>__<kind>.json (job lists); polled every 5 s in filename order, so a file with a
lower prefix jumps the queue. W = BASE/<domain>; keys W_private/keys.json. Exits when BASE/queue/STOP exists and
nothing is pending. Claude token order: anon -> anon -> stanford.
kinds: formality | adj | check | score
gpt56 runs on sk3 (codex exec, usc CODEX_HOME); claude runs on sk2 (claude -p, token stanford -> anon).
Keys for gates: <W>_private/keys.json (outside the judges' cwd tree).
Each attempt in W/<fam>/<kind>/<job>_a<k>/ (prompt.txt, raw reply, meta); final W/<fam>/<kind>/<job>.json.
Gate kinds (formality, adj): max 3 attempts, fresh session each. score: one retry. check: one retry.
A usage-limit reply is not an attempt (claude: next token; gpt: stop family).
"""
import json, os, re, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

BASE, FAM, WORKERS = sys.argv[1], sys.argv[2], int(sys.argv[3])
_KEYS = {}


def keys_for(W):
    if W not in _KEYS:
        _KEYS[W] = json.load(open(f"{W}_private/keys.json"))
    return _KEYS[W]
LOCK = threading.RLock()
PARKED = set()
STOP = threading.Event()
if FAM == "gpt56":
    HOME = "/remote/anon"
    ENV = dict(os.environ, HOME=HOME, PATH=f"{HOME}/.npm-global/bin:" + os.environ.get("PATH", ""),
               CODEX_HOME=f"{HOME}/codex_homes/acct")
else:
    HOME = "/remote/anon"
    TOKENS = [("anon", f"{HOME}/.claude-token.txt"),
              ("anon", f"{HOME}/.claude-token.txt"),
              ("stanford", f"{HOME}/.claude-account.txt")]
CLAUDE_LIMIT = re.compile(r"hit your (session |weekly )?limit|out of (extra )?usage|resets \d", re.I)
CODEX_LIMIT = re.compile(r"usage limit|try again at", re.I)
MAXATT = {"formality": 3, "adj": 3, "check": 2, "score": 2}


def log(msg):
    with LOCK:
        with open(f"{BASE}/runner_{FAM}.log", "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


def runlog(row):
    with LOCK:
        with open(f"{BASE}/runlog_{FAM}.tsv", "a") as f:
            f.write("\t".join(str(x) for x in row) + "\n")


def call_gpt(prompt, d, effort):
    open(f"{d}/prompt.txt", "w").write(prompt)
    cmd = ["codex", "exec", "--model", "gpt-5.6-sol", "-c", 'web_search="disabled"']
    if effort:
        cmd += ["-c", f'model_reasoning_effort="{effort}"']
    cmd += ["--sandbox", "read-only", "--skip-git-repo-check", "--ephemeral", "-C", d, "-o", f"{d}/last.txt", "-"]
    t0 = time.time()
    with open(f"{d}/prompt.txt") as fin, open(f"{d}/codex.log", "w") as flog:
        try:
            rc = subprocess.run(cmd, stdin=fin, stdout=flog, stderr=subprocess.STDOUT, env=ENV, timeout=1200).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    txt = open(f"{d}/last.txt").read() if os.path.exists(f"{d}/last.txt") else ""
    lg = open(f"{d}/codex.log", errors="replace").read()
    m = re.search(r"model:\s*(\S+)", lg)
    served = m.group(1) if m else "?"
    meta = {"account": "usc(CODEX_HOME)", "model_req": "gpt-5.6-sol", "model_served": served,
            "extra": f"effort={effort or 'default'}", "rc": rc, "sec": round(time.time() - t0)}
    if not txt and CODEX_LIMIT.search(lg):
        return "__LIMIT__", meta
    return txt, meta


def call_claude(prompt, d):
    open(f"{d}/prompt.txt", "w").write(prompt)
    for model in ("claude-sonnet-5", "claude-sonnet-5-5"):
        for tok, tf in TOKENS:
            if tok in PARKED:
                continue
            token = "".join(open(tf).read().split())
            env = dict(os.environ, HOME=HOME, CLAUDE_CODE_OAUTH_TOKEN=token)
            out = f"{d}/out_{tok}_{model}.json"
            t0 = time.time()
            with open(f"{d}/prompt.txt") as fin, open(out, "w") as fout, open(f"{d}/err.txt", "w") as ferr:
                try:
                    rc = subprocess.run([f"{HOME}/.npm-global/bin/claude", "-p", "--model", model,
                                         "--output-format", "json"], stdin=fin, stdout=fout, stderr=ferr,
                                        env=env, cwd=d, timeout=1200).returncode
                except subprocess.TimeoutExpired:
                    rc = "timeout"
            raw = open(out, errors="replace").read()
            try:
                res = json.loads(raw)
            except Exception:
                try:  # CLI 2.1.289 sometimes leaves a stale tail after the JSON object: take the first object
                    res = json.JSONDecoder().raw_decode(raw.lstrip())[0]
                except Exception:
                    res = None
            if not isinstance(res, dict):
                meta = {"account": tok, "model_req": model, "model_served": "?", "extra": "unparsed_stdout",
                        "rc": rc, "sec": round(time.time() - t0)}
                return "", meta
            r = str(res.get("result", ""))
            meta = {"account": tok, "model_req": model, "model_served": ",".join(res.get("modelUsage", {}) or ["?"]),
                    "extra": f"is_error={res.get('is_error')}", "rc": rc, "sec": round(time.time() - t0)}
            if res.get("is_error") and CLAUDE_LIMIT.search(r):
                with LOCK:
                    PARKED.add(tok)
                log(f"claude token {tok} parked: {r[:100]}")
                continue
            if res.get("is_error") and re.search(r"model|not found|invalid|400", r, re.I):
                log(f"claude model {model} refused: {r[:120]}")
                break
            if res.get("is_error"):
                return "", meta
            return r, meta
    return "__LIMIT__", {"account": "-", "model_req": "-", "model_served": "-", "extra": "all parked", "rc": "-", "sec": 0}


def parse_obj(text):
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j < 0:
        return None
    try:
        return json.loads(t[i:j + 1])
    except Exception:
        return None


def parse_lines(text):
    rows = []
    for line in text.splitlines():
        line = line.strip().strip("`").strip().rstrip(",")
        if not line.startswith("{"):
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def check(kind, job, txt):
    """-> (ok, msg, parsed)"""
    KEYS = keys_for(job["_W"])
    if kind == "adj":
        kb = KEYS["adj"][job["job"]] if job["job"] in KEYS.get("adj", {}) else KEYS["adj_v2"][job["job"]]
        obj = parse_obj(txt)
        if not isinstance(obj, dict) or not isinstance(obj.get("pairs"), dict) or not isinstance(obj.get("forms"), dict):
            return False, "unparsed", None
        P = {k: str(v).strip().lower() for k, v in obj["pairs"].items()}
        F = {k: str(v).strip().lower() for k, v in obj["forms"].items()}
        miss = [k for k in kb["pairs"] if P.get(k) not in {"same", "overlap", "different"}] + \
               [k for k in kb["forms"] if F.get(k) not in {"better", "worse", "unclear"}]
        if miss:
            return False, f"missing/invalid {len(miss)}", None
        n = ok = 0
        for k, e in kb["pairs"].items():
            if e["kind"] != "real":
                n += 1; ok += P[k] == e["answer"]
        for k, e in kb["forms"].items():
            if e["kind"] != "real":
                n += 1; ok += F[k] == e["answer"]
        return ok / n >= 0.8, f"anchors {ok}/{n}", {"pairs": P, "forms": F, "anchor_acc": ok / n}
    if kind == "formality":
        kb = KEYS["formality"][job["job"]]
        rows = {str(r.get("id")): r for r in parse_lines(txt) if isinstance(r, dict)}
        bad = [i for i in kb if i not in rows or rows[i].get("stratum") not in {"germanic", "latinate", "greek", "mixed"}
               or not isinstance(rows[i].get("formality"), (int, float)) or not 1 <= rows[i]["formality"] <= 7]
        if bad:
            return False, f"missing/invalid {len(bad)}", None
        anc = [i for i, e in kb.items() if e["kind"] == "anchor"]
        ok = sum(rows[i]["stratum"] == kb[i]["stratum"] for i in anc)
        return ok >= 8, f"anchors {ok}/{len(anc)}", {"rows": rows, "anchor_ok": ok}
    if kind == "check":
        obj = parse_obj(txt)
        ok = isinstance(obj, dict) and all(isinstance(obj.get(s), dict) and obj[s].get("verdict") for s in ("S1", "S2"))
        return ok, "parsed" if ok else "unparsed", obj
    if kind == "score":
        want = set(job["idxs"])
        got = {}
        for r in parse_lines(txt):
            try:
                i, s = int(r["item_idx"]), r["score"]
            except Exception:
                continue
            if i in want and isinstance(s, (int, float)) and float(s).is_integer() and 1 <= s <= 7 and i not in got:
                got[i] = int(s)
        ok = set(got) == want
        return ok, f"{len(got)}/{len(want)} valid", [{"item_idx": i, "score": got[i]} for i in job["idxs"] if i in got]


def run(kind, job):
    if STOP.is_set():
        return
    jid = job["job"]
    W = job["_W"]
    od = f"{W}/{FAM}/{kind}"
    os.makedirs(od, exist_ok=True)
    final = f"{od}/{jid}.json"
    if os.path.exists(final):
        return
    att = 0
    base = 0
    while os.path.exists(f"{od}/{jid}_a{base + 1}"):
        base += 1  # earlier attempts (e.g. superseded) are kept; numbering continues
    while att < MAXATT[kind]:
        att += 1
        d = f"{od}/{jid}_a{base + att}"
        os.makedirs(d, exist_ok=True)
        if FAM == "gpt56":
            txt, meta = call_gpt(job["prompt"], d, "high" if kind == "score" else None)
        else:
            txt, meta = call_claude(job["prompt"], d)
        if txt == "__LIMIT__":
            runlog([W.rsplit("/", 1)[-1], kind, jid, att, FAM, meta["account"], meta["model_req"], meta["model_served"], meta["extra"],
                    meta["rc"], meta["sec"], "LIMIT"])
            log(f"{kind} {jid} LIMIT -> stopping family")
            STOP.set()
            return
        open(f"{d}/reply.txt", "w").write(txt)
        ok, msg, parsed = check(kind, job, txt)
        runlog([W.rsplit("/", 1)[-1], kind, jid, att, FAM, meta["account"], meta["model_req"], meta["model_served"], meta["extra"],
                meta["rc"], meta["sec"], ("PASS " if ok else "FAIL ") + msg])
        log(f"{W.rsplit('/', 1)[-1]} {kind} {jid} a{att} {'PASS' if ok else 'FAIL'} {msg} {meta['sec']}s {meta['account']} {meta['model_served']}")
        if ok:
            json.dump({"job": jid, "status": "pass", "attempt": base + att, "meta": meta, "parsed": parsed},
                      open(final, "w"), indent=1)
            return
    json.dump({"job": jid, "status": "failed"}, open(final, "w"))


def main():
    import glob
    qd = f"{BASE}/queue"
    os.makedirs(qd, exist_ok=True)
    log(f"start workers={WORKERS}")
    inflight = {}
    ex = ThreadPoolExecutor(WORKERS)
    while True:
        for k in [k for k, f in inflight.items() if f.done()]:
            try:
                inflight.pop(k).result()
            except Exception as e:
                log(f"EXC {k} {e!r}")
        pending = []
        for qf in sorted(glob.glob(f"{qd}/*.json")):
            _, dom, kind = os.path.basename(qf)[:-5].split("__")
            W = f"{BASE}/{dom}"
            try:
                jl = json.load(open(qf))
            except Exception:
                continue
            for j in jl:
                key = (dom, kind, j["job"])
                if key in inflight or os.path.exists(f"{W}/{FAM}/{kind}/{j['job']}.json"):
                    continue
                j["_W"] = W
                pending.append((key, kind, j))
        if STOP.is_set():
            log("family stopped (limit)")
            break
        if not pending and not inflight and os.path.exists(f"{qd}/STOP"):
            break
        for key, kind, j in pending:
            if len(inflight) >= WORKERS:
                break
            inflight[key] = ex.submit(run, kind, j)
        time.sleep(5)
    ex.shutdown(wait=True)
    log("ALL DONE")


if __name__ == "__main__":
    main()
