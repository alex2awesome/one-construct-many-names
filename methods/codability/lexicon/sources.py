"""Resolve canonical-form keys back to ORIGINAL source text + build extraction contexts.

Key format (canon_all_real_forms.jsonl): ``task::layer::doc::item_idx`` where layer is ``raw``
(scraped HTML) or ``claude-parsed`` (markdown with source_url frontmatter). The gpt-parsed item
(name/description) is used only as a POINTER to locate the passage — the text shown to the
extractor is always the raw/claude-parsed document itself.

Source identity (independence unit for the census): normalized source URL — claude-parsed
frontmatter ``source_url``, raw via the task's urls-visited.csv filename→url map, else the doc
slug. Raw and claude-parsed copies of the same page collapse to one source when the URL resolves.
"""
from __future__ import annotations

import csv
import glob
import html as _html
import json
import os
import re
from functools import lru_cache
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CANON_PATH = os.path.join(ROOT, "outputs", "analyses", "canon_all_real_forms.jsonl")

_WS = re.compile(r"\s+")
_TAG = re.compile(r"<[^>]+>")
_SCRIPT = re.compile(r"<(script|style|noscript)\b[^>]*>.*?</\1>", re.S | re.I)
_WORD = re.compile(r"[a-z][a-z'-]+")
STOP = set("the a an and or of to in for with on is are be as by that this it its from not "
           "your you we they he she at if but so do does can may will would should".split())


def norm_text(s: str) -> str:
    """Whitespace-collapsed casefold — the space quotes/terms are validated in."""
    return _WS.sub(" ", (s or "").casefold()).strip()


def normalize_url(u: str) -> str:
    u = (u or "").strip().casefold()
    u = re.sub(r"^https?://(www\.)?", "", u)
    return u.split("?")[0].split("#")[0].rstrip("/")


def html_to_text(raw: str) -> str:
    try:
        from bs4 import BeautifulSoup  # type: ignore
        soup = BeautifulSoup(raw, "html.parser")
        for t in soup(["script", "style", "noscript"]):
            t.decompose()
        txt = soup.get_text(" ")
    except Exception:
        txt = _TAG.sub(" ", _SCRIPT.sub(" ", raw))
        txt = _html.unescape(txt)
    return _WS.sub(" ", txt).strip()


def rubrics_dir(task: str) -> str:
    p = os.path.join(ROOT, "datasets", task, "online-rubrics")
    if task == "math-stackexchange" and not os.path.isdir(p):
        # dataset moved 2026-07: datasets/math-stackexchange -> datasets/math/stackexchange
        return os.path.join(ROOT, "datasets", "math", "stackexchange", "online-rubrics")
    return p


@lru_cache(maxsize=None)
def url_map(task: str) -> Dict[str, str]:
    """raw filename -> normalized url, from urls-visited.csv (best effort)."""
    out: Dict[str, str] = {}
    p = os.path.join(rubrics_dir(task), "urls-visited.csv")
    if os.path.exists(p):
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                fn, u = (row.get("filename") or "").strip(), row.get("url") or ""
                if fn and u:
                    out[fn] = normalize_url(u)
    return out


_FRONT = re.compile(r"^---\s*\n(.*?)\n---", re.S)


def parse_key(key: str) -> dict:
    task, layer, doc, idx = key.split("::")
    return {"task": task, "layer": layer, "doc": doc, "item_idx": int(idx)}


@lru_cache(maxsize=4096)
def load_doc_text(task: str, layer: str, doc: str) -> Optional[str]:
    p = os.path.join(rubrics_dir(task), layer, doc)
    if not os.path.exists(p):
        return None
    try:
        raw = open(p, encoding="utf-8", errors="replace").read()
    except OSError:
        return None
    if layer == "raw" or doc.endswith(".html"):
        return html_to_text(raw)
    return _WS.sub(" ", _FRONT.sub(" ", raw)).strip()


@lru_cache(maxsize=4096)
def source_id(task: str, layer: str, doc: str) -> str:
    if layer == "claude-parsed":
        p = os.path.join(rubrics_dir(task), layer, doc)
        try:
            m = _FRONT.match(open(p, encoding="utf-8", errors="replace").read())
            if m:
                for line in m.group(1).splitlines():
                    if line.strip().startswith("source_url:"):
                        u = normalize_url(line.split(":", 1)[1])
                        if u:
                            return u
        except OSError:
            pass
    u = url_map(task).get(doc)
    if u:
        return u
    stem = re.sub(r"\.(html|md)$", "", doc)
    return f"{task}/{layer}/{stem}"


@lru_cache(maxsize=4096)
def load_parsed_item(task: str, layer: str, doc: str, item_idx: int) -> Optional[dict]:
    p = os.path.join(rubrics_dir(task), "gpt-parsed", "gpt-5-mini", f"{layer}__{doc}.json")
    if not os.path.exists(p):
        return None
    try:
        d = json.load(open(p))
    except (OSError, json.JSONDecodeError):
        return None
    ex = d.get("extracted") or {}
    items = ex.get("rubrics_metrics") or []
    if not (0 <= item_idx < len(items)):
        return None
    it = dict(items[item_idx])
    it["_strata"] = {k: ex.get(k) for k in ("subtask_short", "orientation", "intended_audience")}
    return it


def content_words(s: str) -> List[str]:
    return [w for w in _WORD.findall((s or "").casefold()) if w not in STOP]


def best_window(doc: str, pointer: str, win: int = 8000, step: int = 4000,
                head: int = 1200, cap: int = 10500) -> str:
    """Whole doc if short; else doc head + the window with the highest pointer-word overlap.
    The returned text is what the extractor sees AND what quotes are validated against."""
    if len(doc) <= cap:
        return doc
    pw = set(content_words(pointer))
    best, best_s = 0.0, head
    for s in range(0, max(1, len(doc) - win // 2), step):
        seg = doc[s: s + win].casefold()
        score = sum(1.0 for w in pw if w in seg)
        if score > best:
            best, best_s = score, s
    parts = [doc[:head]] if best_s > head else []
    parts.append(doc[best_s: best_s + win])
    return " [...] ".join(p.strip() for p in parts if p.strip())[:cap + 100]


def iter_canon(tasks: Optional[List[str]] = None):
    with open(CANON_PATH) as f:
        for line in f:
            r = json.loads(line)
            if tasks is None or r["task"] in tasks:
                yield r


def build_context(canon_row: dict) -> Optional[dict]:
    """One extraction context: pointer item + windowed original text + provenance/strata."""
    k = parse_key(canon_row["key"])
    doc_text = load_doc_text(k["task"], k["layer"], k["doc"])
    if not doc_text or len(doc_text) < 200:
        return None
    item = load_parsed_item(k["task"], k["layer"], k["doc"], k["item_idx"]) or {}
    name = item.get("name") or canon_row.get("canonical", "")[:80]
    desc = item.get("description") or canon_row.get("canonical", "")
    window = best_window(doc_text, f"{name} {desc}")
    return {
        "key": canon_row["key"], "task": k["task"], "bucket": canon_row.get("bucket"),
        "layer": k["layer"], "doc": k["doc"], "item_idx": k["item_idx"],
        "name": name, "description": desc, "guidance": item.get("guidance"),
        "canonical": canon_row.get("canonical"),
        "source_id": source_id(k["task"], k["layer"], k["doc"]),
        "strata": item.get("_strata") or {},
        "doc_text": window, "n_doc_chars": len(doc_text),
    }


def build_contexts(task: str, out_path: str, limit: int = 0) -> dict:
    """Materialize contexts_<task>.jsonl; resumable; returns coverage stats."""
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            done = {json.loads(l)["key"] for l in f if l.strip()}
    n_in = n_ok = n_skip = 0
    with open(out_path, "a") as out:
        for row in iter_canon([task]):
            n_in += 1
            if row["key"] in done:
                continue
            if limit and n_ok >= limit:
                break
            ctx = build_context(row)
            if ctx is None:
                n_skip += 1
                continue
            out.write(json.dumps(ctx) + "\n")
            n_ok += 1
    return {"task": task, "canon_rows": n_in, "written": n_ok,
            "already": len(done), "unresolvable": n_skip}
