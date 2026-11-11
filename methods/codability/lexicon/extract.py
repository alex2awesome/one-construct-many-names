"""Key-term extraction: prompt, mechanical validation, batch runner (GLM/zai backend).

The extractor reads the ORIGINAL document and records the author's own wording. Nothing the
model returns is trusted: the quote must appear verbatim (whitespace/case-normalized) in the
text it was shown; every key term must appear in that text; violations are retried with a
different seed (never a sampling distortion) and finally recorded with a reject status.
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

SYSTEM = (
    "You are a careful corpus annotator for a study of how expert communities write evaluation "
    "criteria. You are given a SOURCE DOCUMENT (text extracted from a web page, possibly noisy) "
    "and one CRITERION that a previous system derived from this document. Go back to the source "
    "itself and record how the ORIGINAL AUTHOR expressed this criterion, in the author's own "
    "words.\n"
    "Rules:\n"
    "1. \"quote\": copy VERBATIM from the source document (no paraphrase, no fixing typos), at "
    "most 60 words — the passage that best expresses the criterion.\n"
    "2. \"key_terms\": up to 8 words or short phrases the AUTHOR uses for the key evaluative "
    "concepts of this criterion. Every term must appear in the source document. Never substitute "
    "your own synonyms or the criterion's wording for the author's.\n"
    "3. \"head_term\": the name or term of art the AUTHOR uses to LABEL this concept — only if "
    "the author actually uses it as a name for it. If the author describes the idea without "
    "naming it, head_term is null and named_in_source is false.\n"
    "4. If the document does not actually express this criterion, set found=false and leave "
    "quote empty and key_terms empty.\n"
    "Reply with STRICT JSON only, no prose, exactly these fields: "
    "{\"found\": true|false, \"quote\": \"...\", \"key_terms\": [\"...\"], "
    "\"head_term\": \"...\"|null, \"named_in_source\": true|false}"
)

USER_TMPL = ("CRITERION (as previously extracted):\nname: {name}\ndescription: {desc}\n\n"
             "SOURCE DOCUMENT:\n{doc}")

_JSON = re.compile(r"\{.*\}", re.S)
_TOKEN = re.compile(r"[a-z0-9]+")


def _seq(s: str) -> str:
    """Compatibility display form; validation itself uses boundary-preserving token sequences."""
    return " ".join(_TOKEN.findall((s or "").casefold()))


def _contains_phrase(needle: str, haystack: str) -> bool:
    """Contiguous, casefolded token match that preserves word boundaries.

    Punctuation/markdown/line wraps may differ, but ``the rapist`` no longer matches ``therapist``.
    """
    n = _TOKEN.findall((needle or "").casefold())
    h = _TOKEN.findall((haystack or "").casefold())
    return bool(n) and any(h[i:i + len(n)] == n for i in range(len(h) - len(n) + 1))


def build_prompt(ctx: dict) -> str:
    return USER_TMPL.format(name=ctx.get("name") or "", desc=ctx.get("description") or "",
                            doc=ctx.get("doc_text") or "")


def parse_reply(txt: str) -> Optional[dict]:
    m = _JSON.search(txt or "")
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict) or "found" not in d:
        return None
    return d


def validate(d: dict, doc_text: str) -> str:
    """'' if valid else reject reason. In-source checks run in _seq space (word-level verbatim)."""
    if type(d.get("found")) is not bool or type(d.get("named_in_source")) is not bool:
        return "flags_not_bool"
    found = d["found"]
    quote, terms = d.get("quote") or "", d.get("key_terms") or []
    if not isinstance(terms, list):
        return "terms_not_list"
    if not found:
        if quote or terms or d.get("head_term") is not None or d["named_in_source"]:
            return "notfound_nonempty"
        return ""
    if not quote:
        return "empty_quote"
    if len(quote.split()) > 60:
        return "quote_too_long"
    if not _contains_phrase(quote, doc_text):
        return "quote_not_in_source"
    if len(terms) > 8:
        return "too_many_terms"
    for t in terms:
        if not isinstance(t, str) or not t.strip():
            return "bad_term"
        if not _contains_phrase(t, doc_text):
            return "term_not_in_source"
    head = d.get("head_term")
    if head is not None and not _contains_phrase(str(head), doc_text):
        return "head_not_in_source"
    if bool(d.get("named_in_source")) and head is None:
        return "named_without_head"
    if not d["named_in_source"] and head is not None:
        return "head_without_named"
    return ""


def finalize(ctx: dict, d: Optional[dict], reason: str, model: str) -> dict:
    ok = d is not None and not reason
    rec = {"key": ctx["key"], "task": ctx["task"], "bucket": ctx.get("bucket"),
           "layer": ctx.get("layer"), "source_id": ctx.get("source_id"),
           "strata": ctx.get("strata") or {}, "model": model,
           "status": "ok" if ok else (reason or "parse_error"),
           "found": bool(d.get("found")) if d else None,
           "quote": (d.get("quote") or "") if d else "",
           "key_terms": (d.get("key_terms") or []) if d else [],
           "head_term": d.get("head_term") if d else None,
           "named_in_source": bool(d.get("named_in_source")) if d else None,
           "name": ctx.get("name"), "canonical": ctx.get("canonical")}
    return rec


class GLMExtractor:
    """Batched extraction via the zai_anthropic backend (HTTP, 0-GPU). Quota is binding —
    caller controls volume; this class just runs what it is given, with retry-by-seed."""

    def __init__(self, model: str = "glm-4.7", backend: str = "zai_anthropic",
                 max_tokens: int = 300):
        from methods.metric_implementer import backends as _b, config as _c
        cfg = _c.ImplementerConfig(backend=backend)
        self.be = _b.LLMBackend(model, "lexicon_extractor", cfg)
        self.model, self.max_tokens = model, max_tokens

    def run(self, contexts: List[dict], out_path: str, flush_every: int = 100,
            retries: int = 2) -> Dict[str, int]:
        done = set()
        if os.path.exists(out_path):
            with open(out_path) as f:
                done = {json.loads(l)["key"] for l in f if l.strip()}
        todo = [c for c in contexts if c["key"] not in done]
        stats = {"ok": 0, "rejected": 0, "skipped": len(contexts) - len(todo)}
        out = open(out_path, "a")
        for lo in range(0, len(todo), flush_every):
            chunk = todo[lo: lo + flush_every]
            pending = list(range(len(chunk)))
            results: Dict[int, tuple] = {}
            for attempt in range(retries + 1):
                if not pending:
                    break
                prompts = [build_prompt(chunk[i]) for i in pending]
                temp = 0.0 if attempt == 0 else 0.5
                outs = self.be.generate_batch(prompts, system=SYSTEM,
                                              max_tokens=self.max_tokens,
                                              temperature=temp, seed=attempt)
                nxt = []
                for i, o in zip(pending, outs):
                    d = parse_reply(o or "")
                    reason = "parse_error" if d is None else validate(d, chunk[i]["doc_text"])
                    if d is not None and not reason:
                        results[i] = (d, "")
                    else:
                        results[i] = (d, reason or "parse_error")
                        nxt.append(i)
                pending = nxt
            for i, c in enumerate(chunk):
                d, reason = results.get(i, (None, "no_reply"))
                rec = finalize(c, d, reason, self.model)
                stats["ok" if rec["status"] == "ok" else "rejected"] += 1
                out.write(json.dumps(rec) + "\n")
            out.flush()
            print(f"  [{self.model}] {min(lo + flush_every, len(todo))}/{len(todo)} "
                  f"(ok {stats['ok']}, rej {stats['rejected']})", flush=True)
        out.close()
        return stats


def load_extractions(path: str, only_ok: bool = True) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if (not only_ok) or r.get("status") == "ok":
                    out[r["key"]] = r
    return out
