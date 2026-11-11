#!/usr/bin/env python3
"""One-shot measurement runner for PREREG-21 Leg 3b.

The realism gate is intentionally a separate command from material construction.
It refuses to overwrite its output, so a completed gate cannot be tuned or rerun.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from methods.metric_implementer import backends as B
from methods.metric_implementer import config as C


OUT = ROOT / "outputs/lexicon/extraction_validity_20260726"
PAGES = OUT / "leg3b_splice_pages.json"
KEY = OUT / "leg3b_planting_key_private.json"
GATE = OUT / "leg3b_realism_gate.jsonl"
GATE_TRANSPORT = OUT / "leg3b_realism_gate_transport_failure.jsonl"
GATE_CODEX_LOG = OUT / "leg3b_realism_gate_codex_runner.log"
GATE_CLAUDE_LOG = OUT / "leg3b_realism_gate_claude_runner.log"
MODEL = "glm-5"
BACKEND = "zai_anthropic"
SHUFFLE_SEED = 20260726

SYSTEM = """You are an adversarial forensic editor judging webpage prose.
Classify whether the supplied page is:
- real: an untouched excerpt from a real human-authored or naturally published page; or
- synthetic: generated, edited, or spliced together for an experiment.

Look for abrupt topic/register changes, inserted policy-like sentences, incoherent
transitions, templated prose, or other assembly artifacts. Be skeptical in both
directions: real pages may be awkward, scraped, or contain navigation noise.
Return strict JSON only:
{"verdict":"real"|"synthetic","confidence":0.0-1.0,"reason":"brief reason"}"""


def load(path: Path):
    with path.open() as f:
        return json.load(f)


def parse(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw or "", re.S)
    if not match:
        return {"verdict": None, "confidence": None, "reason": None, "parse_status": "no_json"}
    try:
        value = json.loads(match.group())
    except json.JSONDecodeError as exc:
        return {
            "verdict": None, "confidence": None, "reason": None,
            "parse_status": f"json_error:{exc.msg}",
        }
    verdict = str(value.get("verdict", "")).casefold()
    confidence = value.get("confidence")
    if verdict not in {"real", "synthetic"}:
        return {
            "verdict": None, "confidence": confidence, "reason": value.get("reason"),
            "parse_status": "bad_verdict",
        }
    return {
        "verdict": verdict,
        "confidence": confidence,
        "reason": value.get("reason"),
        "parse_status": "ok",
    }


def run_gate() -> None:
    if GATE.exists():
        raise SystemExit("realism gate exists; refusing to rerun the confirmatory gate")
    pages = load(PAGES)
    key = load(KEY)
    public_hash = hashlib.sha256(PAGES.read_bytes()).hexdigest()
    if public_hash != key["public_pages_sha256"]:
        raise RuntimeError("public splice-page hash no longer matches the pre-gate freeze")

    items = [
        {
            "truth": "synthetic",
            "source_item_id": page["page_id"],
            "text": page["page_text"],
            "sha256": hashlib.sha256(page["page_text"].encode()).hexdigest(),
        }
        for page in pages["pages"]
    ] + [
        {
            "truth": "real",
            "source_item_id": control["control_id"],
            "text": control["page_text"],
            "sha256": control["page_sha256"],
        }
        for control in key["realism_controls_private"]
    ]
    random.Random(SHUFFLE_SEED).shuffle(items)
    prompts = [
        f"BLINDED ITEM g{i:02d}\n\nPAGE TEXT:\n{item['text']}"
        for i, item in enumerate(items)
    ]
    cfg = C.ImplementerConfig(backend=BACKEND)
    backend = B.LLMBackend(MODEL, "prereg21_realism_gate", cfg)
    raws = backend.generate_batch(
        prompts, system=SYSTEM, max_tokens=180, temperature=0.0, seed=SHUFFLE_SEED
    )
    if len(raws) != len(items):
        raise RuntimeError(f"judge returned {len(raws)} responses for {len(items)} inputs")

    prompt_hash = hashlib.sha256(SYSTEM.encode()).hexdigest()
    with GATE.open("x") as out:
        for i, (item, raw) in enumerate(zip(items, raws)):
            parsed = parse(raw or "")
            record = {
                "schema": "prereg21-leg3b-realism-gate-v1",
                "gate_run": "single_confirmatory_run",
                "judge_model": MODEL,
                "judge_backend": BACKEND,
                "temperature": 0.0,
                "seed": SHUFFLE_SEED,
                "system_prompt_sha256": prompt_hash,
                "public_pages_sha256": public_hash,
                "blind_item_id": f"g{i:02d}",
                "source_item_id": item["source_item_id"],
                "truth": item["truth"],
                "text_sha256": item["sha256"],
                **parsed,
                "raw_response": raw,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    records = [json.loads(line) for line in GATE.read_text().splitlines() if line]
    bad = [r for r in records if r["parse_status"] != "ok"]
    synth = [r for r in records if r["truth"] == "synthetic"]
    real = [r for r in records if r["truth"] == "real"]
    fooled = sum(r["verdict"] == "real" for r in synth)
    real_correct = sum(r["verdict"] == "real" for r in real)
    print(json.dumps({
        "n": len(records),
        "parse_failures": len(bad),
        "splice_fooling_rate": [fooled, len(synth)],
        "real_page_accuracy": [real_correct, len(real)],
        "surviving_splice_pages": [
            r["source_item_id"] for r in synth if r["verdict"] == "real"
        ],
    }, indent=2))


def run_gate_codex_after_transport_failure() -> None:
    """Make the one semantic gate run after the first backend returned no content."""
    if not GATE.exists():
        raise SystemExit("expected the recorded empty-response transport attempt")
    existing = [
        json.loads(line) for line in GATE.read_text().splitlines() if line.strip()
    ]
    if len(existing) != 40 or any(r.get("raw_response") for r in existing):
        raise SystemExit("prior gate contains semantic responses; refusing a second gate")
    if GATE_TRANSPORT.exists() or GATE_CODEX_LOG.exists():
        raise SystemExit("transport archive or Codex gate log already exists; refusing rerun")

    pages = load(PAGES)
    key = load(KEY)
    public_hash = hashlib.sha256(PAGES.read_bytes()).hexdigest()
    if public_hash != key["public_pages_sha256"]:
        raise RuntimeError("public splice-page hash no longer matches the pre-gate freeze")
    lookup = {
        page["page_id"]: {
            "truth": "synthetic",
            "source_item_id": page["page_id"],
            "text": page["page_text"],
            "sha256": hashlib.sha256(page["page_text"].encode()).hexdigest(),
        }
        for page in pages["pages"]
    }
    lookup.update({
        control["control_id"]: {
            "truth": "real",
            "source_item_id": control["control_id"],
            "text": control["page_text"],
            "sha256": control["page_sha256"],
        }
        for control in key["realism_controls_private"]
    })
    # Preserve the exact blinded order of the transport attempt.
    items = [lookup[r["source_item_id"]] for r in existing]
    blocks = [
        f'<item id="g{i:02d}">\n{item["text"]}\n</item>'
        for i, item in enumerate(items)
    ]
    prompt = """Act as the fresh adversarial realism judge for a confirmatory study.
You receive shuffled webpage excerpts. For each item independently, decide:
- real: an untouched excerpt from a real human-authored or naturally published page
- synthetic: generated, edited, or spliced together for an experiment

Look for abrupt topic/register changes, inserted policy-like sentences, incoherent
transitions, templated prose, or assembly artifacts. Real pages may be awkward,
scraped, or noisy. Do not use tools or inspect repository files. Judge only the text
below. Return one verdict for every item in the supplied order.

""" + "\n\n".join(blocks)
    schema = {
        "type": "object",
        "properties": {
            "verdicts": {
                "type": "array", "minItems": 40, "maxItems": 40,
                "items": {
                    "type": "object",
                    "properties": {
                        "blind_item_id": {"type": "string"},
                        "verdict": {"type": "string", "enum": ["real", "synthetic"]},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "reason": {"type": "string"},
                    },
                    "required": ["blind_item_id", "verdict", "confidence", "reason"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["verdicts"],
        "additionalProperties": False,
    }
    with tempfile.TemporaryDirectory(prefix="prereg21_gate_") as tmp:
        tmp_path = Path(tmp)
        schema_path = tmp_path / "schema.json"
        result_path = tmp_path / "result.json"
        schema_path.write_text(json.dumps(schema))
        completed = subprocess.run(
            [
                "codex", "exec", "--ephemeral", "--sandbox", "read-only",
                "--model", "gpt-5.6-sol", "--output-schema", str(schema_path),
                "--output-last-message", str(result_path), "-",
            ],
            input=prompt, text=True, capture_output=True, timeout=900,
            cwd=ROOT,
        )
        GATE_CODEX_LOG.write_text(
            f"returncode={completed.returncode}\n\nSTDOUT\n{completed.stdout}"
            f"\n\nSTDERR\n{completed.stderr}"
        )
        if completed.returncode != 0 or not result_path.exists():
            raise RuntimeError(
                f"fresh Codex judge failed; see {GATE_CODEX_LOG.relative_to(ROOT)}"
            )
        result = json.loads(result_path.read_text())

    verdicts = result.get("verdicts", [])
    if len(verdicts) != 40:
        raise RuntimeError(f"fresh judge returned {len(verdicts)} of 40 verdicts")
    expected_ids = [f"g{i:02d}" for i in range(40)]
    returned_ids = [v.get("blind_item_id") for v in verdicts]
    if returned_ids != expected_ids:
        raise RuntimeError(f"fresh judge item order mismatch: {returned_ids}")

    # Preserve the non-semantic API failure, then create the required gate artifact.
    GATE.replace(GATE_TRANSPORT)
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    with GATE.open("x") as out:
        for i, (item, value) in enumerate(zip(items, verdicts)):
            record = {
                "schema": "prereg21-leg3b-realism-gate-v1",
                "gate_run": "single_confirmatory_semantic_run",
                "judge_model": "gpt-5.6-sol",
                "judge_backend": "codex_exec_ephemeral",
                "temperature": "service_default",
                "seed": None,
                "input_order_source": "preserved from transport attempt",
                "prompt_sha256": prompt_hash,
                "public_pages_sha256": public_hash,
                "blind_item_id": f"g{i:02d}",
                "source_item_id": item["source_item_id"],
                "truth": item["truth"],
                "text_sha256": item["sha256"],
                "verdict": value["verdict"],
                "confidence": value["confidence"],
                "reason": value["reason"],
                "parse_status": "ok",
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    records = [json.loads(line) for line in GATE.read_text().splitlines() if line]
    synth = [r for r in records if r["truth"] == "synthetic"]
    real = [r for r in records if r["truth"] == "real"]
    fooled = sum(r["verdict"] == "real" for r in synth)
    real_correct = sum(r["verdict"] == "real" for r in real)
    print(json.dumps({
        "n": len(records),
        "parse_failures": 0,
        "splice_fooling_rate": [fooled, len(synth)],
        "real_page_accuracy": [real_correct, len(real)],
        "surviving_splice_pages": [
            r["source_item_id"] for r in synth if r["verdict"] == "real"
        ],
    }, indent=2))


def run_gate_claude_after_transport_failures() -> None:
    """Run the semantic gate after GLM silence and Codex initialization failure."""
    if not GATE.exists():
        raise SystemExit("expected the recorded empty-response GLM attempt")
    existing = [
        json.loads(line) for line in GATE.read_text().splitlines() if line.strip()
    ]
    if len(existing) != 40 or any(r.get("raw_response") for r in existing):
        raise SystemExit("a prior gate has semantic responses; refusing a second gate")
    if not GATE_CODEX_LOG.exists() or "failed to initialize" not in GATE_CODEX_LOG.read_text():
        raise SystemExit("expected a documented pre-inference Codex initialization failure")
    if GATE_TRANSPORT.exists() or GATE_CLAUDE_LOG.exists():
        raise SystemExit("gate transport archive or Claude log already exists; refusing rerun")

    pages = load(PAGES)
    key = load(KEY)
    public_hash = hashlib.sha256(PAGES.read_bytes()).hexdigest()
    if public_hash != key["public_pages_sha256"]:
        raise RuntimeError("public splice-page hash no longer matches the pre-gate freeze")
    lookup = {
        page["page_id"]: {
            "truth": "synthetic", "source_item_id": page["page_id"],
            "text": page["page_text"],
            "sha256": hashlib.sha256(page["page_text"].encode()).hexdigest(),
        }
        for page in pages["pages"]
    }
    lookup.update({
        control["control_id"]: {
            "truth": "real", "source_item_id": control["control_id"],
            "text": control["page_text"], "sha256": control["page_sha256"],
        }
        for control in key["realism_controls_private"]
    })
    items = [lookup[r["source_item_id"]] for r in existing]
    blocks = [
        f'<item id="g{i:02d}">\n{item["text"]}\n</item>'
        for i, item in enumerate(items)
    ]
    prompt = """You are the fresh adversarial realism judge for a confirmatory study.
You receive shuffled webpage excerpts. For each item independently, decide:
- real: an untouched excerpt from a real human-authored or naturally published page
- synthetic: generated, edited, or spliced together for an experiment

Look for abrupt topic/register changes, inserted policy-like sentences, incoherent
transitions, templated prose, or assembly artifacts. Real pages may be awkward,
scraped, or noisy. Do not use tools or inspect repository files. Judge only the text
below. Return one verdict for every item in the supplied order.

""" + "\n\n".join(blocks)
    schema = {
        "type": "object",
        "properties": {
            "verdicts": {
                "type": "array", "minItems": 40, "maxItems": 40,
                "items": {
                    "type": "object",
                    "properties": {
                        "blind_item_id": {"type": "string"},
                        "verdict": {"type": "string", "enum": ["real", "synthetic"]},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "reason": {"type": "string"},
                    },
                    "required": ["blind_item_id", "verdict", "confidence", "reason"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["verdicts"],
        "additionalProperties": False,
    }
    completed = subprocess.run(
        [
            "claude", "--print", "--model", "sonnet", "--no-session-persistence",
            "--permission-mode", "plan", "--disable-slash-commands",
            "--json-schema", json.dumps(schema), "--output-format", "json",
        ],
        input=prompt, text=True, capture_output=True, timeout=900, cwd=ROOT,
    )
    GATE_CLAUDE_LOG.write_text(
        f"returncode={completed.returncode}\n\nSTDOUT\n{completed.stdout}"
        f"\n\nSTDERR\n{completed.stderr}"
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"fresh Claude judge failed; see {GATE_CLAUDE_LOG.relative_to(ROOT)}"
        )
    envelope = json.loads(completed.stdout)
    result = envelope.get("structured_output")
    if result is None:
        result = envelope.get("result")
    if isinstance(result, str):
        result = json.loads(result)
    if not isinstance(result, dict):
        raise RuntimeError("Claude runner returned no structured gate output")
    verdicts = result.get("verdicts", [])
    if len(verdicts) != 40:
        raise RuntimeError(f"fresh judge returned {len(verdicts)} of 40 verdicts")
    expected_ids = [f"g{i:02d}" for i in range(40)]
    returned_ids = [v.get("blind_item_id") for v in verdicts]
    if returned_ids != expected_ids:
        raise RuntimeError(f"fresh judge item order mismatch: {returned_ids}")

    GATE.replace(GATE_TRANSPORT)
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    with GATE.open("x") as out:
        for i, (item, value) in enumerate(zip(items, verdicts)):
            record = {
                "schema": "prereg21-leg3b-realism-gate-v1",
                "gate_run": "single_confirmatory_semantic_run",
                "judge_model": envelope.get("model", "sonnet"),
                "judge_backend": "claude_cli_nonpersistent",
                "temperature": "service_default",
                "seed": None,
                "input_order_source": "preserved from transport attempt",
                "prompt_sha256": prompt_hash,
                "public_pages_sha256": public_hash,
                "blind_item_id": f"g{i:02d}",
                "source_item_id": item["source_item_id"],
                "truth": item["truth"],
                "text_sha256": item["sha256"],
                "verdict": value["verdict"],
                "confidence": value["confidence"],
                "reason": value["reason"],
                "parse_status": "ok",
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    records = [json.loads(line) for line in GATE.read_text().splitlines() if line]
    synth = [r for r in records if r["truth"] == "synthetic"]
    real = [r for r in records if r["truth"] == "real"]
    fooled = sum(r["verdict"] == "real" for r in synth)
    real_correct = sum(r["verdict"] == "real" for r in real)
    print(json.dumps({
        "n": len(records),
        "parse_failures": 0,
        "splice_fooling_rate": [fooled, len(synth)],
        "real_page_accuracy": [real_correct, len(real)],
        "surviving_splice_pages": [
            r["source_item_id"] for r in synth if r["verdict"] == "real"
        ],
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "gate", "gate-codex-after-transport-failure",
            "gate-claude-after-transport-failures",
        ],
    )
    args = parser.parse_args()
    if args.command == "gate":
        run_gate()
    elif args.command == "gate-codex-after-transport-failure":
        run_gate_codex_after_transport_failure()
    else:
        run_gate_claude_after_transport_failures()


if __name__ == "__main__":
    main()
