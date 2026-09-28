"""
reasoning_export.py
-------------------
Export of the agents' reasoning for qualitative analysis.

Writes three files next to the run output:

  reasoning.csv      one row per activity - the coding frame
  reasoning.jsonl    the same rows plus the full context each agent received
  run_manifest.json  run configuration and the verbatim system prompts

Silent tools are included: they carry reasoning even without a log event.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import pandas as pd

# Free-text fields across the tool schemas, in the order they are searched.
# The first one present on a tool output is that activity's justification.
_NOTE_FIELDS = [
    "initial_notes",
    "check_notes",
    "validation_notes",
    "risk_assessment",
    "rejection_notes",
    "approval_notes",
    "escalation_notes",
    "info_request",
    "notes",
    "reason",
]

# Structured fields worth having beside the prose when coding coherence.
_FLAG_FIELDS = [
    "recommendation",
    "risk_category",
    "score_acceptable",
    "documents_complete",
    "priority",
]


def _first_note(tool_output: dict) -> tuple[str, str]:
    """Return (field_name, text) for the first free-text field present."""
    for field in _NOTE_FIELDS:
        value = tool_output.get(field)
        if isinstance(value, str) and value.strip():
            return field, value.strip()
    return "", ""


def _flags(tool_output: dict) -> dict:
    return {
        f"flag_{f}": tool_output[f]
        for f in _FLAG_FIELDS
        if tool_output.get(f) is not None
    }


def build_rows(results) -> list[dict]:
    """Flatten CaseResult.agent_history into one row per activity."""
    rows: list[dict] = []

    for result in results:
        history = getattr(result, "agent_history", None) or []
        for step_index, action in enumerate(history, start=1):
            tool_output = action.get("tool_output") or {}
            note_field, note_text = _first_note(tool_output)

            row = {
                "case_id":        result.case_id,
                "case_status":    result.status,
                "rework_count":   result.rework_count,
                "step":           step_index,
                "n_steps_case":   len(history),
                "role":           action.get("agent_name", ""),
                "tool":           action.get("tool_name", ""),
                "sim_timestamp":  action.get("sim_timestamp"),
                "note_field":     note_field,
                "justification":  note_text,
                "complexity":     tool_output.get("case_complexity"),
                "complexity_rationale": tool_output.get("complexity_rationale", ""),
                "typical_minutes": tool_output.get("typical_duration_minutes"),
                "raw_content":    action.get("raw_content", ""),
                "context_chars":  len(action.get("context_sent", "") or ""),
                # Coding columns — left blank on purpose, filled in by hand.
                "coherence":      "",
                "criteria":       "",
                "asymmetry_ok":   "",
                "refers_to_prior": "",
            }
            row.update(_flags(tool_output))
            rows.append(row)

    return rows


def write_reasoning(results, output_dir: str, manifest: dict | None = None) -> dict:
    """Write the coding frame, the full records and the run manifest."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = build_rows(results)
    paths: dict[str, str] = {}

    if not rows:
        return paths

    # 1. The coding frame — open this in a spreadsheet and code it.
    df = pd.DataFrame(rows)
    csv_path = out / "reasoning.csv"
    df.to_csv(csv_path, index=False)
    paths["csv"] = str(csv_path)

    # 2. The full records, context included, for checking anything ambiguous.
    jsonl_path = out / "reasoning.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        by_case: dict[str, list] = {}
        for result in results:
            by_case[result.case_id] = getattr(result, "agent_history", None) or []
        for row in rows:
            history = by_case.get(row["case_id"], [])
            action = history[row["step"] - 1] if row["step"] <= len(history) else {}
            record = dict(row)
            record["context_sent"] = action.get("context_sent", "")
            record["tool_input"] = action.get("tool_input", {})
            record["tool_output"] = action.get("tool_output", {})
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    paths["jsonl"] = str(jsonl_path)

    # 3. The manifest — what produced these rows.
    if manifest:
        manifest_path = out / "run_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        paths["manifest"] = str(manifest_path)

    return paths


def build_manifest(config, process, cases, seed) -> dict:
    """Capture what the run was, including the prompts as they stood."""
    prompts = {}
    no_tool_calls = {}
    for role, agent_cls in process.agent_classes.items():
        prompts[role] = getattr(agent_cls, "system_prompt", "")

    cfg = asdict(config) if is_dataclass(config) else dict(vars(config))

    return {
        "process":       process.name,
        "n_cases":       len(cases),
        "seed":          seed,
        "config":        cfg,
        "system_prompts": prompts,
        "valid_transitions": {
            k: sorted(v) for k, v in process.valid_transitions.items()
        },
        "activity_map":  dict(process.activity_map),
        "silent_tools":  sorted(process.silent_tools),
        "no_tool_calls": no_tool_calls,
    }


def coverage(results) -> pd.DataFrame:
    """How much of the run actually carries a justification, by role and tool."""
    rows = build_rows(results)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["has_justification"] = df["justification"].str.len() > 0
    df["has_prose"] = df["raw_content"].str.len() > 0

    return (
        df.groupby(["role", "tool"])
        .agg(
            n=("case_id", "size"),
            with_justification=("has_justification", "sum"),
            with_prose=("has_prose", "sum"),
            mean_chars=("justification", lambda s: round(s.str.len().mean(), 0)),
        )
        .reset_index()
    )
