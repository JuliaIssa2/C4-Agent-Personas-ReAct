"""Small, reproducible C4 experiment plan and logging helpers.

Experiments use dry-run versions of write tools so evaluation does not create fake
customer leads or feedback records. Business lookup still uses the real local data.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import RateLimitError

from agent_core import PERSONAS, PromptStrategy, ROOT, run_agent_with_trace


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    comparison: str
    persona: str
    prompt_strategy: PromptStrategy
    user_prompt: str
    config_label: str = "persona_default"
    api_overrides: dict[str, Any] | None = None


EXPERIMENT_PLAN = [
    # Persona + matching configuration comparison on three different tasks.
    ExperimentSpec(
        "P01",
        "persona_and_config",
        "Friendly Bakery Concierge",
        "standard",
        "What are your opening hours?",
    ),
    ExperimentSpec(
        "P02",
        "persona_and_config",
        "Precise Bakery Specialist",
        "standard",
        "What are your opening hours?",
    ),
    ExperimentSpec(
        "P03",
        "persona_and_config",
        "Friendly Bakery Concierge",
        "standard",
        "I need a custom cake for 20 people.",
    ),
    ExperimentSpec(
        "P04",
        "persona_and_config",
        "Precise Bakery Specialist",
        "standard",
        "I need a custom cake for 20 people.",
    ),
    ExperimentSpec(
        "P05",
        "persona_and_config",
        "Friendly Bakery Concierge",
        "standard",
        "Do you offer nut-free cakes?",
    ),
    ExperimentSpec(
        "P06",
        "persona_and_config",
        "Precise Bakery Specialist",
        "standard",
        "Do you offer nut-free cakes?",
    ),
    # Zero-shot vs few-shot: the few-shot condition includes the worked example.
    ExperimentSpec(
        "E01",
        "zero_shot_vs_few_shot",
        "Friendly Bakery Concierge",
        "zero_shot",
        "I need a custom cake for 20 people.",
    ),
    ExperimentSpec(
        "E02",
        "zero_shot_vs_few_shot",
        "Friendly Bakery Concierge",
        "few_shot",
        "I need a custom cake for 20 people.",
    ),
    # Without vs with private reasoning. Only final answers are logged.
    ExperimentSpec(
        "E03",
        "without_vs_with_private_reasoning",
        "Precise Bakery Specialist",
        "standard",
        "Do you have a loyalty card?",
    ),
    ExperimentSpec(
        "E04",
        "without_vs_with_private_reasoning",
        "Precise Bakery Specialist",
        "private_reasoning",
        "Do you have a loyalty card?",
    ),
    # System prompt vs the same detailed guidance placed in the user message.
    ExperimentSpec(
        "E05",
        "system_vs_in_message",
        "Friendly Bakery Concierge",
        "few_shot",
        "I have a suggestion about your packaging.",
    ),
    ExperimentSpec(
        "E06",
        "system_vs_in_message",
        "Friendly Bakery Concierge",
        "in_message",
        "I have a suggestion about your packaging.",
    ),
    # Same persona and prompt; vary supported reasoning/output controls only.
    ExperimentSpec(
        "C01",
        "supported_config_controls",
        "Precise Bakery Specialist",
        "standard",
        "What are your opening hours?",
        "minimal_reasoning_280_tokens",
        {"reasoning_effort": "minimal", "max_tokens": 280},
    ),
    ExperimentSpec(
        "C02",
        "supported_config_controls",
        "Precise Bakery Specialist",
        "standard",
        "What are your opening hours?",
        "low_reasoning_420_tokens",
        {"reasoning_effort": "low", "max_tokens": 420},
    ),
]


def run_experiment_suite(
    plan: list[ExperimentSpec] | None = None,
    *,
    inter_experiment_delay: float = 10.0,
    rate_limit_delay: float = 40.0,
    max_rate_limit_retries: int = 3,
    checkpoint_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Run a paced plan and return rows ready for manual rubric scoring.

    Gemini's free tier enforces a per-minute request limit. A single ReAct example
    can use multiple model requests, so the harness spaces examples, honors a
    cooldown after HTTP 429, and can checkpoint after every completed row.
    """

    rows: list[dict[str, Any]] = []
    selected_plan = plan or EXPERIMENT_PLAN
    for index, spec in enumerate(selected_plan):
        for attempt in range(max_rate_limit_retries + 1):
            try:
                result = run_agent_with_trace(
                    spec.user_prompt,
                    history=[],
                    persona_name=spec.persona,
                    prompt_strategy=spec.prompt_strategy,
                    dry_run_tools=True,
                    api_overrides=spec.api_overrides,
                )
                break
            except RateLimitError:
                if attempt >= max_rate_limit_retries:
                    raise
                print(
                    f"Rate limit reached during {spec.experiment_id}; "
                    f"retrying after {rate_limit_delay:.0f} seconds."
                )
                time.sleep(rate_limit_delay)
        config = dict(PERSONAS[spec.persona]["api_config"])
        config.update(spec.api_overrides or {})
        rows.append(
            {
                **asdict(spec),
                "model": config["model"],
                "reasoning_effort": config["reasoning_effort"],
                "max_tokens": config["max_tokens"],
                "temperature": None,
                "top_p": None,
                "tools_used": result.tools_used,
                "iterations": result.iterations,
                "final_response": result.final_answer,
                "correctness": None,
                "helpfulness": None,
                "tone_match": None,
                "hallucination": None,
                "notes": "Add manual scores after reviewing the final answer.",
            }
        )
        if checkpoint_path is not None:
            save_experiment_results(rows, checkpoint_path)
        if inter_experiment_delay > 0 and index < len(selected_plan) - 1:
            time.sleep(inter_experiment_delay)
    return rows


def save_experiment_results(
    rows: list[dict[str, Any]],
    path: Path | None = None,
) -> Path:
    """Save reproducible JSONL results under ignored runtime data."""

    output_path = path or (ROOT / "data" / "c4_experiments.jsonl")
    output_path.parent.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    with output_path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(
                json.dumps({"run_timestamp_utc": timestamp, **row}, ensure_ascii=True)
                + "\n"
            )
    return output_path


def results_markdown(rows: list[dict[str, Any]]) -> str:
    """Build a compact notebook table without adding pandas as a dependency."""

    headers = [
        "experiment_id",
        "persona",
        "prompt_strategy",
        "reasoning_effort",
        "max_tokens",
        "tools_used",
        "final_response",
        "correctness",
        "helpfulness",
        "tone_match",
        "hallucination",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = []
        for header in headers:
            value = row.get(header, "")
            if isinstance(value, list):
                value = ", ".join(str(item) for item in value)
            values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def comparison_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average completed manual scores; leave unscored groups clearly pending."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["comparison"]), []).append(row)

    summary = []
    score_fields = ["correctness", "helpfulness", "tone_match"]
    for comparison, group in grouped.items():
        scored = [
            row
            for row in group
            if all(isinstance(row.get(field), (int, float)) for field in score_fields)
        ]
        if not scored:
            summary.append(
                {
                    "comparison": comparison,
                    "status": "NEEDS MANUAL SCORING",
                    "average_score": None,
                }
            )
            continue
        average = sum(
            sum(float(row[field]) for field in score_fields) / len(score_fields)
            for row in scored
        ) / len(scored)
        summary.append(
            {
                "comparison": comparison,
                "status": "SCORED",
                "average_score": round(average, 2),
            }
        )
    return summary


if __name__ == "__main__":
    result_path = ROOT / "data" / "c4_experiments.jsonl"
    experiment_rows = run_experiment_suite(checkpoint_path=result_path)
    saved_to = save_experiment_results(experiment_rows, result_path)
    print(f"Saved {len(experiment_rows)} experiment rows to {saved_to}")
