"""Gradio interface for the Cedar & Crumb C4 LangGraph ReAct agent."""

from __future__ import annotations

from typing import Any

import gradio as gr

from agent_core import DEFAULT_PERSONA, PERSONAS, run_agent


def chat(
    message: str,
    history: list[dict[str, Any]],
    persona_name: str,
) -> str:
    """Run one agent turn while exposing only the final answer in the UI."""

    try:
        return run_agent(message, history, persona_name or DEFAULT_PERSONA)
    except RuntimeError as error:
        return f"Configuration needed: {error}"
    except Exception as error:  # Keep technical detail in the terminal, not the chat.
        print(f"[ERROR] {type(error).__name__}: {error}")
        return (
            "Something went wrong while contacting the bakery assistant. "
            "Please check the terminal and try again."
        )


persona_selector = gr.Dropdown(
    choices=list(PERSONAS),
    value=DEFAULT_PERSONA,
    label="Assistant persona",
    info="Choose a warmer concierge or a more formal specialist.",
)

demo = gr.ChatInterface(
    fn=chat,
    additional_inputs=[persona_selector],
    title="Cedar & Crumb Agent-Powered Bakery Assistant",
    description=(
        "Ask a verified business question, request a cake or corporate order, "
        "or leave feedback. The assistant uses a manual LangGraph ReAct loop "
        "and never displays private reasoning."
    ),
    examples=[
        ["What are your opening hours?", DEFAULT_PERSONA],
        ["I need a custom cake for 20 people.", DEFAULT_PERSONA],
        ["I have a suggestion about your packaging.", "Precise Bakery Specialist"],
        ["Do you have a loyalty card?", "Precise Bakery Specialist"],
    ],
    save_history=True,
)


if __name__ == "__main__":
    demo.launch()
