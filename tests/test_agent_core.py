"""Offline tests for the manual ReAct mechanics (no API key required)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from agent_core import (
    BUSINESS_KNOWLEDGE,
    PERSONAS,
    build_initial_messages,
    build_system_prompt,
    detect_action,
    execute_tool_call,
    lookup_business_info,
    run_agent_with_trace,
)


class FakeMessage:
    def __init__(self, payload):
        self.payload = payload

    def model_dump(self, exclude_none=True):
        if exclude_none:
            return {key: value for key, value in self.payload.items() if value is not None}
        return dict(self.payload)


class FakeCompletions:
    def __init__(self, payloads):
        self.payloads = iter(payloads)
        self.requests = []

    def create(self, **request):
        self.requests.append(request)
        message = FakeMessage(next(self.payloads))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    def __init__(self, payloads):
        self.chat = SimpleNamespace(completions=FakeCompletions(payloads))


class AgentCoreTests(unittest.TestCase):
    def test_persona_configs_avoid_deprecated_sampling_parameters(self):
        for persona in PERSONAS.values():
            self.assertNotIn("temperature", persona["api_config"])
            self.assertNotIn("top_p", persona["api_config"])
            self.assertIn("reasoning_effort", persona["api_config"])
            self.assertIn("max_tokens", persona["api_config"])

    def test_lookup_finds_verified_hours(self):
        result = lookup_business_info("opening hours")
        self.assertEqual(result["status"], "found")
        self.assertIn("Tuesday", " ".join(result["excerpts"]))

    def test_lookup_rejects_unknown_fact(self):
        result = lookup_business_info("loyalty card points")
        self.assertEqual(result["status"], "not_found")
        self.assertIn("does not mean", result["message"])

    def test_business_facts_are_available_only_through_lookup(self):
        prompt = build_system_prompt("Friendly Bakery Concierge")
        self.assertNotIn(BUSINESS_KNOWLEDGE, prompt)
        self.assertIn("only through lookup_business_info", prompt)

    def test_action_detection(self):
        self.assertTrue(detect_action({"tool_calls": [{"id": "1"}]}))
        self.assertFalse(detect_action({"role": "assistant", "content": "Done"}))

    def test_dry_run_lead_validates_without_writing(self):
        name, result = execute_tool_call(
            {
                "id": "lead-1",
                "type": "function",
                "function": {
                    "name": "record_customer_interest",
                    "arguments": (
                        '{"email":"julia@example.com","name":"Julia",'
                        '"message":"Custom cake for 20 people"}'
                    ),
                },
            },
            dry_run_tools=True,
        )
        self.assertEqual(name, "record_customer_interest")
        self.assertEqual(result["status"], "recorded")
        self.assertTrue(result["simulation"])

    def test_short_term_history_is_preserved(self):
        messages = build_initial_messages(
            "Julia, julia@example.com",
            [
                {"role": "user", "content": "I need a cake for 20 people."},
                {"role": "assistant", "content": "May I have your name and email?"},
            ],
            "Friendly Bakery Concierge",
            "standard",
        )
        self.assertEqual(messages[-3]["role"], "user")
        self.assertIn("20 people", messages[-3]["content"])
        self.assertIn("julia@example.com", messages[-1]["content"])

    def test_manual_graph_executes_action_observation_answer_cycle(self):
        fake_client = FakeClient(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "lookup-1",
                            "type": "function",
                            "function": {
                                "name": "lookup_business_info",
                                "arguments": '{"topic":"opening hours"}',
                            },
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": (
                        "We are open Tuesday through Saturday from 10:00 to 18:00."
                    ),
                },
            ]
        )
        result = run_agent_with_trace(
            "What are your opening hours?",
            persona_name="Precise Bakery Specialist",
            client=fake_client,
            dry_run_tools=True,
        )
        self.assertEqual(result.tools_used, ["lookup_business_info"])
        self.assertEqual(result.iterations, 2)
        self.assertIn("Tuesday through Saturday", result.final_answer)
        self.assertEqual(
            [entry["stage"] for entry in result.react_log],
            ["Action", "Observation"],
        )


if __name__ == "__main__":
    unittest.main()
