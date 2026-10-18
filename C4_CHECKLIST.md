# C4 Assignment Compliance Checklist

Status meanings:

- **COMPLETE** - implemented and inspectable in the project.
- **NEEDS MANUAL RUN** - implementation exists, but a local Python/API execution or
  human score is still required.
- **OPTIONAL** - not required by the official PDF and intentionally not essential.

## Official assignment requirements

| Requirement | Where implemented | Status |
|---|---|---|
| Extend the prior business chatbot into a ReAct-style agent | `agent_core.py`; notebook Sections 10-15 | COMPLETE |
| Use one approved framework | LangGraph in `agent_core.py` | COMPLETE |
| Do not use a prebuilt ReAct executor / AgentExecutor | Explicit `StateGraph`, nodes, conditional edges, and loop in `create_react_graph` | COMPLETE |
| Define a fictional or real-world scenario | Notebook Section 1; README Use case | COMPLETE |
| Specify agent goal | Notebook Section 1; README Use case | COMPLETE |
| Specify target audience | Notebook Section 1; README Use case | COMPLETE |
| Specify available tools | Notebook Sections 1 and 8; `TOOL_SCHEMAS` | COMPLETE |
| Specify constraints and limitations | Notebook Section 1; README Use case | COMPLETE |
| System prompt defines behavior and available tools | `build_system_prompt`; notebook Sections 7, 9, 11 | COMPLETE |
| Tool descriptions embedded in prompt/instructions | `tool_catalog_text()` in `build_system_prompt` | COMPLETE |
| Detect when model suggests an action | `detect_action()` checks structured `tool_calls`; notebook Section 12 | COMPLETE |
| Execute tool and return observation | `execute_tool_call()` and `react_tool_node()`; notebook Section 12 | COMPLETE |
| Continue the loop after observation | `workflow.add_edge("tools", "model")` | COMPLETE |
| Parse and return final answer | `react_model_node`, `route_after_model`, and `run_agent_with_trace` | COMPLETE |
| At least two distinct personas | Friendly Bakery Concierge and Precise Bakery Specialist | COMPLETE |
| Persona tone, vocabulary, response style, and values | `PERSONAS`; notebook Section 5 | COMPLETE |
| Persona strengths and weaknesses | `PERSONAS`; notebook Section 5 | COMPLETE |
| Matching LLM configuration and explanation | `api_config`, `configuration_rationale`; notebook Section 6 | COMPLETE |
| Test both personas | P01-P06 live scored experiment results in notebook Section 19 | COMPLETE |
| With vs without Chain-of-Thought | E03/E04 safe private-reasoning comparison; final answers only | COMPLETE |
| Zero-shot vs few-shot | E01/E02; worked example excluded/included | COMPLETE |
| System prompt vs in-message instructions | E05/E06 | COMPLETE |
| Vary LLM configurations | C01/C02 vary supported reasoning effort and max tokens | COMPLETE |
| Log and compare experiments | `experiment_harness.py`; notebook Section 19 | COMPLETE |
| Dedicated instructions block separate from persona | `BASE_INSTRUCTIONS`, `INSTRUCTIONS`; notebook Section 7 | COMPLETE |
| Instructions are focused | Numbered single-purpose rules in `BASE_INSTRUCTIONS` | COMPLETE |
| Expected inputs and outputs are explicit | Instructions rules 1-2; tool schemas | COMPLETE |
| Tool behavior and constraints are explicit | Instructions rules 3-12; tool catalog | COMPLETE |
| At least one worked example inside instructions | `WORKED_EXAMPLE` included in `INSTRUCTIONS`; notebook Section 8 | COMPLETE |
| Persona and instructions do not conflict | Tone is isolated from factual/tool constraints | COMPLETE |
| Answer all five reflection questions | Notebook Section 20 | COMPLETE |
| Evidence-based reflection conclusions | Scored findings in notebook Sections 19-20 | COMPLETE |

## Project-specific requested requirements

| Requirement | Where implemented | Status |
|---|---|---|
| Preserve Cedar & Crumb scenario | `me/` files unchanged; knowledge loaded by `agent_core.py` | COMPLETE |
| Keep Gemini through Google's OpenAI-compatible endpoint | `GEMINI_BASE_URL`, `create_client()` | COMPLETE |
| Use `GEMINI_API_KEY` from `.env` without exposing it | `create_client()`; `.env.example`; `.gitignore` | COMPLETE |
| Keep configuration in dictionaries | `PERSONAS[*]["api_config"]` | COMPLETE |
| Avoid unsupported sampling claims | Deprecated profiles documented but not sent; README/notebook explanation | COMPLETE |
| `record_customer_interest(email, name, message)` | `agent_core.py` with validation and JSONL logging | COMPLETE |
| `record_feedback(question)` | `agent_core.py` with validation and JSONL logging | COMPLETE |
| Add at most one useful extra tool | One extra: `lookup_business_info(topic)` | COMPLETE |
| Runtime logs in `data/leads.jsonl` and `data/feedback.jsonl` | Tool implementations | COMPLETE |
| Ignore runtime JSONL files | `.gitignore`: `data/*.jsonl` | COMPLETE |
| Short-term conversation memory | `normalize_history()` and `build_initial_messages()`; test and notebook demo | COMPLETE |
| Multi-turn lead flow | Offline memory test plus live dry-run tool call | COMPLETE |
| Persona selector in Gradio | `app.py` dropdown | COMPLETE |
| Gradio shows final answers only | `chat()` calls `run_agent()`; no internal trace returned | COMPLETE |
| New primary C4 notebook | `c4_agent_personas.ipynb` | COMPLETE |
| Preserve prior notebook | `business_agent.ipynb` unchanged | COMPLETE |
| Notebook follows requested 22-section order | `c4_agent_personas.ipynb` Sections 1-22 | COMPLETE |
| Update README for C4 | `README.md` | COMPLETE |
| Add only necessary runtime packages | `requirements.txt` | COMPLETE |
| Safe `.env.example` placeholders | `.env.example` | COMPLETE |
| Ignore caches, venvs, checkpoints, Gradio data | `.gitignore` | COMPLETE |
| Offline syntax/behavior tests provided | `tests/test_agent_core.py` | COMPLETE |
| Run offline tests in this environment | 8/8 passing with Python 3.11 | COMPLETE |
| Run app and live API flows | Gradio stayed running for the startup check; live Gemini calls succeeded | COMPLETE |
| Test normal fact question | Live hours lookup returned correct grounded answer | COMPLETE |
| Test lead flow | Live multi-turn dry-run called `record_customer_interest` | COMPLETE |
| Test feedback | Live dry-run called `record_feedback` for concrete packaging feedback | COMPLETE |
| Test unknown business information | E03/E04 used lookup then feedback without negative inference | COMPLETE |
| Confirm correct tool is called | Live `tools_used` results plus offline fake-client test | COMPLETE |
| Confirm short-term memory | Offline history test and live multi-turn lead flow | COMPLETE |
| Confirm API key is not printed | Static review: key only read into client | COMPLETE |
| Confirm `.env` is ignored | `git check-ignore .env` verification | COMPLETE |
| Confirm JSONL logs are ignored | `git check-ignore data/leads.jsonl data/feedback.jsonl` verification | COMPLETE |
| Do not modify old repository or create a remote | Current repository has no remote; no push/commit performed | COMPLETE |

## Grading criteria map

| Criterion | Points | Evidence | Status |
|---|---:|---|---|
| Use-case summary and write-up | 10 | Notebook Section 1; README Use case | COMPLETE |
| ReAct logic manually implemented | 30 | `AgentState`, model/action nodes, action detection, observation messages, conditional edges, loop guard | COMPLETE |
| Persona design (prompt and configuration) | 20 | `PERSONAS`; notebook Sections 5-6 | COMPLETE |
| Configuration experimentation and comparison | 15 | C01/C02 scored results plus honest compatibility notes | COMPLETE |
| Clear instructions design | 10 | Separate numbered `INSTRUCTIONS` plus embedded worked example | COMPLETE |
| Creativity, clarity, and reflection quality | 15 | Bakery-grounded lookup tool, safe experiment design, scored Section 20 reflection | COMPLETE |

## Manual completion commands

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python experiment_harness.py
python app.py
```

The experiment run and scoring are complete. Re-run the commands only when reproducing
the results on another machine or after changing prompts/configuration.
