# C4 - Cedar & Crumb Agent Personas and Thinking Mechanisms

This is a separate C4 project built from the earlier Cedar & Crumb business
scenario. It transforms the bakery chatbot into an agent with two configurable
personas, a manually implemented ReAct loop, grounded tool use, prompt/configuration
experiments, short-term memory, and a Gradio interface.

The project uses **LangGraph** because its explicit state, nodes, and conditional
edges make every part of the required ReAct cycle inspectable. It does **not** use
LangChain `AgentExecutor` or another prebuilt ReAct executor.

## Use case

**Scenario:** Cedar & Crumb is a fictional Beirut bakery offering celebration
cakes, pastry boxes, and workshops.

**Agent goal:** Answer verified business questions, guide order interest, capture
qualified leads, and record feedback or unknown questions without inventing facts.

**Target audience:** Potential and existing Cedar & Crumb customers.

**Tools:**

- `lookup_business_info(topic)` retrieves relevant passages from the verified local
  business sources.
- `record_customer_interest(email, name, message)` writes a qualified lead only
  after all required fields are known.
- `record_feedback(question)` records feedback, complaints, suggestions, and
  unanswered business questions.

**Constraints and limitations:** The assistant must use only verified local business
information; it cannot confirm live availability, final custom pricing, or delivery
fees; it cannot guarantee allergen-free products; and it requires a Gemini API key
for model turns. Runtime logs are local JSONL files and are excluded from Git.

## Manual ReAct architecture

The implementation is visible in `agent_core.py` and demonstrated in
`c4_agent_personas.ipynb`:

```text
User message + recent history
            |
            v
      MODEL node
  (private reasoning)
       /        \
 tool call     final text
    |              |
    v              v
  TOOLS node       END
 (Action -> Observation)
    |
    +-------> MODEL node
```

1. LangGraph state stores messages, persona configuration, iteration count, tool
   metadata, and the final answer.
2. The model node asks Gemini for either a structured tool call or final text.
3. `detect_action` checks the returned structured `tool_calls` field.
4. A conditional edge routes tool calls to the action node or final text to `END`.
5. The action node validates and executes each tool, then adds a `tool` observation
   message to state.
6. An edge returns the observation to the model. A six-iteration guard prevents an
   accidental endless loop.
7. The Gradio layer returns only `final_answer`; private reasoning is never shown.

## Personas and model configuration

### Friendly Bakery Concierge

- Warm, conversational, supportive, and customer-oriented.
- Uses plain language and gentle follow-up questions.
- Gemini configuration: `reasoning_effort="low"`, `max_tokens=420`.
- Strength: natural rapport. Tradeoff: less formal for policy-heavy questions.

### Precise Bakery Specialist

- Professional, direct, factual, structured, and low-risk.
- Uses specific language and compact next steps.
- Gemini configuration: `reasoning_effort="minimal"`, `max_tokens=280`.
- Strength: clarity and restraint. Tradeoff: less celebratory or personal.

The conceptual sampling profiles (`temperature=0.7/top_p=0.9` versus
`temperature=0.2/top_p=0.7`) are documented in the persona dictionaries but are
**not sent** to `gemini-3.5-flash-lite`. Current Gemini guidance deprecates custom
`temperature`, `top_p`, and `top_k` for that model family. The runnable comparison
therefore varies supported controls: reasoning effort and maximum output tokens.
Frequency and presence penalties are also omitted rather than claiming unverified
OpenAI-compatible support.

References: [Gemini OpenAI compatibility](https://ai.google.dev/gemini-api/docs/openai)
and [Gemini 3.6 Flash / 3.5 Flash-Lite parameter changes](https://ai.google.dev/gemini-api/docs/generate-content/whats-new-gemini-3.6).

## Prompt experiments

`experiment_harness.py` defines a compact 14-run plan covering all required
comparisons:

- both personas with their matching configurations;
- zero-shot versus few-shot (worked example present only in few-shot);
- without versus with private reasoning;
- system-prompt versus in-message instructions;
- minimal/280-token versus low/420-token supported configurations;
- five representative prompts: hours, custom cake interest, allergens, packaging
  feedback, and an unknown loyalty-card question.

Only final answers, tool names, and safe Action/Observation status metadata are
logged. Hidden reasoning is never recorded. Write tools run in dry-run mode during
experiments, so fake leads and feedback are not added to customer logs. Manual
evaluation columns cover correctness, helpfulness, tone match, hallucination, and
notes. The final 14-case run is scored in notebook Sections 19-20; all reviewed
responses were non-hallucinated.

## Project structure

```text
.
|-- agent_core.py                 # Personas, instructions, tools, manual graph
|-- app.py                        # Gradio chat UI and persona selector
|-- experiment_harness.py         # Reproducible C4 experiment plan/logging
|-- c4_agent_personas.ipynb       # Primary C4 notebook and reflection
|-- business_agent.ipynb          # Preserved earlier notebook; not the C4 deliverable
|-- C4_CHECKLIST.md               # Requirement/rubric compliance map
|-- tests/test_agent_core.py      # Offline ReAct and memory tests
|-- me/about_business.pdf         # Detailed fictional business source
|-- me/business_summary.txt       # Concise verified business source
|-- data/*.jsonl                  # Ignored runtime records and experiment output
|-- requirements.txt
|-- .env.example
`-- .gitignore
```

## Setup

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Place the real key only in `.env`:

```text
GEMINI_API_KEY=your_key_here
OPENAI_MODEL=gemini-3.5-flash-lite
```

The code uses Google's OpenAI-compatible endpoint with the `openai` Python client.
It does not use an OpenAI API key or OpenAI billing. Never commit `.env`.

## Run and test

Run the offline tests first; they use a fake model client and do not require an API
key:

```bash
python -m unittest discover -s tests -v
```

Start the Gradio app:

```bash
python app.py
```

Run the experiment harness (requires `GEMINI_API_KEY`):

```bash
python experiment_harness.py
```

Results are written to `data/c4_experiments.jsonl`, which is ignored by Git. Open
the notebook after a reproduction run to review final answers and revise scores only
if the prompt, model, or configuration changed. A static reviewed summary is already
preserved in the notebook so graders do not depend on ignored runtime data.

Launch the primary notebook with either command:

```bash
jupyter lab c4_agent_personas.ipynb
```

```bash
jupyter notebook c4_agent_personas.ipynb
```

Jupyter itself is intentionally not pinned in `requirements.txt`; use an existing
Jupyter environment or install it separately with `python -m pip install jupyterlab`.

## Privacy and repository boundaries

- `.env`, Python caches, virtual environments, Gradio state, notebook checkpoints,
  and `data/*.jsonl` are ignored.
- Tools never print the API key or raw customer records to the terminal.
- This folder has no configured remote. Do not add, commit, or push to the earlier
  C3 repository.
- The UI never displays private chain-of-thought. Conceptual ReAct labels in the
  notebook describe architecture only.

## Suggested manual acceptance flow

1. Ask each persona about opening hours and compare tone.
2. Ask for a custom cake for 20 people; then reply with a name and email. Confirm
   short-term memory and one `record_customer_interest` call.
3. Submit a packaging suggestion and confirm one `record_feedback` call.
4. Ask about nut-free cakes and verify the allergy warning is grounded.
5. Ask about a loyalty card and verify the agent does not invent a policy; it should
   log the unanswered question.
6. Review `data/leads.jsonl` and `data/feedback.jsonl` locally, then verify they stay
   ignored with `git check-ignore`.
