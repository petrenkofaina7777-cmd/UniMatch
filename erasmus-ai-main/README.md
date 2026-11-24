# UniMatch – Erasmus CS Mobility Matcher

Prototype built for UPJŠ Dept. of Computer Science: a chat assistant that matches
students to one of ~26 European partner universities based on a natural-language
description of their academic interests and lifestyle preferences.

Built in a single 8-hour session for a summer school project. See the original
proposal: `docs/proposal.md` (paste it in there if you want it versioned).

## Architecture

```
User prompt
   │
   ▼
[Grader Node]  -- vague? -->  [Clarifier Node] -- asks a follow-up, ends turn
   │ clear enough
   ▼
[Matcher Node]  -- uses the retriever over data/partner_universities/*.txt
   │
   ▼
[Guardrail Node] -- checks the recommended university actually exists
   │                  in the loaded files; corrects if the LLM hallucinated
   ▼
Answer shown in chat, with each step visible as a Chainlit "step" trace
```

- **UI / orchestration (this repo's scaffold):** [Chainlit](https://docs.chainlit.io/)
- **LLM:** local model served by [LM Studio](https://lmstudio.ai/) in server mode
  (OpenAI-compatible endpoint, default `http://localhost:1234/v1`)
- **Agent graph:** [LangGraph](https://langchain-ai.github.io/langgraph/) `StateGraph`
- **Retrieval:** custom directory-scanning retriever over the 26 `partner_*.txt` files

## Folder structure & who edits what

```
app.py                        Chainlit entrypoint — chat lifecycle, calls the graph
chainlit.md                   Welcome screen shown when the chat opens
graph/state.py                Shared state schema passed between nodes
graph/nodes.py                Grader / Clarifier / Matcher / Guardrail nodes  <- RAG team edits here
graph/build_graph.py          Wires the nodes into the LangGraph StateGraph
retriever/loader.py           Loads & parses partner_*.txt into structured records <- RAG team edits here
llm/client.py                 Thin wrapper around the LM Studio OpenAI-compatible client
data/partner_universities/    Drop the 26 partner_*.txt files here (1 example included)
```

Everything the UI needs from the RAG/agent side goes through
`graph/build_graph.py::build_graph()`, which returns a compiled LangGraph graph with
`.ainvoke(state)`. **You don't need to touch `app.py` or Chainlit at all** — just fill
in the logic inside `graph/nodes.py` and `retriever/loader.py` where you see `# TODO`.

The scaffold currently runs end-to-end with placeholder/heuristic logic (no LLM
required) so the chat UI is demoable immediately. Swap in real LLM calls and
semantic retrieval as you build them.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # on Windows
pip install -r requirements.txt
copy .env.example .env          # then edit if your LM Studio port/model differs
```

Start LM Studio, load a small quantized model (e.g. `Phi-3-mini-4k-instruct` or
`Llama-3-8B-Instruct-Q4_K_M` per the proposal's Plan B), and start the local server
(LM Studio → Developer tab → Start Server). Confirm the port matches `.env`.

## Run

```bash
chainlit run app.py -w
```

Opens the chat UI at `http://localhost:8000`. `-w` auto-reloads on file changes.

## Adding the 26 partner university files

Drop each file in `data/partner_universities/` named `partner_<COUNTRY>_<CITY>.txt`
(e.g. `partner_SK_KOSICE02.txt`, already included as a real example). Follow the same
two-section format as the example file — the loader in `retriever/loader.py` splits
on the `SECTION 1` / `SECTION 2` headers, so keep those headers verbatim.

## Known scope cuts (8-hour budget)

- The clarifier loop is single-shot per turn: if the grader flags a prompt as too
  vague, the assistant asks one follow-up and waits for the next message, rather than
  using LangGraph interrupts for a true sub-loop. Good enough for a demo; call it out
  as a "future work" line if asked.
- `matcher_node` ships with a keyword-overlap fallback so the app runs without LM
  Studio connected. Wire in the real LLM call (`llm/client.py::call_llm`) for the
  actual semantic matching described in the proposal.
- No persistence/auth — this is a single-session local prototype, matching the
  proposal's "runs offline on a laptop" framing.

## CV blurb (draft)

> Built a local-first RAG agent (LangGraph + LM Studio + Chainlit) that matches
> students to Erasmus partner universities from unstructured program/city data,
> with a deterministic guardrail node to prevent hallucinated recommendations.
