"""Everything for the UniMatch agent lives in this one file: state, the local
LM Studio client, the partner-file retriever, the four LangGraph nodes
(Grader -> Clarifier | Matcher -> Guardrail), and the compiled graph itself.

Flow:
    Grader decides if the student's prompt has enough detail.
      -> not enough  => Clarifier asks a follow-up question.
      -> enough      => Matcher picks the best-fitting partner university,
                         then Guardrail double-checks the LLM didn't invent
                         a university that isn't in our loaded files.

Each node is wrapped in @cl.step so Chainlit renders it as a visible step in
the chat trace -- that's what makes the routing logic visible during a demo.

If LM Studio isn't running (or a call fails for any reason), call_llm()
returns None and every node below falls back to a cheap heuristic so the app
still works offline.
"""

import os
import re
from pathlib import Path
from typing import Optional, TypedDict

import chainlit as cl
from langgraph.graph import END, StateGraph
from openai import OpenAI

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class GraphState(TypedDict, total=False):
    """State passed between LangGraph nodes for a single turn."""

    history: list[dict]  # [{"role": "user"|"assistant", "content": str}]
    user_input: str

    is_clear: bool
    clarifying_question: Optional[str]

    candidates: list[dict]  # partner records considered by the matcher
    matched_university: Optional[str]
    justification: Optional[str]

    guardrail_passed: bool
    final_answer: str


# ---------------------------------------------------------------------------
# Local LLM client (LM Studio, OpenAI-compatible)
# ---------------------------------------------------------------------------

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            # Nastavené na tvoju lokálnu adresu LM Studio
            base_url=os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1"),
            api_key="lm-studio",  # nepoužíva sa, ale knižnica vyžaduje nejakú hodnotu
        )
    return _client


NO_TOOL_CALL_INSTRUCTION = (
    " Respond with plain natural-language text only. Do not call a function "
    "or tool, and do not output JSON."
)


def _looks_like_tool_call(text: str) -> bool:
    """Some local models (e.g. Llama-3.1-Instruct) sometimes emit a fake
    function-call JSON blob out of habit, even with no tools offered. Treat
    that as a failed generation rather than showing raw JSON to the student."""
    stripped = text.strip()
    return stripped.startswith("{") and '"name"' in stripped and '"parameters"' in stripped


def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str | None:
    """Single-turn chat completion against the local model. Returns None on
    failure, or if the model produced a tool-call-shaped response instead of
    the requested plain text (see _looks_like_tool_call)."""
    try:
        response = _get_client().chat.completions.create(
            # Predvolený model nastavený na meta-llama-3-8b-instruct
            model=os.getenv("LM_STUDIO_MODEL", "meta-llama-3-8b-instruct"),
            messages=[
                {"role": "system", "content": system_prompt + NO_TOOL_CALL_INSTRUCTION},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        content = response.choices[0].message.content
        if content is None or _looks_like_tool_call(content):
            return None
        return content
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Retriever: loads data/partner_universities/*.txt
# ---------------------------------------------------------------------------

SECTION_SPLIT = re.compile(r"SECTION\s*2\s*:.*", re.IGNORECASE)
UNIVERSITY_LINE = re.compile(r"^UNIVERSITY:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
CITY_LINE = re.compile(r"^COUNTRY & CITY:\s*(.+)$", re.IGNORECASE | re.MULTILINE)


def load_partner_files(data_dir: str | None = None) -> list[dict]:
    """Parse every partner_*.txt file into a structured record.

    TODO (RAG team): swap match_candidates() below for real semantic search
    (embeddings + vector store, or an LLM re-ranking pass) once you've settled
    on an approach -- the parsed fields here (university/city/sections) are
    ready to feed into that.
    """
    data_dir = data_dir or os.getenv("DATA_DIR", "data/partner_universities")
    records = []

    for path in sorted(Path(data_dir).glob("*.txt")):
        text = path.read_text(encoding="utf-8")

        university_match = UNIVERSITY_LINE.search(text)
        city_match = CITY_LINE.search(text)
        parts = SECTION_SPLIT.split(text, maxsplit=1)

        records.append({
            "file": path.name,
            "university": university_match.group(1).strip() if university_match else path.stem,
            "city": city_match.group(1).strip() if city_match else "",
            "city_life_section": parts[0].strip() if parts else text.strip(),
            "study_program_section": parts[1].strip() if len(parts) > 1 else "",
            "raw_text": text,
        })

    return records


STOP_WORDS = {
    "about", "and", "are", "building", "can", "could", "erasmus", "for",
    "from", "have", "like", "love", "me", "my", "please", "recommend",
    "student", "that", "the", "this", "university", "want", "with", "you",
}


def normalized_terms(text: str) -> list[str]:
    """Lowercase, drop stopwords/short words, and lightly stem plurals/gerunds
    so e.g. "hiking" in a prompt matches "hikes" or "hike" in a profile."""
    terms: list[str] = []
    for token in re.findall(r"[a-zA-ZÀ-ž]+", text.casefold()):
        if token in STOP_WORDS or len(token) < 3:
            continue
        if token.endswith("ing") and len(token) > 5:
            token = token[:-3]
        elif token.endswith("es") and len(token) > 5:
            token = token[:-2]
        elif token.endswith("s") and len(token) > 4:
            token = token[:-1]
        terms.append(token)
    return terms


def match_candidates(user_input: str, records: list[dict], top_k: int = 3) -> list[dict]:
    """Relevance ranking: term-frequency overlap between the prompt and each
    file, using stemmed/stopword-filtered terms so matches aren't thrown off
    by exact-word mismatches (e.g. "hiking" vs "hikes")."""
    query_terms = set(normalized_terms(user_input))

    def score(record: dict) -> int:
        document_terms = normalized_terms(record["raw_text"])
        return sum(document_terms.count(term) for term in query_terms)

    ranked = sorted(records, key=score, reverse=True)
    return ranked[:top_k]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

MIN_WORDS_FOR_CLEAR_PROMPT = 6

# Keyword signals used by the heuristic fallback / override in grader_node.
# These are intentionally broad -- the goal isn't precision, it's catching
# "the student clearly gave both a subject and a lifestyle preference" even
# when the local LLM's CLEAR/VAGUE judgement is unreliable.
SUBJECT_SIGNAL_WORDS = [
    "study", "studies", "ai", "artificial intelligence", "software",
    "computer science", "data", "engineering", "machine learning",
    "programming", "coding", "cybersecurity", "robotics", "math",
    "business", "economics", "design",
]
LIFESTYLE_SIGNAL_WORDS = [
    "city", "beach", "sunny", "climate", "weather", "budget", "country",
    "coast", "mountain", "capital", "small town", "big city", "warm",
    "cold", "affordable", "cheap", "language",
]


def _has_any(text: str, words: list[str]) -> bool:
    lowered = text.lower()
    return any(w in lowered for w in words)


@cl.step(name="Grader", type="tool")
async def grader_node(state: GraphState) -> GraphState:
    user_input = state["user_input"]

    llm_verdict = call_llm(
        system_prompt=(
            "You are an academic screening coordinator. A student's request "
            "has ENOUGH detail if it mentions AT LEAST (1) a field of study "
            "or academic interest, AND (2) some preference about the "
            "environment, lifestyle, climate, or location. It does NOT need "
            "exact country names, a budget, or a language level -- those are "
            "nice-to-haves, not requirements.\n\n"
            "Example of CLEAR: 'I want to study AI in a sunny city with a "
            "beach.' (has a subject + a lifestyle preference)\n"
            "Example of CLEAR: 'I'm interested in software engineering, "
            "ideally somewhere affordable with practical courses.'\n"
            "Example of VAGUE: 'I want to study something good.' (no subject, "
            "no lifestyle preference)\n"
            "Example of VAGUE: 'Where should I go?' (nothing specified)\n\n"
            "Answer strictly with ONLY 'CLEAR' or 'VAGUE'."
        ),
        user_prompt=user_input,
        temperature=0.0,
    )
    print(f"[GRADER DEBUG] raw verdict: {llm_verdict!r}")

    # Heuristic signal: does the prompt itself contain both a subject and a
    # lifestyle/location cue? If so, treat it as clear regardless of what the
    # (unreliable, 8B-scale) local LLM decided -- a false "VAGUE" verdict from
    # the model should never override an objectively detailed prompt.
    heuristic_clear = (
        _has_any(user_input, SUBJECT_SIGNAL_WORDS)
        and _has_any(user_input, LIFESTYLE_SIGNAL_WORDS)
        and len(user_input.split()) >= MIN_WORDS_FOR_CLEAR_PROMPT
    )
    print(f"[GRADER DEBUG] heuristic_clear: {heuristic_clear}")

    if llm_verdict is not None:
        llm_says_clear = "CLEAR" in llm_verdict.upper()
        is_clear = llm_says_clear or heuristic_clear
    else:
        is_clear = heuristic_clear

    print(f"[GRADER DEBUG] is_clear: {is_clear}")

    state["is_clear"] = is_clear
    if not is_clear:
        state["clarifying_question"] = (
            "Could you tell me a bit more? For example: what should you study "
            "(theory, practical software dev, AI/data science...) and what kind "
            "of city/climate/lifestyle are you hoping for?"
        )
    return state


@cl.step(name="Clarifier", type="tool")
async def clarifier_node(state: GraphState) -> GraphState:
    user_input = state["user_input"]

    llm_answer = call_llm(
        system_prompt=(
            "A student's request to find a partner university was too vague to "
            "make a match. Write a short, friendly response asking them to specify "
            "(1) what computer science subjects they like most, and (2) what kind "
            "of environment or weather they prefer."
        ),
        user_prompt=user_input,
        temperature=0.7,
    )

    state["final_answer"] = llm_answer or state["clarifying_question"]
    return state


@cl.step(name="Matcher", type="tool")
async def matcher_node(state: GraphState) -> GraphState:
    records = load_partner_files()
    candidates = match_candidates(state["user_input"], records, top_k=3)
    state["candidates"] = candidates
    print(f"[MATCHER DEBUG] candidates: {[c['university'] for c in candidates]}")

    if not candidates:
        state["matched_university"] = None
        state["justification"] = (
            "No partner university files are loaded yet in data/partner_universities/."
        )
        return state

    context = "\n\n--- NEXT CANDIDATE ---\n\n".join(c["raw_text"] for c in candidates)
    llm_answer = call_llm(
        system_prompt=(
            "You are an expert academic advisor. Use the retrieved partner "
            "profiles below as your only evidence -- do not use outside "
            "knowledge. Recommend exactly one university from the evidence, "
            "matching both the student's academic interest and lifestyle "
            "preferences. On the FIRST LINE, output ONLY the exact UNIVERSITY "
            "name copied verbatim from the evidence -- no extra words, no "
            "'I recommend', no markdown, no punctuation added. Then on the "
            "following lines explain both the academic match and the "
            "city/lifestyle match. Never recommend a university absent from "
            "the evidence -- if there is no reasonable match, say that "
            "plainly instead of guessing."
        ),
        user_prompt=(
            f"STUDENT PROFILE:\n{state['user_input']}\n\n"
            f"RETRIEVED OFFICIAL PARTNER PROFILES:\n{context}"
        ),
        temperature=0.4,
    )
    print(f"[MATCHER DEBUG] llm raw answer: {llm_answer!r}")

    if llm_answer is not None:
        first_line, *rest = llm_answer.strip().splitlines()
        state["matched_university"] = first_line.strip()
        state["justification"] = "\n".join(rest).strip()
    else:
        # heuristic fallback: top-ranked candidate, canned justification
        top = candidates[0]
        state["matched_university"] = top["university"]
        state["justification"] = (
            f"Based on keyword overlap with your prompt, {top['university']} "
            f"({top['city']}) looks like the closest match in our current data."
        )

    print(f"[MATCHER DEBUG] matched_university: {state['matched_university']!r}")

    return state


def _normalize_for_match(text: str) -> str:
    """Strip markdown/punctuation noise and lowercase, so comparisons aren't
    thrown off by the LLM wrapping a name in "**...**", quotes, or a sentence
    like "I recommend X."."""
    stripped = re.sub(r"[*_`\"'.,:;!]", "", text)
    return re.sub(r"\s+", " ", stripped).strip().lower()


_NAME_STOPWORDS = {
    "university", "universite", "universidad", "univerzita", "uniwersytet",
    "the", "of", "and", "in", "at", "de", "la", "le", "les", "des", "for",
}


def _significant_words(name: str) -> set[str]:
    """Extract the meaningful words from a university name for fuzzy
    matching -- drops generic words like "university"/"of" and short
    tokens/abbreviations that aren't reliable signals on their own."""
    normalized = _normalize_for_match(name)
    words = re.findall(r"[a-zà-ž0-9]+", normalized)
    return {w for w in words if len(w) > 2 and w not in _NAME_STOPWORDS}


def _best_matching_university(matched_text: str, known_universities: set[str]) -> Optional[str]:
    """Find which known university (if any) the LLM's answer refers to, using
    word overlap rather than a strict substring/exact match. Small local
    models often reorder words or drop trailing abbreviations (e.g. answering
    "Grenoble INP - Ensimag (Universite Grenoble Alpes)" for the known name
    "Universite Grenoble Alpes (Grenoble INP - Ensimag, UGA)") -- an exact or
    substring match would wrongly reject that as unverified even though it's
    clearly the same university.
    """
    if not matched_text:
        return None

    matched_words = _significant_words(matched_text)
    if not matched_words:
        return None

    best_name: Optional[str] = None
    best_ratio = 0.0
    for known in known_universities:
        known_words = _significant_words(known)
        if not known_words:
            continue
        overlap = len(known_words & matched_words)
        if overlap == 0:
            continue
        # Score both directions: overlap relative to the known name (handles
        # reordering/dropped trailing words) AND relative to what the LLM
        # actually said (handles the LLM answering with just a short
        # abbreviation, e.g. "VILNIUS TECH" for "Vilnius Gediminas Technical
        # University (VILNIUS TECH)" -- that's 2/4 of the known name's words
        # but 2/2 of what the LLM wrote, i.e. a complete, correct match).
        ratio_vs_known = overlap / len(known_words)
        ratio_vs_matched = overlap / len(matched_words)
        score = max(ratio_vs_known, ratio_vs_matched)
        if score > best_ratio:
            best_ratio = score
            best_name = known

    # Require most of the words on at least one side to line up -- generous
    # enough to survive reordering, dropped abbreviations, or an
    # abbreviation-only answer, but still rejects a genuinely different or
    # hallucinated university (which won't share enough words either way).
    if best_ratio >= 0.6:
        return best_name
    return None


@cl.step(name="Guardrail", type="tool")
async def guardrail_node(state: GraphState) -> GraphState:
    """Deterministic check: the matched university must be one we actually loaded,
    so a hallucinated name never reaches the student.

    The LLM is asked to output just the university name on the first line,
    but small local models sometimes wrap it in a sentence, reorder the
    words, or drop a trailing abbreviation anyway. Rather than requiring an
    exact or substring match (which breaks on that noise), we use fuzzy
    word-overlap matching: a known university counts as confirmed if most of
    its distinctive words appear anywhere in what the LLM returned.
    """
    known_universities = {c["university"] for c in state.get("candidates", [])}
    matched = state.get("matched_university")
    print(f"[GUARDRAIL DEBUG] known_universities: {known_universities}")
    print(f"[GUARDRAIL DEBUG] matched: {matched!r}")

    matched_university_name = _best_matching_university(matched or "", known_universities)

    passed = matched_university_name is not None
    state["guardrail_passed"] = passed
    print(f"[GUARDRAIL DEBUG] resolved match: {matched_university_name!r}")
    print(f"[GUARDRAIL DEBUG] passed: {passed}")

    if passed:
        state["final_answer"] = f"**{matched_university_name}**\n\n{state['justification']}"
    else:
        state["final_answer"] = (
            "I couldn't verify that recommendation against our loaded partner "
            "university files, so I won't guess. Please rephrase or add more "
            "detail so I can retry."
        )

    return state


def route_after_grader(state: GraphState) -> str:
    return "matcher" if state["is_clear"] else "clarifier"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("grader", grader_node)
    workflow.add_node("clarifier", clarifier_node)
    workflow.add_node("matcher", matcher_node)
    workflow.add_node("guardrail", guardrail_node)

    workflow.set_entry_point("grader")
    workflow.add_conditional_edges(
        "grader",
        route_after_grader,
        {"clarifier": "clarifier", "matcher": "matcher"},
    )
    workflow.add_edge("clarifier", END)
    workflow.add_edge("matcher", "guardrail")
    workflow.add_edge("guardrail", END)

    return workflow.compile()