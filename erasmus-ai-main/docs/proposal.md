# Group Project Proposal: Erasmus UniMatch

Targeting European CS Mobilities & Study-Path Alignments - Prototype for UPJŠ -
Department of Computer Science

*Adriana Bujňáková, Terézia Kurimská, Terézia Kurucová, Faina Petrenko*

### Problem

Choosing an Erasmus+ or exchange partner university is a high-stakes, stressful
decision for Computer Science students. There are approximately 26 European
partner universities connected to our department, each offering vastly
different academic profiles (e.g., pure theory vs. practical software
engineering vs. AI/Data Science) and distinct socio-cultural environments
(cost of living, climate, student activities, and industry opportunities).
Currently, students must manually cross-reference outdated university
websites, faculty PDF agreements, and random city guides. This results in
"decision paralysis" or, worse, students matching with a city or study plan
that does not fit their academic interests or lifestyle expectations, or they
end up not going at all.

### Why Generative AI

This problem cannot be solved effectively with classic software (like SQL
databases or conditional if/else filters). Academic curriculum descriptions
and lifestyle preferences are unstructured, natural-language datasets. A
generative model acts as a semantic matching engine:

1. It reads raw text files describing a city's vibe and a department's
   syllabus, mapping them to a student's highly subjective prompt (e.g., "I
   want a warm city with active hiking options, where I can focus on
   hands-on web development").
2. It synthesizes complex comparative justifications, explaining why a
   specific program matches their lifestyle and curriculum needs, which
   standard search queries cannot do.

### Techniques Used

- **Local LLM Host (LM Studio):** A highly efficient, local model runs
  completely offline via LM Studio's server-mode. All student data remains
  local, respecting privacy guidelines.
- **Multi-File Local RAG (Retriever):** A custom directory-scanning retriever
  dynamically parses the directory for all 26 university profile files
  (partner_*.txt). Each file is divided into (1) city lifestyle/vibe and
  (2) active computer science pathways.
- **LangGraph Orchestration (StateGraph & Grading Loop):** An agentic routing
  loop instead of a rigid, linear pipeline:
  1. **The Grader Node:** Reviews the student's input. If too vague (e.g.,
     "find me a school"), routes to a **Clarifier Node** for follow-up
     questions.
  2. **The Matcher Node:** If there is sufficient detail, performs semantic
     pairing between the student's interests and the best partner
     university.
- **Deterministic Guardrails:** After the model recommends a university, a
  deterministic Python-level check cross-references the suggested school
  name against the headers of the 26 loaded files. If the LLM invents a
  non-existent university/program, the guardrail triggers a safety
  correction.

### Risks & Plan B

1. **Risk 1: Local Model Latency or Hardware Limits.** Running a local LLM
   through LM Studio on standard laptops can lead to slow response times
   during live grading loops.
   - *Plan B:* Use a highly quantized, smaller model (such as
     Llama-3-8B-Instruct-Q4_K_M or Phi-3-mini-4K-instruct) and optimize
     prompt lengths to keep generation times under 10 seconds. Record a
     backup video of a successful run.
2. **Risk 2: Retrieval Dilution (Finding the wrong university).** When
   searching across 26 different university text files, the model might
   fetch irrelevant files or mix up city details.
   - *Plan B:* Implement a metadata pre-filter stage. Before running the
     full semantic RAG, a quick keywords-based pass or strict file-indexing
     layout narrows the retrieval context down to the top 3 candidate
     university text files.
