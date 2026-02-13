# CLAUDE.md — Tool Profiling for LLM Agents (GAIA Experiment)

## Project Overview

**Paper title (working):** "Tool Profiling for LLM Agents: Learning Which Tools Matter"

**Core claim:** We can profile tool value and interactions cheaply via LASSO regression, then use those profiles to intervene at inference time — selecting optimal tool subsets and providing interaction hints — to improve agent performance on unseen tasks.

**Pipeline:** Profile tools (LASSO on coalitions) → Predict useful tools for new tasks → Intervene (subset selection + interaction hints) → Agent performs better with fewer tools.

## Codebase

Built on top of the smolagents GAIA implementation: https://github.com/aymeric-roucher/GAIA

We keep their agent runner, tool definitions, and GAIA evaluation intact. We add our tool profiling and intervention layer on top.

### Repo Structure
```
GAIA-agents/
├── app.py              # Gradio web UI — fetch questions, run agent, submit scores
├── agents.py           # Core Agent class wrapping smolagents CodeAgent
├── model.py            # Model factory + LiteLLM wrapper with retry logic
├── tool.py             # Tool aggregator (returns all tools for the agent)
├── tools/
│   ├── tools.py                    # Vision, YouTube, file, audio, OCR tools
│   ├── final_answer.py             # Answer validation via local Ollama models
│   ├── describe_image_tool.py      # GPT-4V image description (not in default tool list)
│   ├── table_extractor_tool.py     # Excel/CSV extraction (not in default tool list)
│   ├── openai_speech_to_text_tool.py   # Whisper transcription (not in default tool list)
│   └── youtube_transcription_tool.py   # YouTube transcript fetching (not in default tool list)
├── utils/
│   └── logger.py
├── requirements.txt
└── README.md
```

### Required Modifications Before Experiments
1. **Swap model to GPT-4o-mini:** Edit `model.py` factory function to use `openai/gpt-4o-mini` via LiteLLM instead of Gemini
2. **Remove Ollama dependency:** Strip `final_answer.py` Ollama validation — use GAIA ground truth exact-match scoring directly
3. **Remove 2-question limit:** `app.py:88` has `questions_data = questions_data[:2]` — remove this
4. **Add OpenAI API key:** Env vars needed: `OPENAI_API_KEY`, `GEMINI_API_KEY` (for vision/video tools), `HF_TOKEN`

### Additional Tools Available (not in default list)
The `tools/` directory contains extra tools not currently returned by `tool.py`:
- `describe_image_tool.py` — GPT-4V image description
- `table_extractor_tool.py` — Excel/CSV extraction
- `openai_speech_to_text_tool.py` — Whisper transcription
- `youtube_transcription_tool.py` — YouTube transcript fetching

These could expand the toolset beyond 14 if needed.

## Setup

### Agent
- **Backbone LLM:** GPT-4o-mini (cheap, weak enough to need tools)
- **Framework:** smolagents
- **Tool-internal models:** Gemini (for vision_tool, ask_youtube_video) — these are black boxes to the agent

### Benchmark
- **GAIA validation set:** 165 questions, 3 difficulty levels
- **Scoring:** Binary exact-match correctness (answer is right or wrong)
- **No semantic similarity** — correctness only

### Tools (14, excluding FinalAnswerTool)
| Tool | Category | Description |
|------|----------|-------------|
| DuckDuckGoSearchTool | Search | Web search |
| WikipediaSearchTool | Search | Wikipedia lookup |
| VisitWebpageTool | Web | Read webpage content |
| PythonInterpreterTool | Code | Execute Python code |
| vision_tool | Vision | Analyze images via Gemini |
| extract_text_via_ocr | Vision | OCR via Tesseract |
| youtube_frames_to_images | Video | Sample video frames |
| ask_youtube_video | Video | QA on video via Gemini |
| transcribe_youtube | Audio | Transcribe YouTube via Whisper |
| audio_to_text | Audio | Transcribe audio files |
| read_text_file | File | Read plain text files |
| file_from_url | File | Download files from URLs |
| summarize_csv_data | File | Summarize CSV files |
| summarize_excel_data | File | Summarize Excel files |

### Expected interactions
- **Synergies:** DuckDuckGo + VisitWebpage (search finds URL, visit reads it), PythonInterpreter + summarize_csv (code can process CSV output)
- **Redundancies:** DuckDuckGo ↔ Wikipedia (both search), summarize_csv ↔ summarize_excel (similar), vision_tool ↔ OCR (overlapping)

## Immediate Next Steps (in order)

1. Swap model to GPT-4o-mini in `model.py`
2. Remove Ollama validation dependency
3. Remove `questions_data[:2]` limit
4. Run 5-10 GAIA questions with all tools → verify it works, estimate cost per question
5. Based on cost estimate, decide coalition sampling strategy
6. Begin Phase 2 coalition evaluations

## Experiment Phases

### Phase 1: Baseline
Run the agent with all 14 tools on all 165 GAIA questions. Record accuracy per question. This is the baseline.

### Phase 2: Coalition Evaluation
Sample coalitions (tool subsets) and run the agent on GAIA questions for each.

**Coalition sampling strategy:**
- All 14 singletons
- All 14 leave-one-out (13-tool subsets)
- Empty set (no tools)
- Full set (all 14)
- Random subsets at various sizes (aim for ~200-300 total coalitions)

**For each coalition × question:** Record binary correctness (1/0).

**Cost management:**
- We do NOT run all 165 questions per coalition — too expensive.
- Option A: Select ~40-50 representative GAIA questions spanning all 3 levels.
- Option B: Run all 165 questions but on ~100-150 coalitions only.
- Decision depends on budget. Figure out API costs first.

### Phase 3: LASSO Regression

**Option A — Per-question LASSO:**
- For each question, fit LASSO: correctness ~ tool indicators + interactions
- Problem: binary outcome per coalition per question, noisy

**Option B — Aggregate LASSO (start here):**
- Each row = one coalition
- y = accuracy of that coalition across all questions (fraction correct)
- X = 14 main effects + 91 pairwise interactions = 105 features
- Fit LASSO: accuracy ~ β₀ + Σ βᵢxᵢ + Σ βᵢⱼxᵢxⱼ
- Need ~200+ coalitions for stable fit

**Option C — Task-conditional LASSO (stretch):**
- Add question features (difficulty level, task type, attached file presence, etc.)
- Predict per-question correctness from (coalition features × question features)
- Most ambitious, strongest transfer story

### Phase 4: Intervention Experiments

Three conditions, same agent, same questions:

1. **Baseline:** All 14 tools, no hints
2. **Subset selection:** Remove tools with βᵢ ≈ 0, keep top-k by LASSO coefficient
3. **Subset + interaction hints:** Remove low-β tools AND add to system prompt:
   ```
   Tool synergies (use together when possible):
   - [tool_a] + [tool_b]: [reason]

   Tool redundancies (avoid using both):
   - [tool_c] + [tool_d]: [reason]
   ```

**Success criterion:** Condition 2 ≥ Condition 1 (fewer tools, same or better performance). Condition 3 > Condition 2 (interaction hints help further).

### Phase 5: Transfer

Split GAIA 165 questions into train (110) and test (55).

1. Profile tools on train split (run coalitions, fit LASSO)
2. Select tool subsets based on train-split LASSO coefficients
3. Evaluate on test split with selected subsets
4. Compare: LASSO-selected tools vs all tools vs random selection

**Stretch goal:** Profile on GAIA, transfer to a different benchmark (MINT, τ-bench).

## Success Criteria

| Metric | Target |
|--------|--------|
| LASSO R² (aggregate) | > 0.7 |
| Subset selection vs baseline | Accuracy matches or improves with ≤ 8 tools |
| Subset + hints vs subset only | Accuracy improves |
| Transfer (train→test split) | LASSO selection beats random at k ≤ 8 |
| Distractor removal | Removing low-β tools doesn't hurt (or helps) |

## What We Add to the Repo

Minimal additions — do NOT restructure their code:
```
tool_profiling/
  run_coalitions.py      # Run agent with tool subsets, log results
  lasso_valuation.py     # Fit LASSO, extract coefficients and interactions
  intervene.py           # Modify system prompt with subset + hints
  evaluate_transfer.py   # Train/test split evaluation
  figures.py             # Generate paper figures
results/
  coalitions.jsonl       # Raw coalition evaluation data
  lasso_results.json     # Fitted coefficients
  intervention_results.json
```

## Key Risks

1. **Cost:** 200 coalitions × 50 questions × multiple LLM calls = expensive. Budget carefully.
2. **Distractors might not hurt:** If agent ignores irrelevant tools, subset selection shows no gain.
3. **LASSO might not fit:** Binary outcomes + 105 features could be noisy. Need enough coalitions.
4. **Agent stochasticity:** Use temperature=0 for reproducibility.

## Technical Notes

- temperature=0 for all runs
- Set token/step limit per question to control costs
- Log everything: coalition, question_id, tools_available, answer, correct, tokens_used, time
- API keys needed: OpenAI (GPT-4o-mini), Google (Gemini for vision/video tools)

## Prior Work

- **AgentSHAP (Horovicz, 2025):** First tool attribution paper. Shapley on API-Bank, 8 tools, semantic similarity metric, trivial tasks, no interactions, no intervention.
- **Datamodels (Ilyas et al., 2022):** Methodological inspiration. Linear models predict neural net behavior from training data subsets.
- **Our contribution beyond AgentSHAP:** Correctness-based scoring, pairwise interactions, sample efficiency, and critically — using profiles to actively improve agent performance via intervention.