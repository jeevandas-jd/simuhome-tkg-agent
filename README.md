# SimuHome TKG Agent

A ReAct-style LLM agent augmented with a **Temporal Knowledge Graph (TKG)** memory layer, evaluated on the [SimuHome](https://arxiv.org/abs/2509.24282) smart home benchmark (ICLR 2026).

The core question: does structured temporal memory change how an LLM agent reasons and acts — and can it do so with fewer tool calls?

---

## What this is

Standard LLM agents in smart home settings have to re-verify device states, room conditions, and time-sensitive constraints from scratch at every step. This project adds a graph memory layer that stores what the agent has already observed — timestamped, entity-linked, and retrievable before each action.

Two agents are compared head-to-head on the same SimuHome episodes:

| Agent | Memory | Description |
|---|---|---|
| Baseline | None | Plain ReAct loop, tools only |
| TKG Agent | Neo4j graph | Retrieves grounded facts before every LLM call |

---

## Early results (5 QT1 episodes)

| Metric | Baseline | TKG Agent |
|---|---|---|
| Success rate | 83% | 83% |
| Avg steps | 5.4 | 3.2 |
| Avg graph hits | — | 3.2 |
| Avg facts written | — | 23.4 |

Same correctness, ~40% fewer tool calls. The TKG agent stops re-verifying facts it already knows.

---

## Architecture

```
SimuHome Simulator (port 8000)
        │
        ▼
  Agent Loop (ReAct)
        │
   ┌────┴────┐
   │         │
Baseline   TKG Agent
           │
     ┌─────┴──────┐
     │            │
  Ingestor    Retriever
     │            │
     └────────────┘
           │
        Neo4j
   (temporal triples)
```

The TKG has three entity types — `room`, `device`, `user` — and stores facts as:

```
(device:bathroom_air_purifier_1, is_on, True, t=2025-08-23 13:57:53)
(device:utility_room_dimmable_light_1, scheduled_action, wf-abc123, valid_from=14:05:00)
(user:resident_1, located_in, room:living_room, t=2025-08-23 13:57:53)
```

---

## Project structure

```
simuhome-tkg-agent/
├── tkg_agent/
│   ├── agent/
│   │   ├── base_agent.py        # Baseline ReAct agent
│   │   ├── tkg_agent.py         # TKG-enhanced agent
│   │   ├── groq_provider.py     # Groq LLM provider
│   │   └── kaggle_provider.py   # Self-hosted Kaggle LLM provider
│   ├── graph/
│   │   └── neo4j_client.py      # Neo4j connection + temporal write/read
│   ├── ingestor/
│   │   └── episode_ingestor.py  # SimuHome obs → temporal triples
│   ├── retrieval/
│   │   └── retriever.py         # Current state + recent history queries
│   ├── eval/
│   │   └── evaluator.py         # Batch eval with resume support
│   └── demo/
│       └── app.py               # Streamlit comparison UI
├── kaggle_scripts/
│   └── kaggle_server.py         # Self-hosted inference server (no rate limits)
├── experiments/
│   └── demo/results.csv         # Evaluation results
├── main.py                      # CLI entrypoint
└── maual_test.py                # Manual debug tests (all 4 blocks)
```

---

## Setup

### Requirements

- Python 3.13+
- Neo4j (local, bolt://localhost:7687)
- [SimuHome](https://github.com/holi-lab/SimuHome) cloned alongside this repo
- Groq API key (free tier) — or Kaggle notebook for self-hosted inference

### Install

```bash
git clone https://github.com/YOUR_USERNAME/simuhome-tkg-agent.git
cd simuhome-tkg-agent
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env   # fill in your keys
```

### Environment variables

```env
GROQ_API_KEY=your_key
GROQ_MODEL=llama-3.3-70b-versatile

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

SIMULATOR_BASE=http://localhost:8000
EPISODE_DIR=../SimuHome/data/benchmark

# Optional — self-hosted Kaggle inference
KAGGLE_LLM_URL=https://abc123.ngrok-free.app/generate
```

---

## Running

### Start SimuHome simulator

```bash
cd ../SimuHome
uv run simuhome server-start   # keeps running in this terminal
```

### Health check

```bash
python main.py health
```

### Run a single episode

```bash
# TKG agent
python main.py run --episode ../SimuHome/data/benchmark/qt1_feasible_seed_23.json

# Baseline
python main.py run --episode ../SimuHome/data/benchmark/qt1_feasible_seed_23.json --baseline
```

### Side-by-side comparison

```bash
python main.py compare --episode ../SimuHome/data/benchmark/qt1_feasible_seed_23.json
```

### Batch evaluation (with resume support)

```bash
# Runs 5 QT1 + 3 QT4 episodes, saves to experiments/demo/results.csv
python -m tkg_agent.eval.evaluator --qt1-n 5 --qt4-n 3 --delay 10

# Resume after a crash — skips already-completed runs automatically
python -m tkg_agent.eval.evaluator --qt1-n 5 --qt4-n 3 --delay 10

# Print summary from existing CSV without re-running
python -m tkg_agent.eval.evaluator --summary
```

### Debug tests

```bash
python maual_test.py              # all 4 blocks
python maual_test.py --block 1   # Neo4j only
python maual_test.py --block 4   # Retriever only
```

---

## Self-hosted inference (no rate limits)

If you hit Groq rate limits, run the model on a Kaggle GPU notebook instead.

1. Open `kaggle_scripts/kaggle_server.py` and paste each cell block into a new Kaggle notebook
2. Run cells 1–5 — the last cell prints your public ngrok URL
3. Add `KAGGLE_LLM_URL=https://...ngrok-free.app/generate` to your `.env`
4. Swap the import in `evaluator.py`:

```python
# was
from tkg_agent.agent.groq_provider import GroqProvider
# now
from tkg_agent.agent.kaggle_provider import KaggleProvider as GroqProvider
```

Recommended model: `Qwen2.5-7B-Instruct` with 4-bit quantization (~3.5 GB VRAM).

---

## Benchmark episodes

SimuHome has 600 episodes across 4 query types:

| Type | Task | Why it's hard |
|---|---|---|
| QT1 | State verification | Agent must check real device state, not hallucinate |
| QT2 | Implicit intent | Infer what the user wants without being told explicitly |
| QT3 | Explicit device control | Execute multi-step control with correct cluster/attribute |
| QT4 | Temporal scheduling | Schedule actions at future simulator times with correct offsets |

This project focuses on QT1 and QT4 — the two types most affected by weak temporal grounding.

---

## How the TKG memory works

On episode start, `EpisodeIngestor.bootstrap()` parses `initial_home_config` and writes every device state, room environment, and user location into Neo4j as timestamped triples.

Before every LLM call, `TKGRetriever` queries:
- Room environment facts (temperature, humidity, illuminance, PM10)
- Device states (is_on, fan_mode, brightness, scheduled actions)
- Recently changed entities since the episode base time

The grounding block is injected into the system message. If the answer is already in the graph, the agent can respond with `finish` immediately without calling any simulator tool.

After every tool observation, `ingest_observation()` writes new facts back into the graph — so the memory grows richer as the episode progresses.

---

## Citation

```bibtex
@inproceedings{seo2026simuhome,
  title={SimuHome: A Temporal- and Environment-Aware Benchmark for Smart Home LLM Agents},
  author={Gyuhyeon Seo and Jungwoo Yang and Junseong Pyo and Nalim Kim and Jonggeun Lee and Yohan Jo},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=LCS1WsGvha}
}
```

---

## License

Research prototype. SimuHome benchmark is licensed under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).
