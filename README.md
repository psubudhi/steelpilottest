# Steel Pilot V2 — Agentic Maintenance Investigation Workspace

Steel Pilot V2 upgrades the previous Steel Pilot V1 build into a cleaner maintenance workflow for the TCM tandem cold rolling mill use case.

## What changed in V2

V2 reorganizes the dashboard around six operational pages:

1. **Plant Command Center** — plant topology + integrated maintenance queue.
2. **Alarm Investigation & RCA** — grouped alarm events, telemetry evidence, physical rules, cascading impact, SOP citations, RCA report.
3. **Live Telemetry Replay** — normalized z-score, percent-change, min-max, raw grouped, and individual signal views to avoid force scale hiding other variables.
4. **Steel Pilot Maintenance Copilot** — concise/detailed answer modes, active alarm context, SOP source expanders, decision evidence, and out-of-domain guardrail.
5. **Operations Logbook & Feedback** — persistent SQLite-backed logbook and feedback memory with auto-filled active alarm context.
6. **Model Health & Evaluation** — graphical drift/feature importance, metric cards, and selectable agent health-check tests.

## Important design notes

- The ML inference code and artifacts remain aligned with the previous ML notebook output.
- Proxy RUL is an urgency estimate, not true run-to-failure life.
- Telemetry is simulated replay from TCM benchmark data, not real SCADA integration yet.
- SOP documents are demo markdown files; production should replace them with plant SOPs/manuals/CMMS logs.
- RAG retrieval now uses FAISS plus local markdown fallback, so newly added SOP files remain discoverable even before re-ingesting the vector index.
- Feedback and logbook data are stored separately in local SQLite so the original TCM data/model artifacts are not modified.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`:

```bash
OPENAI_API_KEY=your_key_here
TCM_MODELLING_ROOT=../tcm_modelling
STEEL_PILOT_SQLITE_DB=./data/runtime/steel_pilot_ops.sqlite
```

## Build SOP/RAG index

```bash
python scripts/create_demo_docs.py
python scripts/ingest_faiss.py
```

If FAISS is not built, Steel Pilot V2 uses a keyword fallback over local markdown docs, but FAISS is recommended for the final demo. Even with FAISS present, local markdown fallback helps surface newly added SOPs/checklists that have not been re-indexed yet.

## Run dashboard

```bash
streamlit run app.py
```

## Recommended demo flow

1. Open **Plant Command Center**.
2. Apply a demo scenario or select top maintenance priority.
3. Bind an alarm.
4. Open **Alarm Investigation & RCA**.
5. Generate concise/detailed RCA.
6. Open **Live Telemetry Replay** and show normalized chart.
7. Ask targeted questions in **Steel Pilot Maintenance Copilot**.
8. Save feedback in **Operations Logbook & Feedback**.
9. Run selected tests in **Model Health & Evaluation**.

## Persistent runtime storage

V2 creates local SQLite storage at:

```text
data/runtime/steel_pilot_ops.sqlite
```

Tables:

- `logbook_events`
- `feedback_events`
- `agent_health_runs`

This means feedback/logbook/eval history remains after closing and reopening the dashboard.
# steelpilottest
