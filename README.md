# Multi-LLM-Agent Business Process Simulation

A discrete event simulation (DES) of a bank's loan application process in
which the process participants are LLM agents. Each agent makes routing
decisions, writes its reasoning, and judges how complex each case is. The
engine handles time: queues, shared workers, office hours, and when each
activity finishes. The output is an XES event log that can be compared
against the real **BPI Challenge 2017** log.

The repository also contains a rule-based baseline that uses the same tools
and routing, and scripts to evaluate and compare the resulting logs.

---

## How it works

### Roles and information asymmetry

| Role | Persona | Tools | Sees |
|---|---|---|---|
| Junior Clerk | Carlos | `IntakeApplication`, `CheckDocuments`, `ForwardCase`, `ReturnApplicationEarly` | Application form only (amount, goal, type) |
| Senior Clerk | Ana | `CheckCreditScore`, `ValidateApplication`, `RequestAdditionalInfo`, `EscalateCase` | Application, plus credit bureau data once `CheckCreditScore` has run |
| Credit Officer | Dr. Mueller | `AssessRisk`, `ApproveLoan`, `RejectLoan` | Everything, including all prior agents' notes |

On each turn an agent receives a case context built for its role
([base_agent.py](src/agents/base_agent.py)) and calls exactly one Pydantic
tool ([schemas.py](src/tools/schemas.py)). The chosen tool sets where the
case goes next. The Senior Clerk can send a case back to the Junior Clerk
for rework.

### Discrete event engine

[simulation_engine.py](src/queue/simulation_engine.py) is domain-agnostic.
Everything specific to the loan process comes in through a
`ProcessDefinition` ([definition.py](src/process/definition.py)):

- **Arrivals.** Cases are replayed at their real BPIC arrival times
  (default), or generated as a Poisson process.
- **Queues.** Each role has one FIFO queue, shared by all of that role's
  workers (an M/M/c-style queue).
- **Shifts.** Work that runs past the end of a shift is paused overnight and
  resumes at the next shift start. Weekends are days off.
- **Durations.** By default the agent rates each case on a 1–5 complexity
  scale, and the engine multiplies a per-tool reference duration by that
  rating ([duration_reference.py](src/tools/duration_reference.py)).
  The agent also gives an unanchored estimate of the typical duration. That
  estimate is recorded for analysis but never moves the clock.
- **Silent tools.** `ForwardCase`, `EscalateCase` and `CheckCreditScore`
  move a case forward without writing an event, because BPIC 2017 has no
  equivalent activity for them.
- **Activity labels.** Tool names are mapped to BPIC 2017 labels (for
  example, `ValidateApplication` becomes `W_Validate application`), so both
  logs use the same activity names.

### Rule-based baseline

[rule_based.py](src/agents/rule_based.py) replaces each LLM agent with fixed
rules. The baseline uses the same tools, routing and activity labels
([loan_application_rules.py](src/process/loan_application_rules.py)). If
both versions run on the same cases with the same seed, any difference
between their logs comes from how each one chose its actions.

---

## Repository layout

```
run_des.py               Run a simulation
evaluate.py              Compare a simulated log with the reference log
compare_logs.py          Compare two simulated logs (e.g. LLM vs rules)
recover.py               Rebuild an event log from an interrupted run's checkpoint

src/
  agents/                LLM personas (junior/senior clerk, credit officer) and the rule baseline
  process/               ProcessDefinition type and the loan process instantiation
  queue/                 DES engine, event queue, worker pool, step executor
  tools/                 Pydantic tool schemas and duration anchors
  clock/                 Log-normal duration sampling (the "distribution" baseline and fallback)
  observer/              Converts agent actions to XES entries
  evaluation/            Log loading and filtering, plus the log-distance metrics
  state.py               Case state and record types
  simulation_controller.py   Loads BPIC cases or generates synthetic ones
  reasoning_export.py    Exports the agents' reasoning as a coding frame

results/                 Outputs of the runs reported in the thesis
```

`src/graph/simulation_graph.py` and `src/queue/process_graph.py` hold the
earlier LangGraph-based design. The engine does not use them.

---

## Setup

Requires Python 3.10 or later.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate elsewhere)
pip install -r requirements.txt
pip install log-distance-measures   # needed by evaluate.py
```

Copy `env.example.txt` to `.env` and set the model:

- **OpenAI.** Any model name starting with `gpt-` (for example
  `gpt-4o-mini`, the default) uses the OpenAI API. Set `OPENAI_API_KEY`.
- **Ollama.** Any other model name (for example `llama3.1`) goes to the
  Ollama server at `OLLAMA_BASE_URL`. Start the server with `ollama serve`.

The model can also be set per run with `--model`.

**Reference data.** Download the
[BPI Challenge 2017](https://data.4tu.nl/articles/dataset/BPI_Challenge_2017/12696884)
event log and put it at `data/BPI_Challenge_2017.xes`. The `data/` folder
is not tracked by git.

---

## Usage

### Run a simulation

```bash
# 20 synthetic cases, quick smoke test
python run_des.py --cases 20

# 200 real BPIC 2017 cases, replayed at their real arrival times
python run_des.py --cases 200 --bpic data/BPI_Challenge_2017.xes --output out_llm

# Same cases, rule-based agents
python run_des.py --cases 200 --bpic data/BPI_Challenge_2017.xes --agents rules --output out_rules

# Larger workforce
python run_des.py --cases 200 --bpic data/BPI_Challenge_2017.xes --jc 12 --sc 6 --co 4
```

| Option | Default | Meaning |
|---|---|---|
| `--cases` / `--offset` | 20 / 0 | Number of cases, and how many to skip first (in arrival order) |
| `--bpic PATH` | none | Load real cases from an XES file. Without it, cases are synthetic |
| `--agents` | `llm` | `llm` or `rules` |
| `--durations` | `complexity` | `complexity`, `llm` (the agent's own estimate), or `distribution` (log-normal sample) |
| `--arrivals` | `replay` | `replay` (real timestamps) or `synthetic` (Poisson, mean `--arrival` minutes) |
| `--jc` / `--sc` / `--co` | 2 / 1 / 1 | Number of workers per role |
| `--noise` | 0.0 | Log-normal noise on work time, as a coefficient of variation |
| `--model` | `$LLM_MODEL` or `gpt-4o-mini` | LLM model name |
| `--seed` | 42 | Random seed |
| `--output` | `output` | Output directory |
| `--verbose` | off | Print debug logs to the console |

Files written to the output directory:

| File | Contents |
|---|---|
| `simulation.xes` / `simulation.json` | The simulated event log, with start and end timestamps for each event |
| `durations.csv` | For each activity: the agent's complexity rating and free estimate, the anchor, and the minutes the engine used |
| `reasoning.csv` | The coding frame: one row per agent action with its justification. The coding columns are left blank for manual coding |
| `reasoning.jsonl` | The same rows, plus the full prompt context and tool input/output |
| `run_manifest.json` | Configuration, seed, and the system prompts exactly as they were at run time |
| `checkpoint.json` | A rolling snapshot, rewritten every 10 completed cases |
| `debug.log` | Full debug log |

### Evaluate against the reference log

```bash
python evaluate.py out_llm/simulation.xes data/BPI_Challenge_2017.xes

# several runs, reported as mean ± CI
python evaluate.py "out_run*/simulation.xes" data/BPI_Challenge_2017.xes

# two configurations against the same reference
python evaluate.py out_llm/simulation.xes data/BPI_Challenge_2017.xes \
    --compare out_rules/simulation.xes --name-a llm --name-b rules
```

The script filters the reference log to the simulated activities and merges
each activity's lifecycle events (schedule/start/complete) into one event.
When the simulation replayed real cases, it matches cases by ID.

It computes the log-distance measures of Chapela-Campa et al.:

- **Control flow:** NGD (n-gram distance)
- **Time:** AED, CED and RED (absolute, circadian and relative event
  distributions)
- **Congestion:** CAR (case arrivals) and CTD (cycle time distribution)

It also prints tables comparing variants, activities and trace lengths. The
report is saved as `evaluation.txt` next to the simulated log. Run
`python evaluate.py -h` for all options.

### Compare two simulated logs

```bash
python compare_logs.py out_llm/simulation.xes out_rules/simulation.xes
```

Reports trace variants, outcomes, the cases where the two versions disagree,
activity frequencies, directly-follows differences and cycle times. The
report is written to `comparison.txt`.

### Recover an interrupted run

```bash
python recover.py out_llm/checkpoint.json
```

Writes an XES log of every case that finished before the run stopped.
Cases that were still in progress are left out.

---

## Results

`results/` holds two runs of 200 replayed BPIC 2017 cases (gpt-4o-mini,
complexity durations, seed 42):

| Run | Workforce (JC / SC / CO) |
|---|---|
| `out_main` | 12 / 6 / 4 |
| `out_small` | 2 / 1 / 1 |

Each folder holds the simulated log, the reasoning export, the run manifest
and `evaluation.txt`. `coherence_sample_classified.csv` is the manually
coded sample used in the qualitative analysis.

---

## Adapting to another process

The engine in `src/queue/` has no knowledge of the loan process. To
simulate a different process:

1. Write a module like
   [loan_application.py](src/process/loan_application.py) that builds a
   `ProcessDefinition`. It needs the roles, agent classes, valid
   transitions, activity and resource maps, silent tools, schedules and an
   initial-state factory.
2. Add the Pydantic tools and agent personas that the definition refers to.
3. Import the new definition in `run_des.py`.

`ProcessDefinition.validate()` runs at engine start-up and reports any
inconsistencies before the simulation begins.
