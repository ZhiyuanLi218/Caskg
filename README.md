<h1 align="center">CaSKG</h1>

<p align="center">
  <strong>Counterfactual-Causal Skill Graphs for LLM Agent Skill Libraries</strong>
</p>

<p align="center">
  Build a skill graph offline, validate candidate dependencies with counterfactual
  probes, and retrieve a small prerequisite-aware skill bundle at runtime.
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10--3.12-3776ab?logo=python&logoColor=white" alt="Python 3.10-3.12"></a>
  <a href="https://docs.astral.sh/uv/"><img src="https://img.shields.io/badge/managed%20with-uv-6e9f18" alt="Managed with uv"></a>
  <img src="https://img.shields.io/badge/status-research%20code-orange" alt="Research code">
</p>

---

CaSKG is a research implementation for turning a directory of `SKILL.md`
documents into a searchable, causal-aware graph. It separates inexpensive
association signals from intervention evidence, then uses the resulting graph
to assemble bounded context for an LLM agent.

This repository contains the core package, command-line and MCP interfaces,
counterfactual validation, benchmark runners, frozen protocol metadata, and
tests. Large skill corpora, API credentials, benchmark installations, raw
model outputs, and prebuilt workspaces are external assets and are not part of
the code checkout.

## Overview

A flat skill library makes it difficult to answer questions such as "what must
run before this skill?" or "which other skill repairs a failed attempt?" CaSKG
represents skills as nodes and directed relations as edges. Relations are
treated as hypotheses until counterfactual evidence supports them.

The online retriever combines semantic and lexical seeds with graph structure,
traverses prerequisite relations backwards, and returns only a character-bounded
bundle. The agent therefore receives relevant skill content without loading the
entire library.

## How It Works

~~~text
SKILL.md library
      |
      v
Phase 1: parse, embed, and induce candidate edges
         semantic | lexical | I/O | co-occurrence | repair | LLM judge
      |
      v
Phase 2: counterfactual validation
         removal | substitution | reordering
         Bayesian edge posterior and status update
      |
      v
Frozen runtime graph
      |
      v
Phase 3: online retrieval
         semantic + lexical seeds -> backward graph propagation
         -> bounded, agent-ready skill bundle
~~~

1. **Candidate induction.** Skill metadata and content are parsed, embedded,
   and scored with six signals: semantic similarity, lexical overlap, input /
   output compatibility, execution co-occurrence, repair traces, and an
   optional LLM edge judge.
2. **Causal validation.** The validation runner tests selected edges by removing
   a skill, substituting an alternative, or reordering a proposed dependency.
   Outcomes update a Beta-Binomial posterior and classify edges as confirmed,
   rejected, or deferred.
3. **Causal retrieval.** At task time, CaSKG seeds retrieval from the query,
   applies causal and prerequisite-aware scoring (including backward PPR), and
   hydrates a bounded context payload for the agent.

## Repository Layout

~~~text
.
|-- caskg/
|   |-- causal/                   candidate induction, interventions, graph maintenance
|   |-- core/                     parsing, storage, embeddings, retrieval
|   |-- interfaces/               CLI and MCP entry points
|   +-- utils/                    environment-based configuration
|-- experiments/
|   +-- run_validation.py         Phase 2 counterfactual validation
|-- evaluation/
|   |-- alfworld_run.py           ALFWorld environment loop
|   |-- scienceworld_eto211_run.py ScienceWorld Unseen-211 evaluator
|   |-- retrievers/               read-only retrieval worker boundary
|   |-- skills_ref/               evaluator-facing skill parser
|   +-- tests/                    evaluator and protocol tests
|-- ablation_experiments/         P0 and A1-A4 controlled interventions
|-- configs/                      benchmark protocols and worker settings
|-- manifests/                    episode selections and checksums
|-- prompts/                      frozen benchmark prompts
|-- tests/                        CaSKG unit and integration tests
|-- .env.example                  credential-free configuration template
|-- .python-version               tested Python version (3.12.13)
|-- pyproject.toml                package metadata and CLI entry points
+-- uv.lock                       locked Python dependency graph
~~~

Generated directories such as `data/`, `results/`,
`ablation_experiments/cache/`, and
`ablation_experiments/graph_views/generated/` are created locally when
running experiments. They are not required for importing the core package.

## Requirements

- Python 3.10 through 3.12. The supplied lock file was checked with Python
  3.12.13.
- [`uv`](https://docs.astral.sh/uv/) for the locked environment.
- An OpenAI-compatible chat endpoint and embedding endpoint for indexing,
  validation, and model-based evaluation.
- ALFWorld 0.4.2 and its TextWorld data for the ALFWorld runner.
- ScienceWorld 1.2.3, Py4J 0.10.9.9, and a working Java runtime for the
  ScienceWorld runner.

On Windows, command-line smoke tests can be run from PowerShell. Native
Windows installation of `hnswlib` may require Microsoft C++ Build Tools, so
WSL2 is recommended for benchmark runs and native dependency installation.
Commands containing `export`, `$PWD`, or Bash line continuations are written
for Bash / WSL; translate environment assignments to PowerShell syntax when
using a native shell. All commands below assume the repository root as the
working directory.

## Installation

Install the locked base environment:

~~~bash
uv python install 3.12.13
uv sync --frozen
~~~

The project accepts any Python version in the 3.10-3.12 range. If your
platform does not offer 3.12.13, install the latest available 3.12.x and
select it explicitly:

~~~bash
uv python install 3.12
uv sync --frozen --python 3.12
~~~

Install the optional ALFWorld dependency only when that benchmark is needed:

~~~bash
uv sync --frozen --extra alfworld
~~~

ScienceWorld is currently installed separately because it is not included in
`uv.lock`:

~~~bash
uv pip install scienceworld==1.2.3 py4j==0.10.9.9
~~~

Create a local environment file and keep credentials out of version control:

~~~bash
# Bash / WSL
cp .env.example .env
~~~

~~~powershell
# PowerShell
Copy-Item .env.example .env
~~~

Check the environment before making model calls:

~~~bash
uv run python --version
uv run python -c "import scienceworld; print(scienceworld.__version__)"
java -version
~~~

## Configuration

The main variables in [`.env.example`](.env.example) are:

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Credential for an OpenAI-compatible chat or embedding service |
| `OPENAI_BASE_URL` | Base URL used by the CaSKG services |
| `API_KEY`, `BASE_URL` | Chat credentials used by the basic ALFWorld runner |
| `CASKG_LLM_MODEL` | Model used for candidate induction and validation |
| `CASKG_EMBEDDING_MODEL` | Embedding model used for indexing and retrieval |
| `CASKG_EMBEDDING_DIM` | Embedding dimension; must match the workspace |
| `CASKG_WORKING_DIR` | Default CaSKG workspace |
| `CASKG_PREBUILT_WORKING_DIR` | Optional prebuilt workspace fallback |
| `CASKG_ENABLE_QUERY_REWRITE` | Enables query rewriting when set to `true` |
| `ROUTER_MASTER_KEY` | Credential for the controlled benchmark router |
| `CASKG_ROUTER_BASE` | Router URL for the ALFWorld parity runner |
| `SCIENCEWORLD_ROUTER_BASE_URL` | Router URL for ScienceWorld |
| `CASKG_EXPECTED_CHAT_PROVIDER` | Provider identity required by the strict router |
| `CASKG_ROUTER_SESSION_ID` | Session identifier for paired benchmark runs |
| `ALFWORLD_DATA` | Root of the downloaded ALFWorld data |
| `JAVA_HOME` | Java installation used by ScienceWorld |

Keep the embedding model and dimension unchanged between indexing and
retrieval. A workspace built with a different embedding model is not
compatible, even when the vector dimensions happen to match.

The formal ScienceWorld runner expects the response header
`X-Router-Policy: strict`. If `--expected-router-provider` is supplied, each
response must also contain the matching `X-Router-Provider` header. A generic
OpenAI-compatible endpoint can be used for development and preflight checks,
but a missing header is recorded as an infrastructure error in a formal run.

The ALFWorld script loads the local environment file on startup. The
ScienceWorld and numbered ablation runners read the process environment
directly, so export the required variables before launching them. In Bash or
WSL, for example:

~~~bash
set -a
. ./.env
set +a
~~~

For PowerShell, set the router and credential variables with environment
assignments before running ScienceWorld or the numbered ablations.

## Quick Start

The shortest end-to-end path is: provide a skill directory, build a candidate
graph, inspect it, and issue a retrieval query. The skill corpus is not bundled
with this repository; use any directory whose entries contain valid
`SKILL.md` files.

Expected input shape:

~~~text
data/
+-- skillsets/
    +-- skills_1000/
        |-- skill-a/
        |   +-- SKILL.md
        +-- skill-b/
            +-- SKILL.md
~~~

Each `SKILL.md` should contain YAML frontmatter with at least a `name`. A
`description` is recommended. The parser can fall back to a Description,
Overview, Summary, or the first non-heading sentence when it is absent.

Build and query the graph:

~~~bash
uv run caskg-index data/skillsets/skills_1000 \
  --workspace data/caskg_workspace/skills_1000 \
  --clear

uv run caskg status \
  --workspace data/caskg_workspace/skills_1000

uv run caskg-query "How do I plan a tool-based task?" \
  --workspace data/caskg_workspace/skills_1000
~~~

`--clear` removes the target workspace before indexing. Omit it to reuse an
existing workspace. Indexing and querying require the configured embedding
service; candidate induction may also call the configured LLM service.

## CLI

The package exposes these entry points through `pyproject.toml`:

| Command | Use |
|---|---|
| `caskg-index PATH` | Parse skills, compute embeddings, and build the CaSKG candidate graph |
| `caskg add PATH` | Add a skill file or directory incrementally |
| `caskg status` | Show node and edge counts for a workspace |
| `caskg-query QUERY` | Run causal-aware retrieval and print a concise result |
| `caskg-query QUERY --json` | Emit the structured causal retrieval record |
| `caskg-server` | Start the basic MCP server using `CASKG_WORKING_DIR` |

Use `uv run <command> --help` for the complete option list. The general
`caskg query` command is also available for the lower-level graph retriever;
`caskg-query` is the causal-aware entry point used by the benchmark adapters.

## MCP Integration

For the basic MCP server, set the workspace in the environment and launch:

~~~bash
CASKG_WORKING_DIR=data/caskg_workspace/skills_1000 uv run caskg-server
~~~

PowerShell equivalent:

~~~powershell
$env:CASKG_WORKING_DIR = "data/caskg_workspace/skills_1000"
uv run caskg-server
~~~

The richer Claude Code-compatible server accepts an explicit workspace:

~~~bash
uv run python -m caskg.interfaces.claude_code \
  --workspace data/caskg_workspace/skills_1000
~~~

The MCP tools expose search, bundle retrieval, hydration by skill name, graph
status, skill details, and graph-neighbor inspection. A workspace must already
be indexed before retrieval tools can return skill content.

## External Assets

The following assets are deliberately separate from the code checkout:

~~~text
data/
|-- skillsets/
|   +-- skills_1000/
|       +-- <skill-name>/SKILL.md
+-- caskg_workspace/
    |-- skills_1000/
    +-- skills_1000_v32_scaffold_publish_gospath/
~~~

`configs/retrievers_v1.json` expects
`data/caskg_workspace/skills_1000` for the main ScienceWorld run. The numbered
ablation builder expects the frozen
`skills_1000_v32_scaffold_publish_gospath` workspace. They may be copies or
symbolic links to the same frozen workspace.

For exact P0/A1-A4 reproduction, the frozen workspace must contain at least:

~~~text
candidate_checkpoint.jsonl
caskg_state.json
chunks_kv_data.pkl
entities_hnsw_index_4096.bin
entities_hnsw_metadata.pkl
graph_igraph_data.pklz
map_e2r_blob_data.pkl
map_r2c_blob_data.pkl
~~~

The reference Skill1000 workspace has 1,000 nodes and 3,292 published edges.
The graph-view builder checks these additional digests:

| Asset | Expected SHA256 |
|---|---|
| Runtime graph | `ff64ad00ef7b35e29586ac59ed8bec07f0a1452555a9ff19f89712fd85654770` |
| Phase 1 checkpoint | `dd46baaab214491ebcf75b769d4608e067abc5aefbf8b97253edad7924816be0` |

There is no public workspace download URL in this repository. A newly built
workspace is suitable for method-level experimentation but will not pass the
byte-level ablation gate unless it has the expected hashes.

### ALFWorld data

Install the optional dependency, download the environment data, and set the
root path:

~~~bash
uv sync --frozen --extra alfworld
uv run alfworld-download --data-dir data/alfworld
export ALFWORLD_DATA="$PWD/data/alfworld"
~~~

The supplied ALFWorld manifest records task identities and checksums only; it
does not include the skill corpus or environment data.

### ScienceWorld data

ScienceWorld 1.2.3 installs its environment resources with the Python package.
Set `JAVA_HOME` if `java` is not already on `PATH`, then validate the frozen
episode selection:

~~~bash
uv run python -m evaluation.scienceworld_eto211_run --validate-only
~~~

A successful validation reports 211 episodes, 24 task types, the official
`test` split, `easy` simplification, a 30-step limit, and 1,819 official test
variations before selection.

## Reproducibility Workflow

### Phase 1: candidate graph construction

Run the indexer from the repository root:

~~~bash
uv run caskg-index data/skillsets/skills_1000 \
  --workspace data/caskg_workspace/skills_1000 \
  --clear
~~~

The command discovers `SKILL.md` files recursively, builds skill nodes and
embeddings, fuses the six candidate signals, writes `caskg_state.json`, and
publishes the candidate retrieval graph. Every edge is still a hypothesis at
this point.

### Phase 2: counterfactual validation

Validate selected candidate edges with all three probe types:

~~~bash
uv run python experiments/run_validation.py \
  --workspace data/caskg_workspace/skills_1000 \
  --skills-dir data/skillsets/skills_1000 \
  --probe-types removal,substitution,reordering \
  --threshold 0.05 \
  --max-edges 500 \
  --resume \
  --call-delay 1.05 \
  --batch-concurrency 16 \
  --seed 42
~~~

The runner appends `validation_log.jsonl`, updates `caskg_state.json`, writes
`validation_summary.json`, freezes edges outside the validation budget, and
republishes the runtime graph. `--resume` skips probes already present in the
log. `--dry-run` simulates outcomes and is useful for testing the control flow,
but its results are not experimental evidence.

The reference Skill1000 graph has these checkpoints:

| Stage | Count |
|---|---:|
| Phase 1 candidate edges | 9,937 |
| Edges selected for Phase 2 | 500 |
| Counterfactual probes | 1,500 |
| Confirmed causal | 35 |
| Rejected non-causal | 215 |
| Deferred uncertain | 250 |
| Deferred unvalidated | 9,437 |
| Published runtime edges | 3,292 |

Remote model calls and scheduling can change a newly built graph. These counts
are reference checkpoints, not a guarantee that a fresh run will be identical.

## Benchmark Evaluation

### ALFWorld ID-140

The basic runner evaluates the `eval_in_distribution` games when `--split dev`
is used. The complete reference condition has 140 episodes:

~~~bash
uv run python evaluation/alfworld_run.py \
  --model MiniMax-M2.7 \
  --split dev \
  --max_workers 4 \
  --max_steps 30 \
  --exp_name skills_1000 \
  --use_skill \
  --mode caskg \
  --caskg_workspace data/caskg_workspace/skills_1000 \
  --skills_dir data/skillsets/skills_1000
~~~

The basic runner writes one record per episode under:

~~~text
results/alfworld/MiniMax-M2.7/dev_skills_1000_mode_caskg/idx_<episode>.json
~~~

The underscore-style flags (`--max_workers`, `--caskg_workspace`, and so on)
match the existing runner interface. Use the parity runner in the ablation
section when the frozen ID-140 task manifest and checksums must be enforced.

### ScienceWorld Unseen-211

Run a read-only retrieval preflight before a paid model run:

~~~bash
uv run python -m evaluation.scienceworld_eto211_run \
  --retriever caskg \
  --max-episodes 1 \
  --retrieval-preflight
~~~

Run the complete CaSKG condition with the formal four-worker protocol:

~~~bash
uv run python -m evaluation.scienceworld_eto211_run \
  --retriever caskg \
  --model MiniMax-M2.7 \
  --api-base "$SCIENCEWORLD_ROUTER_BASE_URL" \
  --api-key-env ROUTER_MASTER_KEY \
  --expected-router-provider "$CASKG_EXPECTED_CHAT_PROVIDER" \
  --run-cohort caskg-u211 \
  --max-workers 4 \
  --output-dir results/scienceworld
~~~

The formal protocol requires one attempt for each of 211 episodes, four
workers, a nonempty provider identity, and a nonempty cohort name. Existing
records are resumed only when their protocol, prompt, manifest, retriever
configuration, model, route, and evaluator fingerprint match the current run.
The summary is written below:

~~~text
results/scienceworld/eto_skillnet_unseen211/caskg/MiniMax-M2.7/summary.json
~~~

## Ablation Experiments

The numbered protocol compares each intervention with a concurrent P0
condition. The same provider, prompt, task manifest, evaluator, worker count,
and time window are held fixed.

| Condition | Intervention |
|---|---|
| `p0-api2-full-control` | Full CaSKG retrieval on the frozen graph |
| `a1-vector-only-no-ppr` | Vector top-N only; PPR and lexical expansion disabled |
| `a2-matched-rewired-s7302` | Within-type, degree-preserving endpoint rewiring (seed 7302) |
| `a3-fixed-topology-phase1-weights` | Frozen topology with Phase 1 association weights |
| `a4-phase2-evidence-shuffle-s7301` | Matched Phase 2 evidence-package shuffle (seed 7301) |

The protocol and acceptance rules are frozen in
[`ablation_experiments/protocols/caskg-s1000-a1-a4-main-parity-v1.yaml`](ablation_experiments/protocols/caskg-s1000-a1-a4-main-parity-v1.yaml).
Generated graph views are not included; build them from the external frozen
workspace:

~~~bash
uv run python ablation_experiments/scripts/build_numbered_graph_views.py \
  --source-workspace data/caskg_workspace/skills_1000_v32_scaffold_publish_gospath \
  --output-root ablation_experiments/graph_views/generated/a1-a4-main-parity-v1
~~~

The builder treats the source workspace as read-only and writes isolated views
plus `asset_manifest.json`. It refuses to overwrite an existing output unless
`--force` is supplied. P0 intentionally uses the byte-identical A1 workspace;
the runtime configuration, rather than the graph bytes, disables the A1
retrieval operation.

### ALFWorld ablation condition

Set `VARIANT`, `WORKSPACE`, and `RUN_ROOT` for a row in the table above. This
example runs P0:

~~~bash
export VARIANT=p0-api2-full-control
export WORKSPACE="$PWD/ablation_experiments/graph_views/generated/a1-a4-main-parity-v1/a1-vector-only-no-ppr/workspace"
export RUN_ROOT="$PWD/ablation_experiments/results/a1-a4-main-parity-v1/formal/reproduction"

export CASKG_ABLATION_EMBEDDING_CACHE="$PWD/ablation_experiments/cache/a1-a4-main-parity-v1/query_embeddings.sqlite3"
export CASKG_ABLATION_RETRIEVAL_AUDIT_DIR="$PWD/ablation_experiments/audits/runtime/a1-a4-main-parity-v1/$VARIANT/alfworld"
export CASKG_ABLATION_TRACE_DIR="$PWD/ablation_experiments/traces/a1-a4-main-parity-v1/$VARIANT/alfworld"

uv run python -m ablation_experiments.runners.run_alfworld_numbered_staged \
  --variant "$VARIANT" \
  --workspace "$WORKSPACE" \
  --skills-dir data/skillsets/skills_1000 \
  --output-dir "$RUN_ROOT/alfworld-id140/$VARIANT" \
  --model MiniMax-M2.7 \
  --split dev \
  --max-workers 4 \
  --max-steps 30 \
  --historical-task-manifest ablation_experiments/manifests/generated/alfworld-s1000-main-a0-task-manifest-v1.json
~~~

This runner also requires `ROUTER_MASTER_KEY`, `CASKG_ROUTER_BASE`,
`CASKG_EXPECTED_CHAT_PROVIDER`, and `CASKG_ROUTER_SESSION_ID`. Repeat the
command for A1 through A4 using their mapped workspaces. The preregistered
groups are P0/A1, A2/A3, and A4; keep the total active workers within the
available provider budget.

### ScienceWorld ablation condition

Change `VARIANT` and `CONFIG` for each row in the runtime mapping:

| Condition | Workspace suffix | Runtime configuration |
|---|---|---|
| P0 | `a1-vector-only-no-ppr/workspace` | `scienceworld-p0-api2-full-control.json` |
| A1 | `a1-vector-only-no-ppr/workspace` | `scienceworld-a1-vector-only-no-ppr.json` |
| A2 | `a2-matched-rewired-s7302/workspace` | `scienceworld-a2-matched-rewired-s7302.json` |
| A3 | `a3-fixed-topology-phase1-weights/workspace` | `scienceworld-a3-fixed-topology-phase1-weights.json` |
| A4 | `a4-phase2-evidence-shuffle-s7301/workspace` | `scienceworld-a4-phase2-evidence-shuffle-s7301.json` |

~~~bash
export VARIANT=p0-api2-full-control
export CONFIG=scienceworld-p0-api2-full-control.json
export RUN_ROOT="$PWD/ablation_experiments/results/a1-a4-main-parity-v1/formal/reproduction"

uv run python -m ablation_experiments.runners.run_scienceworld_numbered \
  --retriever-config "ablation_experiments/configs/runtime/numbered/$CONFIG" \
  --retriever caskg \
  --model MiniMax-M2.7 \
  --api-base "$SCIENCEWORLD_ROUTER_BASE_URL" \
  --api-key-env ROUTER_MASTER_KEY \
  --expected-router-provider "$CASKG_EXPECTED_CHAT_PROVIDER" \
  --run-cohort minimax-m27-a1-a4-main-parity-v1 \
  --max-workers 4 \
  --output-dir "$RUN_ROOT/scienceworld-unseen211/$VARIANT"
~~~

### Analyze paired results

The analysis is fail-closed and requires all 140 ALFWorld and all 211
ScienceWorld records for P0 and A1-A4, plus historical A0 records supplied
outside this repository:

~~~text
results/historical/alfworld/MiniMax-M2.7/idx_*.json
results/historical/scienceworld/MiniMax-M2.7/**/episode_*.json
~~~

~~~bash
uv run python ablation_experiments/scripts/analyze_numbered_results.py \
  --run-root "$RUN_ROOT" \
  --historical-a0-alfworld results/historical/alfworld/MiniMax-M2.7 \
  --historical-a0-scienceworld results/historical/scienceworld/MiniMax-M2.7
~~~

The script writes `numbered_ablation_analysis.json` and
`NUMBERED_ABLATION_RESULTS.md` under `$RUN_ROOT/analysis/`. It uses 10,000
paired bootstrap samples, exact McNemar tests for binary outcomes, a paired
sign-flip test for ScienceWorld scores, and Holm correction across the four
P0-minus-ablation contrasts per benchmark.

## Reference Run Checkpoints

The following values are recorded reference-run checkpoints. Raw episode
outputs are external and are not generated during installation.

| Condition | ALFWorld success | ScienceWorld mean score | ScienceWorld full success |
|---|---:|---:|---:|
| Historical A0 | 103/140 (73.57%) | 68.33 | 43.13% |
| Concurrent P0 | 96/140 (68.57%) | 72.09 | 49.29% |
| A1 | 105/140 (75.00%) | 68.46 | 47.39% |
| A2 | 95/140 (67.86%) | 68.49 | 45.97% |
| A3 | 102/140 (72.86%) | 72.59 | 51.66% |
| A4 | 93/140 (66.43%) | 70.57 | 46.92% |

Historical A0 and concurrent P0 used different ScienceWorld provider and
evaluator conditions. Only paired P0 versus A1-A4 comparisons are used for
ablation inference. A remote-model rerun is stochastic and may differ at the
episode level; protocol, graph, task, prompt, and completion checks are the
relevant parity signals.

## Tests

Run the model-free unit and protocol tests:

~~~bash
uv run pytest -m "not integration" tests evaluation/tests \
  ablation_experiments/tests/test_causal_graph_views.py
~~~

In the checked Python 3.12.13 environment this command completed with
`125 passed, 5 deselected`. The evaluator-only suite can also be run directly:

~~~bash
uv run pytest evaluation/tests
~~~

The checked evaluator-only run completed with 13 passed. Test counts can
change when dependencies or protocol files change.

The numbered graph-view tests require the external frozen workspace and the
generated views:

~~~bash
uv run pytest \
  ablation_experiments/tests/test_numbered_graph_views.py \
  ablation_experiments/tests/test_numbered_runtime.py
~~~

The ALFWorld parity test imports a runner that checks for a router credential at
import time. A non-secret placeholder is enough for this unit test:

~~~bash
ROUTER_MASTER_KEY=test-only uv run pytest \
  ablation_experiments/tests/test_alfworld_main_parity.py
~~~

Check syntax without starting an experiment:

~~~bash
uv run python -m compileall -q \
  caskg experiments evaluation ablation_experiments tests
~~~

The complete `tests/` suite includes integration tests that need a real model
endpoint and workspace. Without those external services, a small number of
integration failures is expected; it does not indicate that the model-free
tests failed. In the checked environment, the full suite finished with
123 passed and 3 integration failures.

## Output Checks

A complete ALFWorld condition contains 140 valid episode records and a summary
with:

~~~text
metrics.valid_score_count = 140
metrics.missing_or_infrastructure_count = 0
metrics.formal_complete = true
~~~

A complete ScienceWorld condition contains 211 valid records and a summary
with:

~~~text
metrics.valid_score_count = 211
metrics.infrastructure_error_count = 0
metrics.formal_complete = true
~~~

These summary fields apply to the protocol/parity runners. The basic
`evaluation/alfworld_run.py` entry point writes per-episode `idx_*.json` files
and does not by itself create a formal summary.

Keep completed episodes even when their outcomes differ from a reference run.
Infrastructure failures are reported separately from model failures.

## Known Limitations

- The Skill1000 skill text, API credentials, raw model responses, benchmark
  installations, and frozen workspaces are not included.
- No public URL for the exact frozen CaSKG workspace is configured here.
  Method-level runs can build a new graph, but exact byte-level ablations need
  the matching external workspace.
- `ablation_experiments/graph_views/generated/` is generated output, so the
  numbered ablation is not a one-command checkout-and-run experiment.
- `configs/retrievers_v1.json` currently describes the CaSKG worker. A complete
  paired comparison-method implementation is outside this repository.
- `--seed 42` controls local random choices but cannot make remote LLM calls or
  concurrent provider scheduling fully deterministic.
- Formal ScienceWorld evaluation requires the strict router headers described
  above. A plain OpenAI-compatible endpoint is suitable only for development
  and retrieval preflight.
- Native Windows may expose path-separator differences in one parser assertion;
  WSL2 or Linux matches the tested benchmark environment.

This means the repository supports method inspection, local smoke tests, and
controlled reproduction when the external assets and services are supplied. It
does not claim that every reported benchmark table can be regenerated from the
checkout alone.

## Troubleshooting

| Symptom | Action |
|---|---|
| `No usable Java runtime was found` | Install a JDK, set `JAVA_HOME` to its root, and confirm `java -version` in the same shell. |
| `Embedding dimension mismatch` | Restore the embedding model and dimension used to build the workspace. The reference workspace uses dimension 4096. |
| `Workspace does not exist` or `Invalid CaSKG workspace` | Check both the skill directory and its matching workspace path relative to the repository root. |
| `Runtime graph checksum mismatch` | Use the frozen external workspace and verify the two SHA256 values in External Assets. |
| `Router provider` or `policy` mismatch | Use the controlled strict router and set `CASKG_EXPECTED_CHAT_PROVIDER` to the returned provider identity. |
| `retrieval worker exited` or timed out | Confirm the runtime JSON points to this project root, the selected workspace exists, and the embedding endpoint is reachable. |
| Incomplete ALFWorld or ScienceWorld records | Supply every required episode; the analysis intentionally fails closed on missing records. |
| `ALFWORLD_DATA` paths unresolved | Confirm the root contains `json_2.1.1/valid_seen` and `logic/alfred.pddl`. |
| One parsing test fails only on native Windows | Run the suite under WSL/Linux; the assertion compares a portable `scripts/run.py` spelling with Windows backslashes. |

For additional command options, inspect the `--help` output of the relevant
runner and the protocol files under `configs/` and
`ablation_experiments/protocols/`.
