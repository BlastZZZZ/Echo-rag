# EchoRAGv3

EchoRAGv3 is a frozen research implementation of **EchoRAGv2 with typed-atom completion**.

## Core Idea

EchoRAGv3 keeps the EchoRAGv2 Top-5 as the backbone. It uses source-grounded
typed atoms to check whether a required evidence-chain role is missing. If the
missing role can be covered by another document already in the EchoRAGv2
candidate pool, EchoRAGv3 swaps that document into the final Top-5.

```text
EchoRAGv2 candidate pool
  -> EchoRAGv2 initial Top-5
  -> source-grounded typed atoms
  -> completion-only Top-5 repair
  -> final Top-5 + typed atom annotation
  -> reader
```

## Boundary

EchoRAGv3 does not:

- expand the candidate pool;
- train a selector;
- use gold answers for selection;
- add an answer gate;
- replace the reader;
- do full Top-5 re-selection.

## Supported Claim

On the frozen 3x1000 main-pair evaluation:

| system | pooled F1 |
|---|---:|
| EchoRAGv2 raw | 0.5461 |
| EchoRAGv3 | 0.5551 |

Delta F1: `+0.0090`, CI95 `[+0.0002, +0.0182]`.

This supports:

> EchoRAGv3 improves over EchoRAGv2 raw under the same reader setting.

It does **not** support:

> EchoRAGv3 is stronger than EchoRAGv2-TA.

## Repository Layout

```text
src/echoragv3_selector.py     completion-only Top-5 repair
src/proof_adapter.py          source-grounded proof atom projection
src/reader_eval.py            optional reader replay/evaluation
tests/test_selector.py        selector unit tests
scripts/run_select.sh         relative-path selector runner
scripts/run_reader_eval.sh    relative-path reader-eval runner
docs/                         method and implementation review
results/                      frozen reported results
```

## Inputs

The selector expects a JSON file with rows containing at least:

- `candidate_docs`: candidate document ids in retrieval order;
- `selected_doc_indices`: EchoRAGv2 initial Top-5;
- `roles` or `role_diagnostics`: required evidence-chain roles;
- `typed_atoms` or `typed_atom_proof_atoms`: source-grounded atoms.

Each atom should include:

- `source_doc_index`;
- `role_id`;
- `source_authorization`;
- `exact_source_span`.

## Run Selector

```bash
python src/echoragv3_selector.py \
  --typed-atoms-json data/input_typed_atoms.json \
  --corpus-json data/corpus.json \
  --output-json runs/typed_atoms_echoragv3.json \
  --output-md runs/typed_atoms_echoragv3.md \
  --mode completion
```

or:

```bash
scripts/run_select.sh data/input_typed_atoms.json data/corpus.json runs/typed_atoms_echoragv3.json runs/typed_atoms_echoragv3.md
```

## Optional Reader Evaluation

Copy `.env.example` to `.env` and set `OPENAI_API_KEY`, or export it in your shell.

```bash
scripts/run_reader_eval.sh \
  runs/typed_atoms_echoragv3.json \
  data/corpus.json \
  data/dataset.json \
  MuSiQue \
  runs/reader_eval.json \
  runs/reader_eval.md \
  runs/reader_eval_cache.json \
  1000
```

The reader-eval script is optional. The selector itself has no network dependency.

## Tests

```bash
python tests/test_selector.py
python -m py_compile src/*.py
```

## Notes

- No API keys are stored in this repository.
- Scripts use relative paths and environment variables.
- Large datasets, caches, and model outputs are intentionally not included.
