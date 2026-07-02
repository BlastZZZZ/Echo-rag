# EchoRAGv3 Method Spec

## Name

EchoRAGv3 = EchoRAGv2 with typed-atom completion.

## Core Idea

EchoRAGv3 does not rebuild the whole Top-5.

It keeps EchoRAGv2's selected Top-5 as the backbone, then checks whether typed
atoms show that a required evidence-chain role is missing. If a missing role can
be covered by another document already in the EchoRAGv2 candidate pool,
EchoRAGv3 swaps in that document.

## Flow

```text
question
  -> EchoRAGv2 candidate pool
  -> EchoRAGv2 initial Top-5
  -> typed atoms over candidate pool
  -> complete missing evidence-chain roles
  -> final Top-5
  -> Top-5 + typed atoms to reader
  -> answer
```

## Boundary

EchoRAGv3 does not:

- expand the candidate pool;
- train a selector;
- use gold answers for selection;
- add an answer gate;
- replace the reader;
- do full Top-5 re-selection.

## Current Status

EchoRAGv3 is clean and improves over raw EchoRAGv2 in the 3x1000 main-pair
reader evaluation.

It has not yet shown clear improvement over reader-side EchoRAGv2-TA.

Current supported claim:

> EchoRAGv3 improves over EchoRAGv2 raw on pooled reader utility, but the extra
> completion step is not yet stronger than simply adding Top-5-grounded typed
> atom annotations to EchoRAGv2.
