# EchoRAGv3 Main-Pair Limit1000 Result

## Setup

Goal: test whether **EchoRAGv3 > EchoRAGv2**.

Frozen conditions:

- datasets: HotpotQA / 2Wiki / MuSiQue, first 1000 rows each;
- reader: `gpt-4o-mini`;
- same reader prompt and answer parser;
- no answer gate;
- no selector training;
- no candidate pool expansion;
- no ComoRAG comparison in this test.

Compared systems:

- **EchoRAGv2 raw**: original EchoRAGv2 Top-5, no typed atom annotation.
- **EchoRAGv2-TA**: original EchoRAGv2 Top-5 plus Top-5-grounded typed atoms.
- **EchoRAGv3**: EchoRAGv2 Top-5 backbone plus typed-atom completion from the existing candidate pool, then typed atoms to reader.

## Main Result

| comparison | F1 | dF1 | CI95 | EM | dEM | CI95 |
|---|---:|---:|---|---:|---:|---|
| EchoRAGv2 raw -> EchoRAGv3 | 0.5461 -> 0.5551 | +0.0090 | [+0.0002, +0.0182] | 0.4250 -> 0.4380 | +0.0130 | [+0.0030, +0.0237] |
| EchoRAGv2-TA -> EchoRAGv3 | 0.5563 -> 0.5551 | -0.0012 | [-0.0057, +0.0029] | 0.4390 -> 0.4380 | -0.0010 | [-0.0060, +0.0037] |
| EchoRAGv2 raw -> EchoRAGv2-TA | 0.5461 -> 0.5563 | +0.0101 | [+0.0022, +0.0189] | 0.4250 -> 0.4390 | +0.0140 | [+0.0050, +0.0243] |

## Per-Dataset Result

| dataset | EchoRAGv2 raw F1 | EchoRAGv3 F1 | dF1 | CI95 | W/L/T |
|---|---:|---:|---:|---|---|
| HotpotQA | 0.6789 | 0.6745 | -0.0044 | [-0.0184, +0.0094] | 65 / 71 / 864 |
| 2Wiki | 0.5505 | 0.5642 | +0.0137 | [-0.0054, +0.0318] | 84 / 58 / 858 |
| MuSiQue | 0.4089 | 0.4265 | +0.0176 | [+0.0030, +0.0324] | 70 / 52 / 878 |
| pooled | 0.5461 | 0.5551 | +0.0090 | [+0.0002, +0.0182] | 219 / 181 / 2600 |

## Verdict

Supported:

> EchoRAGv3 improves over EchoRAGv2 raw on pooled 3x1000 reader utility.

Not supported:

> EchoRAGv3 is stronger than EchoRAGv2-TA.

Plain meaning:

> typed atoms help EchoRAGv2; the completion step makes this a named v3 algorithm and beats raw v2, but it has not yet proved extra value over simply annotating the original Top-5.

## ComoRAG

ComoRAG is not required for the narrow claim **EchoRAGv3 > EchoRAGv2**.

ComoRAG is only needed for a broader external-baseline claim, such as:

> EchoRAGv3 is competitive with or better than another multi-hop RAG algorithm.
