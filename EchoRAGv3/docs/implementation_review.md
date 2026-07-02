# EchoRAGv3 Implementation Review

## Verdict

No blocking implementation error remains for the claim:

> EchoRAGv3 improves over EchoRAGv2 raw under the same reader setting.

## Fixed During Review

1. Completion replacement is now monotonic.
   A candidate document is promoted only if replacing a current Top-5 document
   strictly increases required-role coverage.

2. Proof atoms shown to the reader are now explicitly source-grounded.
   The adapter filters out atoms without `source_authorization` or
   `exact_source_span`.

3. Missing obligations now prefer EchoRAGv3 selector metadata.
   For v3 rows, `ta_preselector.missing_required_role_ids` is the authoritative
   missing-role field.

4. The preselector test file now runs under plain `python`.
   It also covers the no-net-gain swap case.

5. Reader-eval wording was corrected.
   Paired variants share the same selected Top-5; v3 may differ from the
   original EchoRAGv2 Top-5.

## Verification

- Python compile check passed.
- Preselector tests passed.
- 3x1000 data invariant check passed:
  - Top-5 length valid;
  - no duplicate selected docs;
  - selected docs stay inside candidate pool;
  - rendered atoms stay inside selected Top-5;
  - rendered atoms are source-grounded.
- Main-pair result after fixes still supports the v3 > v2 raw claim:
  - pooled F1: `0.5461 -> 0.5551`;
  - dF1: `+0.0090`;
  - CI95: `[+0.0002, +0.0182]`.

## Remaining Boundary

EchoRAGv3 is proven stronger than EchoRAGv2 raw, not stronger than EchoRAGv2-TA.
