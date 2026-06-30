from ...utils.llm_utils import convert_format_to_template

causal_system = """You extract explicit causal relations between proposition facts from a passage.

Rules:
- Only use the provided fact IDs and proposition triples.
- Output only direct causal relations that are explicitly supported by the passage.
- Allowed relation types are exactly: causes, enables, prevents.
- Keep the direction strict: source_fact_id -> target_fact_id.
- If no explicit causal relation exists, return {"causal_relations": []}.
"""


causal_frame = """Passage:
```
{passage}
```

Candidate propositions:
{fact_records_json}

Return a JSON object with key "causal_relations".
"""


one_shot_input = causal_frame.format(
    passage="Heavy rain caused flooding, which prevented traffic from moving through downtown.",
    fact_records_json="""[
  {"fact_id": "fact-rain", "triple": ["heavy rain", "caused", "flooding"]},
  {"fact_id": "fact-flood", "triple": ["flooding", "prevented", "traffic moving through downtown"]},
  {"fact_id": "fact-traffic", "triple": ["traffic", "moved through", "downtown"]}
]""",
)


one_shot_output = """{
  "causal_relations": [
    {
      "source_fact_id": "fact-rain",
      "target_fact_id": "fact-flood",
      "relation_type": "causes",
      "confidence": 0.98
    },
    {
      "source_fact_id": "fact-flood",
      "target_fact_id": "fact-traffic",
      "relation_type": "prevents",
      "confidence": 0.92
    }
  ]
}"""


prompt_template = [
    {"role": "system", "content": causal_system},
    {"role": "user", "content": one_shot_input},
    {"role": "assistant", "content": one_shot_output},
    {"role": "user", "content": convert_format_to_template(causal_frame)},
]
