from ...utils.llm_utils import convert_format_to_template

proposition_system = """Your task is to decompose a passage into fully contextualized atomic propositions.

Each proposition must:
- express one complete unit of meaning
- preserve critical context such as time, location, comparison, causality, and conditions
- use only entities from the provided named_entities list
- be written as a standalone natural-language statement

Return strict JSON with one field: "propositions".
Each proposition must be an object with:
- "text": the proposition as a complete natural-language statement
- "entities": an array of entities from named_entities that appear in the proposition
"""

proposition_frame = """Passage:
```
{passage}
```

Named entities: {named_entities}
"""

proposition_example_input = """Passage:
```
In September 2023, Apple replaced the Lightning connector with USB-C on the iPhone 15 after the European Union passed regulations requiring a standardized charging port.
```

Named entities: ["Apple", "Lightning connector", "USB-C", "iPhone 15", "European Union", "regulations", "September 2023", "standardized charging port"]
"""

proposition_example_output = """{
  "propositions": [
    {
      "text": "The iPhone 15 uses USB-C instead of the Lightning connector.",
      "entities": ["iPhone 15", "USB-C", "Lightning connector"]
    },
    {
      "text": "The European Union passed regulations requiring a standardized charging port.",
      "entities": ["European Union", "regulations", "standardized charging port"]
    },
    {
      "text": "Apple changed the iPhone 15 from Lightning to USB-C in September 2023 after the European Union regulations.",
      "entities": ["Apple", "iPhone 15", "Lightning connector", "USB-C", "September 2023", "European Union", "regulations"]
    }
  ]
}"""

prompt_template = [
    {"role": "system", "content": proposition_system},
    {"role": "user", "content": proposition_example_input},
    {"role": "assistant", "content": proposition_example_output},
    {"role": "user", "content": convert_format_to_template(original_string=proposition_frame, placeholder_mapping=None, static_values=None)},
]
