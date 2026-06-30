import unittest

from build_ergr_role_cache_v1 import (
    extract_json_object,
    normalize_roles,
    unsupported_question_surfaces,
)


class BuildErgrRoleCacheTest(unittest.TestCase):
    def test_extract_json_object_repairs_invalid_escaped_apostrophe(self):
        text = r'''{
          "roles": [
            {
              "role_id": "r0",
              "role_type": "grounding",
              "description": "Identify 'Lovin\' All Night'.",
              "required": true
            }
          ]
        }'''

        parsed = extract_json_object(text)
        roles = normalize_roles(parsed)

        self.assertEqual(roles[0]["description"], "Identify 'Lovin' All Night'.")

    def test_extract_json_object_repairs_missing_comma_between_fields(self):
        text = '''{
          "roles": [
            {
              "role_id": "r0",
              "role_type": "verification",
              "description": "Identify a director.",
              "outputs": ["director"]
              "must_connect_to": ["r1"],
              "required": true
            }
          ]
        }'''

        parsed = extract_json_object(text)
        roles = normalize_roles(parsed)

        self.assertEqual(roles[0]["outputs"], ["director"])
        self.assertEqual(roles[0]["must_connect_to"], ["r1"])

    def test_extract_json_object_repairs_array_quote_bracket_transposition(self):
        text = '''{
          "roles": [
            {
              "role_id": "r0",
              "role_type": "verification",
              "description": "Identify a director.",
              "outputs": ["director of 'Les Misérables (1917 Film)']",
              "required": true
            }
          ]
        }'''

        parsed = extract_json_object(text)
        roles = normalize_roles(parsed)

        self.assertEqual(roles[0]["outputs"], ["director of 'Les Misérables (1917 Film)'"])

    def test_question_grounding_audit_flags_answer_entity_injection(self):
        question = "Who is the sister of the actress who played Susie in Miracle on 34th Street?"
        text = "Find evidence that Natalie Wood has sister Lana Wood."

        unsupported = unsupported_question_surfaces(text, question)

        self.assertIn("Natalie Wood", unsupported)
        self.assertIn("Lana Wood", unsupported)
        self.assertNotIn("Susie", unsupported)

    def test_normalize_roles_can_drop_ungrounded_role_in_strict_mode(self):
        question = "Who is the sister of the actress who played Susie in Miracle on 34th Street?"
        payload = {
            "roles": [
                {
                    "role_id": "r0",
                    "role_type": "bridge",
                    "description": "Identify the actress who played Susie in Miracle on 34th Street.",
                },
                {
                    "role_id": "r1",
                    "role_type": "answer_bearing",
                    "description": "Find evidence that Natalie Wood has sister Lana Wood.",
                },
            ]
        }

        audited = normalize_roles(payload, question=question)
        strict = normalize_roles(payload, question=question, strict_question_grounding=True)

        self.assertEqual(len(audited), 2)
        self.assertTrue(audited[0]["question_grounded"])
        self.assertFalse(audited[1]["question_grounded"])
        self.assertEqual([role["role_id"] for role in strict], ["r0"])


if __name__ == "__main__":
    unittest.main()
