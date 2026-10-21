# Antechamber Benchmark — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""The one plain rendering of a recall payload."""
import unittest

from acbench import render


class Render(unittest.TestCase):
    def test_date_then_sender_and_message(self):
        text = render.render({"winners": [{"date": "2023-05-08T00:00:00Z",
                                           "lines": [{"sender": "Caroline", "message": "hello there"}]}]})
        self.assertEqual(text, "2023-05-08T00:00:00Z\nCaroline: hello there")

    def test_winners_are_separated_by_a_blank_line(self):
        text = render.render({"winners": [{"lines": [{"sender": "a", "message": "one"}]},
                                          {"lines": [{"sender": "b", "message": "two"}]}]})
        self.assertEqual(text, "a: one\n\nb: two")

    def test_a_winner_without_lines_is_one_title_text_line(self):
        self.assertEqual(render.render({"winners": [{"title": "Note", "text": "body"}]}), "Note: body")
        self.assertEqual(render.render({"winners": [{"source": "s", "snippet": "x"}]}), "s: x")

    def test_a_memory_shows_its_own_text_above_source_lines_unless_it_is_one_of_them(self):
        own = render.render({"winners": [{"text": "Topic: a summary", "lines": [{"sender": "u", "message": "line"}]}]})
        self.assertEqual(own, "Topic: a summary\nu: line")
        same = render.render({"winners": [{"text": "line", "lines": [{"sender": "u", "message": "line"}]}]})
        self.assertEqual(same, "u: line")
        capped = render.render({"winners": [{"text": "a long line cut…", "lines": [{"sender": "u", "message": "a long line cut off here"}]}]})
        self.assertEqual(capped, "u: a long line cut off here")

    def test_bad_payloads_render_as_nothing(self):
        self.assertEqual(render.render(None), "")
        self.assertEqual(render.render({"winners": ["junk", {}]}), "")
        self.assertEqual(render.render({"winners": [{"lines": ["junk"]}]}), "")


if __name__ == "__main__":
    unittest.main()
