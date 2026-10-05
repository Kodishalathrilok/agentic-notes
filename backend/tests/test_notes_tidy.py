"""Deterministic clean-up of finished notes: pure text, no model, no keys.

Observed on a 55-page document: a heading ("OWL Purpose") with nothing under
it, and a line the model wrote about its own sources ("(Passages do not detail
...)"). Grounding cannot remove either - it deletes claim lines, never
headings, and a statement about absence has no evidence to be judged against.
"""
import pytest

import notes_tidy as nt


# ---------------------------------------------------------------------------
# lines about the passages
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("line", [
    "(Passages do not detail the purpose of OWL.)",
    "- The passages do not mention reasoning.",
    "*The provided context does not cover SPARQL.*",
    "- Inference rules are not covered in the provided text.",
    "- This topic is not mentioned in the passages.",
    "Note: the passages give no dates for this.",
    "Note: The provided text does not include examples.",
    "> The retrieved passages don't describe the syntax.",
])
def test_a_line_about_the_passages_is_removed(line):
    notes = f"## OWL\n- OWL is an ontology language [3].\n{line}\n- It builds on RDF [4].\n"
    out = nt.clean_notes(notes)
    assert line not in out
    assert "- OWL is an ontology language [3]." in out
    assert "- It builds on RDF [4]." in out


@pytest.mark.parametrize("line", [
    "- The passage of the Reform Act did not extend the vote [4].",
    "- The passage of the Reform Act did not extend the vote.",
    "- The context does not determine the meaning of a word.",
    "- The text does not name its narrator.",
    "- Note: sources of energy include coal and gas.",
    "- Water was not found in the provided sample.",
    # Cited, so it is a claim about the source: grounding's job, not ours.
    "- The passages do not rhyme, which marks the shift to prose [7].",
])
def test_a_lookalike_line_is_kept(line):
    notes = f"## Topic\n{line}\n- Another fact [2].\n"
    assert nt.clean_notes(notes) == notes


def test_a_trailing_comment_about_the_passages_is_cut_from_a_real_line():
    notes = ("## OWL\n"
             "- OWL adds classes [3] (the passages do not say which).\n"
             "- It builds on RDF [4]. (Passages do not detail how.)\n"
             "- Reasoners check consistency (for example HermiT) [5].\n")
    assert nt.clean_notes(notes) == ("## OWL\n"
                                     "- OWL adds classes [3].\n"
                                     "- It builds on RDF [4].\n"
                                     "- Reasoners check consistency (for example HermiT) [5].\n")


# ---------------------------------------------------------------------------
# headings with nothing under them
# ---------------------------------------------------------------------------

def test_an_empty_heading_is_removed():
    notes = "## OWL Purpose\n\n## RDF\n- RDF is a graph model [2].\n"
    assert nt.clean_notes(notes) == "## RDF\n- RDF is a graph model [2].\n"


def test_an_empty_heading_at_the_end_is_removed():
    notes = "## RDF\n- RDF is a graph model [2].\n\n## OWL Purpose\n"
    assert "OWL Purpose" not in nt.clean_notes(notes)


def test_a_heading_left_empty_by_a_removed_line_goes_too():
    """The exact pair observed: the comment was the heading's only content."""
    notes = ("## OWL Purpose\n(Passages do not detail the purpose of OWL.)\n\n"
             "## RDF\n- RDF is a graph model [2].\n")
    assert nt.clean_notes(notes) == "## RDF\n- RDF is a graph model [2].\n"


def test_a_parent_heading_with_filled_subsections_is_kept():
    notes = "## OWL\n### Classes\n- A class groups individuals [3].\n"
    assert nt.clean_notes(notes) == notes


def test_a_parent_whose_only_subsection_is_empty_goes_with_it():
    notes = "## OWL\n### Purpose\n\n## RDF\n- RDF is a graph model [2].\n"
    assert nt.clean_notes(notes) == "## RDF\n- RDF is a graph model [2].\n"


def test_an_empty_bold_heading_is_removed():
    notes = "**OWL Purpose**\n\n**RDF**\n- RDF is a graph model [2].\n"
    assert nt.clean_notes(notes) == "**RDF**\n- RDF is a graph model [2].\n"


def test_a_comment_line_inside_a_code_block_is_not_a_heading():
    notes = "## Example\n```python\n# set up\n```\n"
    assert nt.clean_notes(notes) == notes


# ---------------------------------------------------------------------------
# it must be safe to run anywhere, any number of times
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("notes", [
    "Some notes.",
    "content. content. ",
    "## A\n\n\n- one [1].\n\n\n\n- two [2].\n",
    "",
])
def test_notes_with_nothing_to_clean_come_back_untouched(notes):
    assert nt.clean_notes(notes) == notes


def test_cleaning_twice_changes_nothing_more():
    notes = ("## OWL\n### Purpose\n(Passages do not detail the purpose.)\n\n"
             "## RDF\n- RDF is a graph model [2] (not covered in the provided text).\n"
             "**Syntax**\n")
    once = nt.clean_notes(notes)
    assert nt.clean_notes(once) == once
