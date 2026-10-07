"""Deterministic clean-up of finished notes: pure text, no model, no keys.

Observed on a 55-page document: a heading ("OWL Purpose") with nothing under
it, and a line the model wrote about its own sources ("(Passages do not detail
...)"). Grounding cannot remove either - it deletes claim lines, never
headings, and a statement about absence has no evidence to be judged against.
"""
import re

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


# ===========================================================================
# merge pass: long documents are written a window at a time and joined
# ===========================================================================
# Observed on the same document: the same topic under separate headings, and
# sections out of page order in the first half. Nothing merged the windows.

def _first_citations(notes):
    """The first passage each '##' section cites, top to bottom."""
    out = []
    for section in notes.split("\n## ")[0 if notes.startswith("## ") else 1:]:
        m = nt.CITATION_RE.search(section)
        if m:
            out.append(int(m.group(1)))
    return out


def test_sections_with_the_same_heading_become_one():
    notes = ("## RDF\n- RDF stores facts as triples [1].\n\n"
             "## OWL\n- OWL adds classes on top of RDF [4].\n\n"
             "## RDF\n- A triple has a subject, a predicate and an object [2].\n")
    assert nt.merge_sections(notes) == (
        "## RDF\n- RDF stores facts as triples [1].\n"
        "- A triple has a subject, a predicate and an object [2].\n\n"
        "## OWL\n- OWL adds classes on top of RDF [4].")


@pytest.mark.parametrize("first, second", [
    ("## Overview of OWL", "## OWL Overview"),
    ("## Key Concepts", "## Key concepts:"),
    ("### 1. Reasoning", "### 3. Reasoning"),
    ("**RDF Triples**", "**RDF triples**"),
])
def test_near_same_headings_become_one(first, second):
    notes = f"{first}\n- First fact about it [1].\n\n{second}\n- Second fact about it [2].\n"
    out = nt.merge_sections(notes)
    assert first in out and second not in out
    assert "- First fact about it [1].\n- Second fact about it [2]." in out


def test_different_headings_stay_separate():
    notes = ("## OWL Classes\n- A class groups individuals [1].\n\n"
             "## OWL Properties\n- A property links individuals [2].\n")
    assert nt.merge_sections(notes) == notes


def test_paragraph_bodies_are_joined_as_separate_paragraphs():
    notes = "## RDF\nRDF is a graph model [1].\n\n## RDF\nIt stores triples [2].\n"
    assert nt.merge_sections(notes) == (
        "## RDF\nRDF is a graph model [1].\n\nIt stores triples [2].")


def test_a_repeated_bullet_is_dropped_and_its_citation_kept():
    notes = ("## RDF\n- RDF stores facts as subject, predicate, object triples [1].\n\n"
             "## Storage\n- An index speeds up lookups by subject [5].\n"
             "- **RDF** stores facts as subject–predicate–object triples [9].\n")
    out = nt.merge_sections(notes)
    assert out.count("RDF stores facts") == 1
    assert "- RDF stores facts as subject, predicate, object triples [1][9]." in out
    assert "- An index speeds up lookups by subject [5]." in out


@pytest.mark.parametrize("one, two", [
    # one letter apart, and both true
    ("- Process A uses 2 ATP in every full cycle [5].",
     "- Process B uses 2 ATP in every full cycle [5]."),
    # a different number is a different fact
    ("- The reaction releases 36 molecules of ATP in total [5].",
     "- The reaction releases 38 molecules of ATP in total [6]."),
    # opposite claims must both stay visible
    ("- Mitosis does not occur in the mature neurons of the adult brain [2].",
     "- Mitosis does occur in the mature neurons of the adult brain [3]."),
    # same words, other direction
    ("- Enzyme alpha activates enzyme beta in the second stage [4].",
     "- Enzyme beta activates enzyme alpha in the second stage [4]."),
])
def test_distinct_bullets_that_look_alike_both_stay(one, two):
    notes = f"## Topic\n{one}\n{two}\n"
    assert nt.merge_sections(notes) == notes


def test_a_short_repeated_bullet_is_left_alone():
    """Too few words to be sure two bullets say the same thing."""
    notes = "## A\n- See above [1].\n\n## B\n- See above [2].\n"
    assert nt.merge_sections(notes) == notes


def test_sections_are_put_in_the_order_of_the_pages_they_cite():
    notes = ("## Inference\n- Rules derive new facts [12].\n\n"
             "## Triples\n- A triple has three parts [3].\n\n"
             "## Classes\n- A class groups individuals [7].\n")
    out = nt.merge_sections(notes)
    assert _first_citations(out) == [3, 7, 12]
    assert out.index("## Triples") < out.index("## Classes") < out.index("## Inference")


def test_a_section_with_no_citation_stays_behind_the_one_before_it():
    notes = ("## Inference\n- Rules derive new facts [12].\n\n"
             "## Worked example\n- Take a rule and apply it once.\n\n"
             "## Triples\n- A triple has three parts [3].\n")
    out = nt.merge_sections(notes)
    assert out.index("## Triples") < out.index("## Inference") < out.index("## Worked example")


def test_subsections_travel_with_their_parent_and_are_ordered_inside_it():
    notes = ("## OWL\n### Properties\n- A property links individuals [9].\n"
             "### Classes\n- A class groups individuals [8].\n\n"
             "## RDF\n- RDF is a graph model [2].\n")
    out = nt.merge_sections(notes)
    assert out.index("## RDF") < out.index("## OWL") < out.index("### Classes") \
        < out.index("### Properties")


def test_text_before_the_first_heading_stays_first():
    notes = "An opening line.\n\n## B\n- late [9].\n\n## A\n- early [1].\n"
    out = nt.merge_sections(notes)
    assert out.startswith("An opening line.\n\n## A")


@pytest.mark.parametrize("notes", [
    "Some notes.",
    "content. content. ",
    "## A\n\n\n- one fact about the first topic [1].\n\n\n\n## B\n- another fact [2].\n",
    "## A\n```python\nx = rows[1]\n```\n\n## B\n- a fact [2].\n",
    "",
])
def test_notes_already_in_order_come_back_untouched(notes):
    assert nt.merge_sections(notes) == notes


def test_merging_twice_changes_nothing_more():
    notes = ("## Inference\n- Rules derive new facts from old ones [12].\n\n"
             "## RDF\n- RDF stores facts as subject, predicate, object triples [1].\n\n"
             "## Inference\n- Rules derive new facts from old ones [3].\n"
             "- A reasoner applies the rules until nothing changes [13].\n\n"
             "## rdf\n- RDF stores facts as subject, predicate, object triples [9].\n")
    once = nt.merge_sections(notes)
    assert nt.merge_sections(once) == once


def test_tidy_removes_a_heading_emptied_by_the_merge():
    notes = ("## RDF\n- RDF stores facts as subject, predicate, object triples [1].\n\n"
             "## Recap\n- RDF stores facts as subject, predicate, object triples [1].\n")
    assert nt.tidy(notes, merge=True) == (
        "## RDF\n- RDF stores facts as subject, predicate, object triples [1].")


def test_tidy_without_merge_only_cleans():
    notes = "## B\n- late [9].\n\n## A\n- early [1].\n\n## Empty\n"
    assert nt.tidy(notes) == "## B\n- late [9].\n\n## A\n- early [1]."


# ---------------------------------------------------------------------------
# the app's own bullet mark
# ---------------------------------------------------------------------------
# The writer is told to start bullets with "• " (agent._format_instructions),
# and these passes only knew "-", "*", "+" and numbers. On the first real
# 55-page run the repeated-bullet check looked at 7 of 163 bullets.

def test_the_bullet_the_writer_is_told_to_use_is_recognised():
    import agent
    told = re.search(r"Start each bullet with `(.+?)`", agent._format_instructions("bullet"))
    assert told, "the bullet instruction changed shape; update this test with it"
    assert nt._BULLET.match(told.group(1) + "a point about the topic [1]."), (
        f"the writer is told to use {told.group(1)!r} and the merge pass cannot see it")
    numbered = re.search(r"Start each point with `(.+?)`", agent._format_instructions("numbered"))
    assert nt._BULLET.match(numbered.group(1) + " a point about the topic [1].")


def test_a_repeated_dot_bullet_is_dropped_and_its_citation_kept():
    notes = ("**RDF**\n• RDF stores facts as subject, predicate, object triples [1].\n\n"
             "**Storage**\n• An index speeds up lookups by subject [5].\n"
             "• RDF stores facts as subject, predicate, object triples [9].\n")
    out = nt.merge_sections(notes)
    assert out.count("RDF stores facts") == 1
    assert "• RDF stores facts as subject, predicate, object triples [1][9]." in out


def test_merged_sections_of_dot_bullets_read_as_one_list():
    notes = ("**RDF**\n• First fact about it [1].\n\n"
             "**RDF:**\n• Second fact about it [2].\n")
    assert nt.merge_sections(notes) == (
        "**RDF**\n• First fact about it [1].\n• Second fact about it [2].")


def test_a_dot_bullet_about_the_passages_is_removed():
    notes = ("**OWL**\n• OWL is an ontology language [3].\n"
             "• The passages do not mention reasoning.\n")
    assert nt.clean_notes(notes) == "**OWL**\n• OWL is an ontology language [3].\n"


# ===========================================================================
# cut-off text: a window that stopped in the middle of a bullet
# ===========================================================================
# Observed on the same document: two bullets ending mid-word. The stream ended
# without an error and nothing looked at the last line.

_FULL = ("## Triples\n"
         "- A triple has a subject, a predicate and an object [1].\n"
         "- The subject names the thing described [1].\n"
         "- The predicate names the relation [2].\n"
         "- The object is the value or the other thing [2].\n")


def test_text_that_ends_on_a_whole_bullet_is_not_a_fragment():
    assert nt.trailing_fragment(_FULL) is False
    assert nt.drop_fragment(_FULL) == _FULL


def test_a_last_bullet_that_stops_mid_word_is_a_fragment():
    text = _FULL + "- Literals carry a datatype such as xsd:inte"
    assert nt.trailing_fragment(text) is True
    assert nt.drop_fragment(text).rstrip() == _FULL.rstrip()


def test_unclosed_bold_on_the_last_line_is_a_fragment():
    assert nt.trailing_fragment("- **Triple**: three parts [1].\n- **Predic") is True


def test_a_citation_after_the_full_stop_still_ends_the_bullet():
    assert nt.trailing_fragment(_FULL + "- Literals carry a datatype. [3]\n") is False
    assert nt.trailing_fragment(_FULL + "- **Literals carry a datatype [3].**\n") is False


@pytest.mark.parametrize("text", [
    "surviving window content",                      # one line: nothing to compare with
    "UNIQUEMARKER partial text ",
    "- a point\n- another point\n- a third point\n- a fourth point",  # no full stops anywhere
    _FULL + "\n## Literals\n",                       # a bare heading is the cleaner's job
    "",
])
def test_text_that_only_might_be_cut_off_is_left_alone(text):
    assert nt.trailing_fragment(text) is False
    assert nt.drop_fragment(text) == text


def test_when_the_stream_is_known_to_be_cut_the_last_line_is_dropped():
    """No full stops anywhere, so style proves nothing - but the provider said
    it stopped at the token cap, and the text does not end on a line break."""
    text = "- a point\n- another point\n- a third poi"
    assert nt.trailing_fragment(text, cut=True) is True
    assert nt.drop_fragment(text, cut=True) == "- a point\n- another point\n"
    # cut exactly at a line break: the last line is whole
    assert nt.trailing_fragment("- a point\n- another point\n", cut=True) is False
