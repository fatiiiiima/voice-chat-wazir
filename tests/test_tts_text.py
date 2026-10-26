import pytest

from agent.tts_kokoro import EarlyClauseAggregator, speakable
from pipecat.utils.text.base_text_aggregator import TextType


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Welcome to AGMC.", "Welcome to A-G-M-C."),
        ("Is your number 0502635478?", "Is your number 0 5 0, 2 6 3, 5 4 7 8?"),
        ("Is it 050 263 5478?", "Is it 0 5 0, 2 6 3, 5 4 7 8?"),
        ("Call +971 50 263 5478.", "Call plus, 9 7 1, 5 0, 2 6 3, 5 4 7 8."),
        ("Your reference is TD-45TML.", "Your reference is T D, 4 5 T M L."),
        ("The 2021 BMW X5", "The twenty twenty-one BMW X5"),
        ("AED 145,000 for 48,000 km", "AED 145,000 for 48,000 km"),
        ("Tuesday at 17:00", "Tuesday at 17:00"),
    ],
)
def test_speakable(text, expected):
    assert speakable(text) == expected


async def collect(agg, chunks):
    out = []
    for c in chunks:
        out += [a.text async for a in agg.aggregate(c)]
    return out


async def test_first_chunk_cut_at_comma_only_once():
    agg = EarlyClauseAggregator(aggregation_type=TextType.SENTENCE)
    tokens = ["Sure", ", we", " have", " the", " X3", " at", " Motor", " City", ", for", " one", " hundred",
              ", and", " it", " is", " great", ". Want", " a", " test", " drive", "?"]
    out = await collect(agg, tokens)
    assert out[0] == "Sure, we have the X3 at Motor City,"
    # Later commas don't cut; the rest of the first sentence stays together.
    assert out[1] == "for one hundred, and it is great."
    assert (await agg.flush()).text == "Want a test drive?"


async def test_opener_sentence_emits_first_and_numbers_not_split():
    agg = EarlyClauseAggregator(aggregation_type=TextType.SENTENCE)
    out = await collect(agg, ["Sure", "!", " It", " costs", " AED", " 145", ",", "000", ", which", " is", " fair", "."])
    assert out == ["Sure!"]
    assert (await agg.flush()).text == "It costs AED 145,000, which is fair."
    # Next reply gets an early cut again.
    out = await collect(agg, ["Okay", " so", " the", " car", ", is", " ready"])
    assert out == ["Okay so the car,"]
