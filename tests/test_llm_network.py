"""Real Groq smoke test. Needs internet and a filled-in .env.

Run it with:  python -m pytest -m network tests/test_llm_network.py -v
"""

import pytest

from docqa.config import load_settings
from docqa.llm import build_llm


@pytest.mark.network
def test_real_groq_round_trip():
    llm = build_llm(load_settings())
    reply = llm.complete("Reply with one word.", "Say ready.")
    assert reply.strip()
