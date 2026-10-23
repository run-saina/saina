"""A self-hosted Saina server with MOCKED inference for SDK acceptance runs: two deterministic fake models
(tests/test_decisions.py HashScorer, text; OmniHashScorer, text and images) whose probabilities are a hash of
the prompt. It checks wire compatibility only; it says nothing about answer quality.

    python scripts/acceptance/server.py PORT     # API key: k
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tests'))
import uvicorn  # noqa: E402
from saina.server import create_app  # noqa: E402
from test_decisions import HashScorer, OmniHashScorer  # noqa: E402

if __name__ == '__main__':
    app = create_app(scorers=[HashScorer(), OmniHashScorer()], api_key='k')
    uvicorn.run(app, host='127.0.0.1', port=int(sys.argv[1]), log_level='warning')
