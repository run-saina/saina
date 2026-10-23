"""Browser-test fixture only. Never a production model fallback."""
import sys
from pathlib import Path
import uvicorn
from saina.local import create_local_app
from test_server import FakeScorer
scorer = FakeScorer()
scorer.model_id = 'saina-helm-0.8b'
port = int(sys.argv[1])
app = create_local_app(scorer, 'browser-test-token', Path(__file__).parents[1]/'src/saina/ui', port=port)
uvicorn.run(app, host='127.0.0.1', port=port, log_level='warning')
