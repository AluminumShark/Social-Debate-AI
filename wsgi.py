"""
WSGI entry point for production servers (gunicorn).

Usage:
  gunicorn -k gthread -w 2 --threads 8 -t 0 -b 0.0.0.0:5000 wsgi:app

Threaded workers (gthread) + timeout 0 are used so long-lived SSE streams
(/api/debate/stream) are not killed mid-debate.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ui.app import app, initialize_system  # noqa: E402

# Initialize orchestrators/config at import so each gunicorn worker is ready.
initialize_system()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
