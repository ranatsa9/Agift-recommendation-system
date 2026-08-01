"""Vercel entry point for the Giftly FastAPI service."""
from pathlib import Path
import sys


# Vercel executes this file from inside /api. Add the repository root so the
# shared API and model modules can always be imported.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api_main import app  # noqa: E402


handler = app
