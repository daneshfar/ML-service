# tests/conftest.py
import sys
from pathlib import Path

# Project root (one level above tests/)
ROOT = Path(__file__).resolve().parents[1]

# Add src/ to sys.path so `import ml_service` works
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
