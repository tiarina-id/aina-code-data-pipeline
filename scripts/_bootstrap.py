from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
sys.path[:] = [path for path in sys.path if Path(path or ".").resolve() != SCRIPTS]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
