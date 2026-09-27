"""predict.py end to end: needs a trained model (run split.py and train.py first)."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_sparse_person_gets_a_probability():
    person = {"age": 27, "sex": "female", "earnings": None}
    result = subprocess.run([sys.executable, "src/predict.py"], input=json.dumps(person),
                            capture_output=True, text=True, cwd=ROOT, check=True)
    assert 0 <= float(result.stdout) <= 1
