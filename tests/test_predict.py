"""predict.py end to end: needs a trained model (run split.py and train.py first)."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def predict(person):
    result = subprocess.run([sys.executable, "src/predict.py"], input=json.dumps(person),
                            capture_output=True, text=True, cwd=ROOT, check=True)
    return float(result.stdout)


def test_sparse_person_gets_a_probability():
    assert 0 <= predict({"age": 27, "sex": "female", "earnings": None}) <= 1


def test_null_category_is_the_same_as_an_omitted_one():
    assert predict({"age": 27, "sex": "female", "religion": None}) == predict({"age": 27, "sex": "female"})
