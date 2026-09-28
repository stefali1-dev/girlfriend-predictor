PYTHON ?= .venv/bin/python

.PHONY: data all test

data:
	$(PYTHON) src/get_nhanes.py
	$(PYTHON) src/get_hcmst.py
	@test -f data/raw/nlsy97/partner-model.csv || echo "NLSY97 is a manual download: see README.md"

all:
	$(PYTHON) src/clean_nlsy97.py
	$(PYTHON) src/clean_nhanes.py
	$(PYTHON) src/clean_hcmst.py
	$(PYTHON) src/split.py
	$(PYTHON) src/train.py
	$(PYTHON) src/evaluate.py
	$(PYTHON) src/explain.py
	$(PYTHON) src/forecast.py
	$(PYTHON) src/outside_check.py

test:
	$(PYTHON) -m pytest -q tests
