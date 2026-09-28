PYTHON ?= .venv/bin/python
CDK = cd infra && BUILDX_NO_DEFAULT_ATTESTATIONS=1 npx aws-cdk@2.1143.0

.PHONY: data all test bootstrap deploy destroy

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
	$(PYTHON) sagemaker/typical.py

test:
	$(PYTHON) -m pytest -q tests

# The hosted model for the web page (infra/app.py). Needs infra/.venv with infra/requirements.txt.
bootstrap:
	$(CDK) bootstrap

deploy:
	$(CDK) diff
	$(CDK) deploy

destroy:
	$(CDK) destroy
