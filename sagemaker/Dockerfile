# Built from the repo root: docker build -f sagemaker/Dockerfile -t girlfriend-predictor .
# Python and library versions must match the ones that pickled models/partnered.joblib.
FROM python:3.14-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -c requirements.txt catboost scikit-learn pandas numpy

COPY src/predict.py src/
COPY sagemaker/form.py sagemaker/serve.py sagemaker/
COPY models/partnered.joblib models/typical.json models/

ENTRYPOINT ["python", "sagemaker/serve.py"]
