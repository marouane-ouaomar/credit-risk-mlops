FROM python:3.11-slim

WORKDIR /code

# Install deps first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code + pre-trained model artifacts
# (models/ must contain model.pkl, encoder.pkl, explainer.pkl, metrics.json,
#  feature_list.json — generated locally by `python -m src.train` before
#  building the image; this project does not train at container startup)
COPY app/ app/
COPY src/ src/
COPY models/ models/

# Ensure the SQLite log directory exists and is writable
RUN mkdir -p data/logs

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
