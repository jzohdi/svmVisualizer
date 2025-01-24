FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set Python to unbuffered mode to ensure real-time logs
ENV PYTHONUNBUFFERED=1

# Expose the port FastAPI runs on
EXPOSE 8000

CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]