FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir --index-url https://pypi.org/simple -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["uvicorn", "app.web:app", "--host", "0.0.0.0", "--port", "5000", "--reload"]