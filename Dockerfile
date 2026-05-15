FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ccppm ./ccppm
COPY app.py .
ENV PYTHONPATH=/app

EXPOSE 8092
CMD ["python", "/app/app.py"]
