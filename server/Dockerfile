FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY email_triage_env /app/email_triage_env
COPY openenv.yaml /app/openenv.yaml
COPY README.md /app/README.md

EXPOSE 7860

CMD ["uvicorn", "email_triage_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]

