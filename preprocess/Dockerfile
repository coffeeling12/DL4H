FROM python:3.9-slim

RUN apt-get update && apt-get install -y gcc && apt-get install -y git

COPY requirements.txt .
RUN pip install --upgrade -r requirements.txt --no-cache-dir

ENV APP_FOLDER="/app"

COPY *.py $APP_FOLDER/

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=$APP_FOLDER
WORKDIR /

ENTRYPOINT ["python", "/app/preprocess_main.py"]