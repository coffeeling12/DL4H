FROM python:3.9-slim

RUN apt-get update && apt-get install -y gcc && apt-get install -y git

COPY requirements.txt .
RUN pip install --upgrade -r requirements.txt --no-cache-dir

ENV APP_FOLDER="/app"

COPY trainers/ $APP_FOLDER/trainers/
COPY utils/ $APP_FOLDER/utils/
COPY modules/ $APP_FOLDER/modules/
COPY models/ $APP_FOLDER/models/
COPY datasets/ $APP_FOLDER/datasets/
COPY main.py $APP_FOLDER/

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=$APP_FOLDER
WORKDIR /

ENTRYPOINT ["python", "/app/main.py"]