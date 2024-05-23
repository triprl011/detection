FROM python:latest

LABEL Maintainer="grebnev.vitalii"

WORKDIR /usr/app/src

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY . .

RUN python3 -m pip install --upgrade pip wheel
RUN pip install -r requirements.txt

CMD [ "python", "./detection_PR_curve.py"]