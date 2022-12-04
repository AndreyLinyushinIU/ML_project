FROM python:3.9

WORKDIR /opt/tg

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY bot/ bot/
COPY DeepPhotoStyle_pytorch DeepPhotoStyle_pytorch/
COPY models/ models/

COPY requirements.txt .
RUN pip install -r requirements.txt

CMD ["python", "-m", "bot"]