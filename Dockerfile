FROM python:3.9

WORKDIR /opt/tg
COPY bot/ bot/
COPY models/ models/

COPY requirements.txt .
RUN pip install -r requirements.txt

CMD ["python", "-m", "bot"]