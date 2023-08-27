FROM python:3.10

WORKDIR /code

RUN apt-get update && apt-get install libasound-dev libportaudio2 libportaudiocpp0 portaudio19-dev espeak -y

COPY requirements.txt .
RUN /usr/local/bin/python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt


COPY src/ /code/
COPY data/wakeword/ /code/
COPY .env /code/

CMD ["python", "main.py"]