FROM python:3.9-slim

WORKDIR /usr/src/app

COPY ./requirements.txt /usr/src/app/requirements.txt

RUN /usr/local/bin/python3 -m pip install --upgrade pip

RUN pip3 install --no-cache-dir --upgrade -r /usr/src/app/requirements.txt

COPY . /usr/src/app

EXPOSE 8080