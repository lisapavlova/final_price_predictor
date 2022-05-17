from ubuntu:20.04
MAINTAINER Elizaveta Pavlova
RUN apt-get update -y && \
	apt install -y python3-pip
COPY requirements.txt requirements.txt 
RUN pip3 install -r requirements.txt
COPY . .
CMD python3 app.py
