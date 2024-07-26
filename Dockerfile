FROM  ubuntu

WORKDIR /thinh

RUN apt-get update
RUN apt-get -y install python3-sklearn python3-sklearn-lib

RUN apt-get -y install python3-pip
RUN pip install pandas --break-system-packages

COPY train_fastercnn.py ./cls.py