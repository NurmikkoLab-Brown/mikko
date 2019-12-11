FROM tensorflow/tensorflow:latest-gpu-py3

WORKDIR /home/lstm_rnn

ADD modules/decoders/keras/lstm_rnn/requirements.txt /home/lstm_rnn/requirements.txt
RUN pip install -r /home/lstm_rnn/requirements.txt

COPY modules/decoders/keras/lstm_rnn /home/lstm_rnn

RUN apt-get install -y git && git clone https://github.com/ChrisHeelan/base_dockex.git

ENV PYTHONPATH /home/lstm_rnn

ENTRYPOINT ["python", "-u", "/home/lstm_rnn/lstm_rnn.py"]
