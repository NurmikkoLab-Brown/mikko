FROM tensorflow/tensorflow:latest-gpu-py3

WORKDIR /home/lstm_rnn

ADD modules/decoders/keras/lstm_rnn/requirements.txt /home/lstm_rnn/requirements.txt
RUN pip install -r /home/lstm_rnn/requirements.txt

COPY modules/decoders/keras/lstm_rnn /home/lstm_rnn

RUN apt-get update
RUN apt-get install -y git && \
    git clone https://github.com/ChrisHeelan/base_dockex.git && \
    git -C /home/lstm_rnn/base_dockex checkout 913c732d12e647782b5a3628cb6529147611f171

ENV PYTHONPATH /home/lstm_rnn

ENTRYPOINT ["python", "-u", "/home/lstm_rnn/lstm_rnn.py"]
