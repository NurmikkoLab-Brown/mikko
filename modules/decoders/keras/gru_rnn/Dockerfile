FROM tensorflow/tensorflow:latest-gpu-py3

WORKDIR /home/gru_rnn

ADD modules/decoders/keras/gru_rnn/requirements.txt /home/gru_rnn/requirements.txt
RUN pip install -r /home/gru_rnn/requirements.txt

COPY modules/decoders/keras/gru_rnn /home/gru_rnn

RUN apt-get install -y git && git clone https://github.com/ChrisHeelan/base_dockex.git

ENV PYTHONPATH /home/gru_rnn

ENTRYPOINT ["python", "-u", "/home/gru_rnn/gru_rnn.py"]
