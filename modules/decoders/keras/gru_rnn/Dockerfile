FROM tensorflow/tensorflow:latest-gpu-py3

WORKDIR /home/gru_rnn

ADD modules/decoders/keras/gru_rnn/requirements.txt /home/gru_rnn/requirements.txt
RUN pip install -r /home/gru_rnn/requirements.txt

COPY modules/decoders/keras/gru_rnn /home/gru_rnn

RUN apt-get update
RUN apt-get install -y git && \
    git clone https://github.com/ChrisHeelan/base_dockex.git && \
    git -C /home/gru_rnn/base_dockex checkout 913c732d12e647782b5a3628cb6529147611f171

ENV PYTHONPATH /home/gru_rnn

ENTRYPOINT ["python", "-u", "/home/gru_rnn/gru_rnn.py"]
