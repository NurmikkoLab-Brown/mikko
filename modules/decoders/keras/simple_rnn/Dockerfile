FROM tensorflow/tensorflow:latest-gpu-py3

WORKDIR /home/simple_rnn

ADD modules/decoders/keras/simple_rnn/requirements.txt /home/simple_rnn/requirements.txt
RUN pip install -r /home/simple_rnn/requirements.txt

COPY modules/decoders/keras/simple_rnn /home/simple_rnn

RUN apt-get update
RUN apt-get install -y git && \
    git clone https://github.com/ChrisHeelan/base_dockex.git && \
    git -C /home/simple_rnn/base_dockex checkout 913c732d12e647782b5a3628cb6529147611f171

ENV PYTHONPATH /home/simple_rnn

ENTRYPOINT ["python", "-u", "/home/simple_rnn/simple_rnn.py"]
