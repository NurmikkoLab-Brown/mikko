FROM tensorflow/tensorflow:latest-gpu-py3

WORKDIR /home/simple_rnn

ADD modules/decoders/keras/simple_rnn/requirements.txt /home/simple_rnn/requirements.txt
RUN pip install -r /home/simple_rnn/requirements.txt

COPY modules/decoders/keras/simple_rnn /home/simple_rnn

RUN apt-get install -y git && git clone https://github.com/ChrisHeelan/base_dockex.git

ENV PYTHONPATH /home/simple_rnn

ENTRYPOINT ["python", "-u", "/home/simple_rnn/simple_rnn.py"]
