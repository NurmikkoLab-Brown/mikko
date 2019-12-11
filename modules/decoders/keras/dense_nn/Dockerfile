FROM tensorflow/tensorflow:latest-gpu-py3

WORKDIR /home/dense_nn

ADD modules/decoders/keras/dense_nn/requirements.txt /home/dense_nn/requirements.txt
RUN pip install -r /home/dense_nn/requirements.txt

COPY modules/decoders/keras/dense_nn /home/dense_nn

RUN apt-get install -y git && git clone https://github.com/ChrisHeelan/base_dockex.git

ENV PYTHONPATH /home/dense_nn

ENTRYPOINT ["python", "-u", "/home/dense_nn/dense_nn.py"]
