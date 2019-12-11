FROM continuumio/miniconda3:4.7.12

WORKDIR /home/create_window_features

ADD modules/preprocessing/create_window_features/requirements.txt /home/create_window_features/requirements.txt
RUN conda install --file /home/create_window_features/requirements.txt

COPY modules/preprocessing/create_window_features /home/create_window_features

RUN git clone https://github.com/ChrisHeelan/base_dockex.git

ENV PYTHONPATH /home/create_window_features

ENTRYPOINT ["python", "-u", "/home/create_window_features/create_window_features.py"]
