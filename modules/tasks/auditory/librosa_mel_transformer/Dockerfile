FROM continuumio/miniconda3:4.7.12

WORKDIR /home/librosa_mel_transformer

ADD modules/tasks/auditory/librosa_mel_transformer/requirements.txt /home/librosa_mel_transformer/requirements.txt
RUN conda config --add channels conda-forge && \
    conda config --set channel_priority strict && \
    conda install --file /home/librosa_mel_transformer/requirements.txt

COPY modules/tasks/auditory/librosa_mel_transformer /home/librosa_mel_transformer

RUN git clone https://github.com/ChrisHeelan/base_dockex.git

ENV PYTHONPATH /home/librosa_mel_transformer

ENTRYPOINT ["python", "-u", "/home/librosa_mel_transformer/librosa_mel_transformer.py"]
