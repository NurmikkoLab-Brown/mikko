FROM continuumio/miniconda3:4.7.12

WORKDIR /home/wiener_filter

ADD modules/decoders/wiener_filter/requirements.txt /home/wiener_filter/requirements.txt
RUN conda install --file /home/wiener_filter/requirements.txt

COPY modules/decoders/wiener_filter /home/wiener_filter

RUN git clone https://github.com/ChrisHeelan/base_dockex.git && \
    git -C /home/wiener_filter/base_dockex checkout 913c732d12e647782b5a3628cb6529147611f171

ENV PYTHONPATH /home/wiener_filter

ENTRYPOINT ["python", "-u", "/home/wiener_filter/wiener_filter.py"]
