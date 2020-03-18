FROM continuumio/miniconda3:4.7.12

WORKDIR /home/regression_metrics

ADD modules/analysis/regression_metrics/requirements.txt /home/regression_metrics/requirements.txt
RUN conda install --file /home/regression_metrics/requirements.txt

COPY modules/analysis/regression_metrics /home/regression_metrics

RUN git clone https://github.com/ChrisHeelan/base_dockex.git && \
    git -C /home/regression_metrics/base_dockex checkout 913c732d12e647782b5a3628cb6529147611f171

ENV PYTHONPATH /home/regression_metrics

ENTRYPOINT ["python", "-u", "/home/regression_metrics/regression_metrics.py"]
