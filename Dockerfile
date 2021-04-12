# Our docker image code from the efficientQA competition
FROM nvidia/cuda:10.2-base-centos8
MAINTAINER Martin Fajcik <ifajcik@fit.vutbr.cz>

COPY requirements.txt .
# install and compress
RUN yum install -y python36 \
    && python3.6 -m pip install --no-cache-dir  torch==1.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html  \
    && python3.6 -m pip install --no-cache-dir -r requirements.txt \
    # TAR.GZ is used as ZIP is not preinstalled natively
    && tar czf /usr/local/lib64/python3.6/python36_size_packages.tgz /usr/local/lib64/python3.6/site-packages \
    && rm -r /usr/local/lib64/python3.6/site-packages

# Add code
COPY .checkpoints .checkpoints/
COPY .index .index/
COPY .Transformers_cache .Transformers_cache/
COPY configurations configurations/
COPY scalingQA scalingQA
COPY prediction.py .
COPY run_prediction.py .
COPY submission.sh .
