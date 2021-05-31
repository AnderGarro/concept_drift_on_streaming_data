# This file is a template, and might need editing before it works on your project. FROM 
#nvidia/cuda:10.2-devel-ubuntu18.04 FROM nvidia/cuda:10.1-base-ubuntu18.04 FROM nvidia/cuda:10.0-base-ubuntu18.04 
#FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
#install required libraries - the rm- rf command cleans the package cache the following RUN Block is an example, 
#you should replace it

ENV TZ=Europe/Madrid 
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

#python and R
RUN apt-get update \
  && apt-get install -y \
  apt-utils \
  python3-pip \
  python3-dev \
  r-base \
  libcurl4-openssl-dev \
  libssl-dev \
  locales 

RUN apt-get install nano -y

RUN pip3 install bencode-python3 
RUN pip3 install --upgrade pip

#RStudio
RUN apt-get update \
  && apt-get install -y wget 
RUN apt-get install -y gdebi-core 
RUN wget https://download2.rstudio.org/server/trusty/amd64/rstudio-server-1.2.5033-amd64.deb 
RUN gdebi rstudio-server-1.2.5033-amd64.deb 
RUN pip3 install setuptools==41.0.0

#python libraries for machine learning
RUN pip3 install \
    tensorflow \
    torch \
    jupyter \
    matplotlib \
    h5py \
    scipy \
    pandas \
    scikit-learn \
    seaborn \
    deap \
    imblearn \
    distributed \
    blosc \
    lz4 \
    pickle5 \
    xgboost \
    keras \
    mxnet==1.6.0 \
    mlxtend \
    psycopg2 \
    statsmodels

RUN pip3 install plotly
RUN pip3 install dask
RUN pip3 install H2O
RUN pip3 install mlflow

RUN echo "LC_ALL=en_US.UTF-8" >> /etc/environment \
    echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen \
    echo "LANG=en_US.UTF-8" > /etc/locale.conf \
    locale-gen en_US.UTF-8


ENV AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
ENV AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
ENV MLFLOW_S3_ENDPOINT_URL=http://172.16.56.53:9000
ENV MLFLOW_TRACKING_URI=http://172.16.56.53:80

#RUN apt install git -y

#clean the package cache
RUN rm -rf /var/lib/apt/lists/*

#SHELL ["/bin/bash", "-c"]
ENV SHELL=/bin/bash 

WORKDIR /notebooks 
ENTRYPOINT ["jupyter", "notebook", "--ip=0.0.0.0","--NotebookApp.token='deepikerlan'","--allow-root"]
#jupyter notebook --ip 0.0.0.0 --NotebookApp.token='deepikerlan' --allow-root
