FROM jupyter/scipy-notebook

WORKDIR /mydata

RUN pip install joblib

COPY train.csv ./train.csv
COPY test.csv ./test.csv

COPY train.py ./train.py
COPY inference.py ./inference.py

RUN python3 train.py