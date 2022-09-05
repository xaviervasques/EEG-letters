FROM jupyter/scipy-notebook

RUN pip install joblib

COPY train.csv ./train.csv
COPY test.csv ./teset.csv

COPY train.py ./train.py
COPY inference.py ./inference.py

RUN python3 train.py