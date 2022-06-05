FROM jupyter/scipy-notebook

RUN mkdir my-model
ENV MODEL_DIR=/home/jovyan/my-model
ENV LOCAL_PATH=/home/jovyan/data
ENV MODEL_FILE_LDA=clf_lda.joblib
ENV MODEL_FILE_NN=clf_nn.joblib

RUN pip install joblib

#COPY train.csv ./train.csv
#COPY test.csv ./test.csv

COPY train.py ./train.py
#COPY inference.py ./inference.py



#從本地資料夾內copy run.sh檔案
COPY run.sh ./
ENTRYPOINT ["/bin/bash", "./run.sh"]
