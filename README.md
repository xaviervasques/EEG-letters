# Build and Run a Docker Container for your Machine Learning Model

The idea is to do a quick and easy build of a Docker container with a simple machine learning model and run it. In order to start building a Docker container for a machine learning model, let’s consider three files: 
-	Dockerfile
-	train.py
-	inference.py

The train.py is  a python script that ingest and normalize EEG data in a csv file (train.csv) and train two models to classify the data (using scikit-learn). The script saves two models: Linear Discriminant Analysis (clf_lda) and Neural Networks multi-layer perceptron (clf_NN). 

The inference.py will be called to perform batch inference by loading the two models that has been previously created. The application will normalize new EEG data coming from a csv file (test.csv), perform inference on the dataset and print the classification accuracy and predictions. 

Read the complete post:  https://xaviervasques.medium.com/build-and-run-a-docker-container-for-your-machine-learning-model-60209c2d7a7f
