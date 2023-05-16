# KGE-TGI
A graph-based model for predicting TF-target gene interactions and their types.

# Overview

- `data/` contains the necessary dataset files;
- `main.py` main function for KGE-TGI.
- `utils.py` contains functions for loading data, constructing heterogeneous graphs, and more.
- `train.py` contains functions for traing model.
- `model.py` contains the implementation of the KGE-TGI model and other related code. 

# Requirment
This project is built on the following library versions:

- python==3.7
- numpy=1.21.5
- pytorch=1.8.1=py3.7_cuda10.2_cudnn7.6.5_0
- dgl-cu102==0.6.1

Use`conda env create -f env.yaml` to set up the environment.

# Work in process
Use the command `python main.py` to run the KGE-TGI model based on the provided data. 
However, due to the combination of multiple biological network information, if you want to use your own data, data calibration and matching are currently required. 
We will update the code as soon as possible to make it more user-friendly, allowing users to derive transcriptional regulatory relationships using their own data.
