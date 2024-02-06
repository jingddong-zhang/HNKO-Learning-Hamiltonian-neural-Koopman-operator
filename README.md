
![image](https://github.com/jingddong-zhang/HNKO/blob/main/logo.jpg)

# [PRR] Learning Hamiltonian neural Koopman operator and simultaneously sustaining and discovering conservation laws
This repository contains the code for the paper: *Learning Hamiltonian neural Koopman operator and simultaneously sustaining and discovering conservation law* by Jingdong Zhang, Qunxi Zhu, and Wei Lin.

### Introduction

A machine learning framework, equipped with a unitary Koopman structure, is designed to reconstruct Hamiltonian systems using either noise-perturbed or partially observational data. This framework can discover conservation laws and scale effectively to physical models even with hundreds and thousands of freedoms. Specifically, the framework is comprised of an __auto-encoder__ with latent space being an high dimensional sphere, and a __neural unitary Koopman operator__ constructed by the Lie exponent map of neural network.


![image](https://github.com/jingddong-zhang/HNKO/blob/main/HNKO_sketch.png)


# Installation
Please download the packages in the **requirements.txt** file.

# Data
The data of HNKO_ast is provided in the [Google Drive](https://drive.google.com/file/d/1_4_n5GAD2jS-SqP-enf8S-5cI781qTZu/view?usp=sharing)

# Usage
The newly added directory **Model in replys** contains the reproduced python code of CNN-LSTM, Hamiltonian ODE graph networks (HOGN) and reservoir computing.
For a standard comparison with these models, we apply the model structures in [CNN-LSTM](https://github.com/ozancanozdemir/CNN-LSTM), [graph-neural-ode](https://github.com/jaketae/graph-neural-ode/tree/master), [RC](https://github.com/zhuqunxi/RC_Lorenz).

The **hnko_feature.py** documents in directory **threeboy** and **kepler** are used to discover the Hamiltonians.

# Acknowledgement
Authors appreciate Phoenix, a talented artist, for designing the logo of [Research Institute of Intelligent Complex Systems](https://iics.fudan.edu.cn/main.htm).

# Citation
Please cite the paper
