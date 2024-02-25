
![image](https://github.com/jingddong-zhang/HNKO/blob/main/logo.jpg)

# [PRR] Learning Hamiltonian neural Koopman operator and simultaneously sustaining and discovering conservation laws
This repository contains the code for the paper: *Learning Hamiltonian neural Koopman operator and simultaneously sustaining and discovering conservation law* by [Jingdong Zhang](https://scholar.google.cz/citations?user=Bjo3nfwAAAAJ&hl=zh-CN&oi=ao), [Qunxi Zhu](https://scholar.google.cz/citations?user=45oFQD4AAAAJ&hl=zh-CN&oi=ao), and [Wei Lin](https://faculty.fudan.edu.cn/wlin/zh_CN/zdylm/652034/list/index.htm).

### Introduction

A machine learning framework, equipped with a unitary Koopman structure, is designed to reconstruct Hamiltonian systems using either noise-perturbed or partially observational data. This framework can discover conservation laws and scale effectively to physical models even with hundreds and thousands of freedoms. Specifically, the framework is comprised of an __auto-encoder__ with latent space being an high dimensional sphere, and a __neural unitary Koopman operator__ constructed by the Lie exponent map of neural network.


![image](https://github.com/jingddong-zhang/HNKO/blob/main/HNKO_sketch.png)


# Installation
Please download the packages in the **requirements.txt** file.

# Data
The data of HNKO_ast is provided in the [Google Drive](https://drive.google.com/file/d/1_4_n5GAD2jS-SqP-enf8S-5cI781qTZu/view?usp=sharing).

We thank Prof. [Fusco](https://www.math.unipd.it/en/department/people/giovanni.fusco/) for providing the orbit data of $n$-body problem in $3$-D space, including $n=4,12,24,60$. The [source data](http://adams.dm.unipi.it/~gronchi/nbody/) is orbit data of position variables for single body, we recover the whole orbit data for all the bodies by the method proposed in [[1]](https://link.springer.com/content/pdf/10.1007/s00222-010-0306-3.pdf). Notice that the whole orbit data for position $\mathbf{q}$ is still partial observation of the $n$-body problem, in which the full state $(\mathbf{q},\mathbf{p})$ also covers the momentum variable. Still, we numerically demonstate our HNKO performs well in this task. 

# Usage
The directory **Model in replys** contains the reproduced python code of CNN-LSTM, Hamiltonian ODE graph networks (HOGN) and reservoir computing.
For a standard comparison with these models, we apply the model structures in [CNN-LSTM](https://github.com/ozancanozdemir/CNN-LSTM), [graph-neural-ode](https://github.com/jaketae/graph-neural-ode/tree/master), [RC](https://github.com/zhuqunxi/RC_Lorenz).

The **hnko_feature.py** documents in directory **threeboy** and **kepler** are used to discover the Hamiltonians.

# Acknowledgement
Authors appreciate Phoenix, a talented artist, for designing the logo of [Research Institute of Intelligent Complex Systems](https://iics.fudan.edu.cn/main.htm).

# Citation
If you use our work in your research, please cite:

```
@article{PhysRevResearch.6.L012031,  
  title = {Learning Hamiltonian neural Koopman operator and simultaneously sustaining and discovering conservation laws},  
  author = {Zhang, Jingdong and Zhu, Qunxi and Lin, Wei},  
  journal = {Phys. Rev. Res.},  
  volume = {6},  
  issue = {1},  
  pages = {L012031},  
  numpages = {7},  
  year = {2024},  
  month = {Feb},  
  publisher = {American Physical Society},  
  doi = {10.1103/PhysRevResearch.6.L012031},  
  url = {https://link.aps.org/doi/10.1103/PhysRevResearch.6.L012031}  
}
```

# Reference

[1] Fusco, G., Gronchi, G. F., & Negrini, P. (2011). Platonic polyhedra, topological constraints and periodic solutions of the classical N-body problem. Inventiones mathematicae, 185(2), 283-332.

[2] Lezcano-Casado, M., & Martınez-Rubio, D. (2019, May). Cheap orthogonal constraints in neural networks: A simple parametrization of the orthogonal and unitary group. In International Conference on Machine Learning (pp. 3794-3803). PMLR.
