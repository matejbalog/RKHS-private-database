# Differentially Private Database Release via Kernel Mean Embeddings

[Matej Balog](http://matejbalog.eu/en/research/), [Ilya Tolstikhin](http://tolstikhin.org), [Bernhard Schölkopf](http://is.tuebingen.mpg.de/person/bs)

*35th International Conference on Machine Learning ([ICML 2018](https://icml.cc/Conferences/2018))*

[[PDF](http://matejbalog.eu/research/database_RKHS_privacy.pdf)] 
[[arXiv](https://arxiv.org/abs/1710.01641)]

This repository contains scripts to reproduce the experiments appearing in this academic paper.

### Setup

Conda environment setup:
```
conda create -n RKHS-private-database python=3.6.3 matplotlib=2.1.0 numpy=1.13.3 pytorch=0.2.0 scikit-learn=0.19.0
source activate RKHS-private-database
```

### Data generation

Two synthetic data files were used to generate the plots in the paper:
* `D=2`: `data/mixture_of_Gaussians_N100000_D2{.npz, .json}`
* `D=5`: `data/mixture_of_Gaussians_N100000_D5{.npz, .json}`

You can re-generate these files yourself by executing:
```
python data.py 100000 2
python data.py 100000 5
```

### Experiments

#### Figure 1 ("Publishable subset" experiments)

Results of the experiments shown in Figure 1 are stored in the two files
* `D=2`: `results/D2_alg1_leak_M10000.json`
* `D=5`: `results/D5_alg1_leak_M10000.json`

You can re-generate these files by re-running the respective experiments as follows:
```
python experiments.py ../data/mixture_of_Gaussians_N100000_D2 leak --M 10000 1
python experiments.py ../data/mixture_of_Gaussians_N100000_D5 leak --M 10000 1
```

To then re-generate the plots shown in Figure 1, execute:
```
python plot.py --alg1 ../results/D2_alg1_leak_M10000.json --path_save ../figures/leaksD2
python plot.py --alg1 ../results/D5_alg1_leak_M10000.json --path_save ../figures/leaksD5
```

[figures/leaksD2](/figures/leaksD2.png?raw=true "Figure 1 (left)") |  [figures/leaksD5](/figures/leaksD5.png?raw=true "Figure 1 (right)")
:-----------------------------------------------------:|:-------------------------------------------------------:
![Figure 1](/figures/leaksD2.png?raw=true "Figure 1 (left)")  |  ![Figure 1](/figures/leaksD5.png?raw=true "Figure 1 (right)")


#### Figure 2 ("No publishable subset" experiments)

To re-run the experiments shown in Figure 2:
```
python experiments.py ../data/mixture_of_Gaussians_N100000_D2 random --M 10000 1
python experiments.py ../data/mixture_of_Gaussians_N100000_D5 random --M 10000 1
python experiments.py ../data/mixture_of_Gaussians_N100000_D2 random --M 10000 2
python experiments.py ../data/mixture_of_Gaussians_N100000_D5 random --M 10000 2
```

To then re-generate the plots shown in Figure 2, execute:
```
python plot.py --alg1 ../results/D2_alg1_random_M10000.json --alg2 ../results/D2_alg2_random_M10000.json --path_save ../figures/nodataD2
python plot.py --alg1 ../results/D5_alg1_random_M10000.json --alg2 ../results/D5_alg2_random_M10000.json --path_save ../figures/nodataD5
```

[figures/nodataD2](/figures/nodataD2.png?raw=true "Figure 2 (left)") |  [figures/nodataD5](/figures/nodataD5.png?raw=true "Figure 2 (right)")
:------------------------------------------------------:|:--------------------------------------------------------:
![Figure 2](/figures/nodataD2.png?raw=true "Figure 2 (left)")  |  ![Figure 2](/figures/nodataD5.png?raw=true "Figure 2 (right)")


### BibTeX
```
@inproceedings{balog2018privacy,
  author = {Balog, Matej and Tolstikhin, Ilya and Sch\"olkopf, Bernhard},
  title = {Differentially {Private} {Database} {Release} via {Kernel} {Mean} {Embeddings}},
  booktitle = {35th International Conference on Machine Learning (ICML)},
  year = {2018},
  month = {July}
}
```
