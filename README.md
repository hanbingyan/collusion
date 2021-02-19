# Algorithmic collusion and experience replay

This repository contains code for the paper ["Understanding algorithmic collusion with experience replay"](https://arxiv.org/abs/2102.09139). Feel free to contact me for any problems in the code.

- [Dependencies](#dependencies)
- [Usages](#usages)

 ## Dependencies

 - Python 3 with NumPy 
 - PyTorch 

 ## Usages

To reproduce results in the paper, a random seed `12345` is used for PyTorch, NumPy, and Random module in Python. When generating benchmark results, I run the code with CPU to avoid unexpected randomness from GPU. However, GPU also reproduces the same outcomes in my tests. I use Python 3.7.2 with NumPy 1.16.2 and Torch 1.7.1. Refer to the paper for abbreviations used below. 

 ### Classic Q-learning

- C-Online: `classic_online.py`
- C-Random: `classic_random.py`
- C-Rank: `classic_rank.py`

### Deep Q-learning

- D-Random: `deep_random.py`
- D-Online: use `deep_random.py`. But modify `optimize_model` method in `utils.py` by changing sampling to `transitions = memory.recent(BATCH_SIZE)` instead
- D-Rank: `deep_rank.py`

### Heterogeneous players
- C-Online vs D-Random: `htrg_conline_drand.py`
- C-Online vs D-Rank: Use `htrg_crank_drank.py` but modify Line 80, If statement with `l_state`. Make the classic player use online algorithm
- C-Rank vs D-Rank: `htrg_crank_drank.py`

### Figures

Jupyter Notebooks in `figs` generate figures in the paper, with outputs from code above. In addition, `cdrk_mimic.ipynb` produces the last figure for the mimicking experiment.
