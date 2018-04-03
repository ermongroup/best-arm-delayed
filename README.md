Best arm identification in multi-armed bandits with delayed feedback
============================================

This repository provides a reference implementation for best arm identification in multi-armed bandits as described in the paper:

> Best arm identification in multi-armed bandits with delayed feedback  
Aditya Grover, Todor Markov, Peter Attia, Norman Jin, Nicholas Perkins, Bryan Cheong, Michael Chen, Zi Yang, Stephen Harris, William Chueh, Stefano Ermon.  
International Conference on Artificial Intelligence and Statistics (AISTATS), 2018.   
Paper: https://arxiv.org/abs/1803.10937

## Requirements

The codebase is implemented in Python 3.6. To install the necessary requirements, run the following commands:

```
pip install -r requirements.txt
```

## Options

The `main.py` script which provides the following command line arguments.

```
  --exp_type STR		free_means', 'bounded_means', 'delay', or 'none'. First three reproduce paper results, 'none' takes user-defined parameters below
  --n INT			number of total arms
  --delta FLOAT			1 - target confidence
  --k INT			number of top arms to be identified
  --std INT			standard deviation for full feedback
  --bsize INT			batch size for parallel MAB
  --d INT			delay of feedback
  --pstd FLOAT			standard deviation for partial unbiased feedback
  --seed INT			seed for simulations
  --num_tries INT		number of simulations for reproducing paper experiments
```

*Note*: For simplicity, we input scalars for `std`, `pstd`, `delay` parameters from the command line that are shared across arms. The code supports setting these parameters independently for each arm.

## Examples

* Test simulation for sequential MAB, partial feedback (everything else default)

```
python main.py --bsize 1
```

* Test simulation for parallel MAB, partial feedback (everything else default)

```
python main.py --bsize 10
```

* Resimulate the *bounded means* experiment in the paper

```
python main.py --exp_type=bounded_means
```

* Resimulate the *free means* experiment in the paper

```
python main.py --exp_type=free_means
```

* Resimulate the variation in *delays* experiment in the paper

```
python main.py --exp_type=delay
```


## Citing

If you find this repository useful in your research, please consider citing the following paper:

>@inproceedings{grover2018best,  
  title={Best arm identification in multi-armed bandits with delayed feedback},  
  author={Grover, Aditya and Markov, Todor and Attia, Peter and Jin, Norman and Perkins, Nicholas and Cheong, Bryan and Chen, Michael and Yang, Zi and Harris, Stephen and Chueh, William and Ermon, Stefano},  
  booktitle={International Conference on Artificial Intelligence and Statistics},  
  year={2018}}