# Generating-Correct-Answers-for-Progressive-Matrices-Intelligence-Tests
code for the paper "Generating Correct Answers for Progressive Matrices Intelligence Tests" to appear in NeurIPS 2020. 

(Niv Pekar,Yaniv Benny, Lior Wolf, 2020)

![](https://github.com/nivPekar/Generating-Correct-Answers-for-Progressive-Matrices-Intelligence-Tests/tree/main/images/intro.png)


## Requirements
* python 3.6
* NVIDIA GPU with CUDA 10.0+ capability
* numpy, scipy, matplotlib
* torch==1.4.0
* torchvision==0.5.0
* scikit-image


## Data
* [PGM](https://github.com/deepmind/abstract-reasoning-matrices)
* [RAVEN-FAIR](https://github.com/yanivbenny/RAVEN_FAIR) (Balanced version of RAVEN)


## Code
Please note that the git contains a working code. Soon we will upload the pretrained model, and we will tidy up the code in the subsequent weeks.

## Usage
To train the generation network:
```
python3 train_generation.py --epochs <number of epochs> --data_path <path to the dataset> --batch_size <batch size number>
```


## Evaluating models
* [Multi-scale Reasoning Network (MRNet)](https://github.com/yanivbenny/MRNet)
* [Wild Relational Network (WReN)](https://github.com/Fen9/WReN)
* [Logic Embedding Network (LEN)](https://github.com/zkcys001/distracting_feature)

## Performance
For details, please check our paper. 
