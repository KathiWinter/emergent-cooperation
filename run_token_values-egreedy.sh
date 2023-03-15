#!/bin/sh

# 1. Performance Evaluation

python train.py CoinGame-2 MATE-TD-EPSGREEDY
python train.py CoinGame-2 MATE-TD-EPSGREEDY-CONT

python train.py CoinGame-2 MATE-TD-EPSGREEDY
python train.py CoinGame-2 MATE-TD-EPSGREEDY-CONT

python train.py CoinGame-2 MATE-TD-EPSGREEDY
python train.py CoinGame-2 MATE-TD-EPSGREEDY-CONT

