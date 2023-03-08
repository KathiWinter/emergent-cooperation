#!/bin/sh

# 1. Performance Evaluation MATE TD with random Token Values in ranges [0.5, 1, 2], [0.5, 1, 1.5], [0.25, 0.5, 1, 2, 4], [0.25, 0.5, 1, 2, 4, 8]

python train.py CoinGame-2 MATE-TD-RANDOM
python train.py CoinGame-2 MATE-TD-RANDOM-B
python train.py CoinGame-2 MATE-TD-RANDOM-C
python train.py CoinGame-2 MATE-TD-RANDOM-D

python train.py CoinGame-2 MATE-TD-RANDOM
python train.py CoinGame-2 MATE-TD-RANDOM-B
python train.py CoinGame-2 MATE-TD-RANDOM-C
python train.py CoinGame-2 MATE-TD-RANDOM-D

python train.py CoinGame-2 MATE-TD-RANDOM
python train.py CoinGame-2 MATE-TD-RANDOM-B
python train.py CoinGame-2 MATE-TD-RANDOM-C
python train.py CoinGame-2 MATE-TD-RANDOM-D

python train.py CoinGame-2 MATE-TD-RANDOM
python train.py CoinGame-2 MATE-TD-RANDOM-B
python train.py CoinGame-2 MATE-TD-RANDOM-C
python train.py CoinGame-2 MATE-TD-RANDOM-D

python train.py CoinGame-2 MATE-TD-RANDOM
python train.py CoinGame-2 MATE-TD-RANDOM-B
python train.py CoinGame-2 MATE-TD-RANDOM-C
python train.py CoinGame-2 MATE-TD-RANDOM-D

