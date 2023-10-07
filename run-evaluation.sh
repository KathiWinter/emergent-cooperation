#!/bin/sh

# 1. Ablation

python train.py Matrix-IPD MATE-TD-SYNC
python train.py Matrix-IPD MATE-TD-SOV
python train.py Matrix-IPD MATE-TD-AUTOMATE
python train.py Matrix-IPD MATE-TD-T1
python train.py Matrix-IPD IAC
python train.py Matrix-IPD UCB-DEC

python train.py CoinGame-2 MATE-TD-SYNC
python train.py CoinGame-2 MATE-TD-SOV
python train.py CoinGame-2 MATE-TD-AUTOMATE
python train.py CoinGame-2 MATE-TD-T1
python train.py CoinGame-2 IAC
python train.py CoinGame-2 UCB-DEC

python train.py CoinGame-4 MATE-TD-SYNC
python train.py CoinGame-4 MATE-TD-SOV
python train.py CoinGame-4 MATE-TD-AUTOMATE
python train.py CoinGame-4 MATE-TD-T1
python train.py CoinGame-4 IAC
python train.py CoinGame-4 UCB-DEC

# 2. Performance Evaluation 

python train.py CoinGame-2x01 MATE-TD-T1
python train.py CoinGame-2x01 MATE-TD-SYNC
python train.py CoinGame-2x01 MATE-TD-SOV
python train.py CoinGame-2x01 LIO
python train.py CoinGame-2x01 Gifting-BUDGET
python train.py CoinGame-2x01 Gifting-ZEROSUM
python train.py CoinGame-2x01 IAC

python train.py CoinGame-6 MATE-TD-T1
python train.py CoinGame-6 MATE-TD-SYNC
python train.py CoinGame-6 MATE-TD-SOV
python train.py CoinGame-6 LIO
python train.py CoinGame-6 Gifting-ZEROSUM
python train.py CoinGame-6 Gifting-BUDGET
python train.py CoinGame-6 IAC

python train.py Harvest-6 MATE-TD-T1
python train.py Harvest-6 MATE-TD-SYNC
python train.py Harvest-6 MATE-TD-SOV
python train.py Harvest-6 LIO
python train.py Harvest-6 Gifting-ZEROSUM
python train.py Harvest-6 Gifting-BUDGET
python train.py Harvest-6 IAC

