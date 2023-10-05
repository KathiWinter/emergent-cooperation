#!/bin/sh

# Centralized 

python train.py Matrix-IPD MATE-TD-T0.25
python train.py Matrix-IPD MATE-TD-T0.5
python train.py Matrix-IPD MATE-TD-T0.75
python train.py Matrix-IPD MATE-TD-T1
python train.py Matrix-IPD MATE-TD-T1.5
python train.py Matrix-IPD MATE-TD-T2
python train.py Matrix-IPD MATE-TD-T4
python train.py Matrix-IPD MATE-TD-T8

python train.py CoinGame-2 MATE-TD-T0.25
python train.py CoinGame-2 MATE-TD-T0.5
python train.py CoinGame-2 MATE-TD-T0.75
python train.py CoinGame-2 MATE-TD-T1
python train.py CoinGame-2 MATE-TD-T1.5
python train.py CoinGame-2 MATE-TD-T2
python train.py CoinGame-2 MATE-TD-T4
python train.py CoinGame-2 MATE-TD-T8

python train.py CoinGame-4 MATE-TD-T0.25
python train.py CoinGame-4 MATE-TD-T0.5
python train.py CoinGame-4 MATE-TD-T0.75
python train.py CoinGame-4 MATE-TD-T1
python train.py CoinGame-4 MATE-TD-T1.5
python train.py CoinGame-4 MATE-TD-T2
python train.py CoinGame-4 MATE-TD-T4
python train.py CoinGame-4 MATE-TD-T8

python train.py CoinGame-2x01 MATE-TD-T0.25
python train.py CoinGame-2x01 MATE-TD-T0.5
python train.py CoinGame-2x01 MATE-TD-T0.75
python train.py CoinGame-2x01 MATE-TD-T1
python train.py CoinGame-2x01 MATE-TD-T1.5
python train.py CoinGame-2x01 MATE-TD-T2
python train.py CoinGame-2x01 MATE-TD-T4
python train.py CoinGame-2x01 MATE-TD-T8

python train.py CoinGame-6 MATE-TD-T0.25
python train.py CoinGame-6 MATE-TD-T0.5
python train.py CoinGame-6 MATE-TD-T0.75
python train.py CoinGame-6 MATE-TD-T1
python train.py CoinGame-6 MATE-TD-T1.5
python train.py CoinGame-6 MATE-TD-T2
python train.py CoinGame-6MATE-TD-T4
python train.py CoinGame-6 MATE-TD-T8


# Decentralized

python train.py CoinGame-2 MATE-TD-T0.25
python train.py CoinGame-2 MATE-TD-T0.5
python train.py CoinGame-2 MATE-TD-T0.75
python train.py CoinGame-2 MATE-TD-T1
python train.py CoinGame-2 MATE-TD-T1.5
python train.py CoinGame-2 MATE-TD-T2
python train.py CoinGame-2 MATE-TD-T4
python train.py CoinGame-2 MATE-TD-T8

python train.py CoinGame-2 MATE-TD-T0.25-0.5
python train.py CoinGame-2 MATE-TD-T0.25-1
python train.py CoinGame-2 MATE-TD-T0.25-2
python train.py CoinGame-2 MATE-TD-T0.25-4
python train.py CoinGame-2 MATE-TD-T0.5-1
python train.py CoinGame-2 MATE-TD-T0.5-2
python train.py CoinGame-2 MATE-TD-T0.5-4
python train.py CoinGame-2 MATE-TD-T1-2
python train.py CoinGame-2 MATE-TD-T1-4
python train.py CoinGame-2 MATE-TD-T2-4