# AutoMATE with Consensus

AutoMATE is an extension of the Mutual Acknowledgment Token Exchange (MATE) [1] for decentralized and automatic token development. This repository is forked from the repository of Thomy Phan, which can be found at https://github.com/thomyphan/emergent-cooperation. We added the AutoMATE and MEDIATE (Consensus) extensions, as well as two environment settings for Coin Game. This Readme specifies only the relevant algorithms and domains for the AutoMATE extension.

## 3. How to run AutoMATE and Consensus

To run training, the following command must be executed. Algorithm M (insert algorithm label here) runs in domain D (insert domain label here). The available domains and algorithms are listed below.

`python train.py D M`

Example: `python train.py CoinGame-2 MATE-TD` to run AutoMATE w. Synchronized Consensus in the Coin Game 2.

The scripts `run-analysis.sh` and `run-evaluation.sh` include all commands to reproduce our experiments.

The trained models are saved to the `output` folder, which can be changed in `settings.py`.

### AutoMATE algorithms

| Algorithm                         | Label                  |
| --------------------------------- | ---------------------- |
| MEDIATE w. Synchronized Consensus | `MATE-TD-MEDIATE-SYNC` |
| MEDIATEE w. Sovereign Consensus   | `MATE-TD-MEDIATE-SOV`  |
| AutoMATE without Consensus        | `MATE-TD-AUTOMATE`     |

### Other available MARL algorithms

| Algorithm             | Label                                                 |
| --------------------- | ----------------------------------------------------- |
| MATE-TD (fixed token) | `MATE-TD-T{0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 8}` |
| Naive Learner         | `IAC`                                                 |
| Gifting (Zero-Sum)    | `Gifting-ZEROSUM`                                     |
| Gifting (Budget)      | `Gifting-BUDGET`                                      |
| LIO                   | `LIO`                                                 |

### Available domains used in the Paper Evaluations

| Domain      | Label           | Description                                         |
| ----------- | --------------- | --------------------------------------------------- |
| IPD         | `Matrix-IPD`    | Iterated Prisoner's Dilemma                         |
| Coin[2]     | `CoinGame-2`    | 2-player version of Coin                            |
| Coin[2] x01 | `CoinGame-2x01` | 2-player version of Coin with rewards of (0.1/-0.2) |
| Coin[4]     | `CoinGame-4`    | 4-player version of Coin                            |
| Coin[6]     | `CoinGame-6`    | 6-player version of Coin                            |
| Harvest[6]  | `Harvest-6`     | Harvest domain with 6 agents                        |

## Experiment parameters

The experiment parameters are specified in `settings.py` and can be adjusted.

| Parameter                       | Default Value           |
| ------------------------------- | ----------------------- |
| `params["episodes_per_epoch"]`  | 10                      |
| `params["nr_epochs"]`           | 5000                    |
| `params["nr_hidden_units"]`     | 64                      |
| `params["clip_norm"]`           | 1                       |
| `params["learning_rate"]`       | 0.001                   |
| `params["output_folder"]`       | "output"                |
| `params["data_prefix_pattern"]` | {}-agents*domain-{}*{}" |

## Dependencies

To run the scripts in this repository, make sure you have Python installed along with the following packages:

- numpy
- torch

To install these packages with pip, enter `pip install numpy torch` in your terminal.
Note: The script also uses Python's standard libraries like sys, json, os, and copy, which are available with the Python installation.

## References

- [1] T. Phan et al., ["Emergent Cooperation from Mutual Acknowledgment Exchange"](https://ifaamas.org/Proceedings/aamas2022/pdfs/p1047.pdf), in AAMAS 2022
