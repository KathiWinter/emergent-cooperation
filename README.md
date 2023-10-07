# AutoMATE with Consensus

AutoMATE is an extension of the Mutual Acknowledgment Token Exchange (MATE) [1] for decentralized and automatic token development. This repository is forked from the repository by Thomy Phan, which can be found at https://github.com/thomyphan/emergent-cooperation. We added the AutoMATE and Consensus extensions, as well as two Coin Game settings (6-agent and option for rescaled rewards). This Readme specifies only the relevant algorithms and domains for the AutoMATE extension.

## How to run AutoMATE and Consensus

To run training, the following command must be executed. Algorithm M (insert algorithm label here) runs in domain D (insert domain label here): 

`python train.py D M`

Example: `python train.py CoinGame-2 MATE-TD` to run AutoMATE w. Synchronized Consensus in the Coin Game 2.

The available domains and algorithms are listed below. Domains and Algorithms marked with an asterisk (*) are not used in the experiments but are left for reference and potential future work.

The scripts `run-analysis.sh` and `run-evaluation.sh` include all commands to reproduce my experiments.

The trained models are saved to the `output` folder, which will be automatically generated. The output directory can be changed in `settings.py`.

### AutoMATE algorithms

| Algorithm                          | Label            |
| ---------------------------------- | ---------------- |
| AutoMATE w. Synchronized Consensus | `MATE-TD-SYNC`   |
| AutoMATE w. Sovereign Consensus    | `MATE-TD-SOV`    |
| AutoMATE without Consensus         | `MATE-TD-AUTOMATE`|

### Other available MARL algorithms

| Algorithm             | Label                                                 |
| --------------------- | ----------------------------------------------------- |
| MATE-TD (fixed token) | `MATE-TD-T{0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 8}` |
| Naive Learner         | `IAC`                                                 |
| LOLA                  | `LOLA`(*)                                             |
| Gifting (Zero-Sum)    | `Gifting-ZEROSUM`                                     |
| Gifting (Budget)      | `Gifting-BUDGET`                                      |
| LIO                   | `LIO`                                                 |

### Other available discrete token Algorithms based on MATE

All algorithms sample from [0.25, 0.5, 1, 2, 4].

| Algorithm                   | Label                | Description                                    |
| --------------------------- | -------------------- | ---------------------------------------------- |
| MATE w. UCB (Centralized)   | `MATE-TD-UCB-CENT`   | UCB (centralized)(*)                           |
| MATE w. UCB (Decentralized) | `MATE-TD-UCB-DEC`    | UCB (decentralized)                            |
| MATE Random (per epoch)     | `MATE-TD-RANDOM`     | Random Token at each epoch(*)                  |
| MATE Random (per time step) | `MATE-TD-RANDOM-TS`  | Random Token at each time-step (centralized)   |
| MATE Random (Reflecting)    | `MATE-TD-REFLECTING` | Random Token at each time-step (decentralized) |
| MATE Random (Holding)       | `MATE-TD-HOLDING`    | Random Token at each time-step (decentralized) |

### Available domains

| Domain      | Label           | Description                                         |
| ----------- | --------------- | --------------------------------------------------- |
| IPD         | `Matrix-IPD`    | Iterated Prisoner's Dilemma                         |
| Coin[2]     | `CoinGame-2`    | 2-player version of Coin Game                            |
| Coin[2] x01 | `CoinGame-2x01` | 2-player version of Coin Game with rewards of (0.1/-0.2) |
| Coin[4]     | `CoinGame-4`    | 4-player version of Coin Game                           |
| Coin[6]     | `CoinGame-6`    | 6-player version of Coin Game                           |
| Harvest[6]  | `Harvest-6`     | Harvest domain with 6 agents                        |
| Harvest[12] | `Harvest-12`    | Harvest domain with 12 agents(*)                    |

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
