# AutoMATE with Consensus

AutoMATE is an extension of the Mutual Acknowledgment Token Exchange (MATE) [1] for decentralized and automatic token development. This repository is forked from the repository by Thomy Phan, which can be found at https://github.com/thomyphan/emergent-cooperation. We added the AutoMATE and Consensus extensions, as well as two Coin Game settings. This Readme specifies only the relevant algorithms and domains for the AutoMATE extension. 

## 3. How to run AutoMATE and Consensus

To run training, the following command must be executed. Algorithm M (insert algorithm label here) runs in domain D (insert domain label here). The available domains and algorithms are listed below.

`python train.py D M`

Example: `python train.py CoinGame-2 MATE-TD` to run AutoMATE w. Synchronized Consensus in the Coin Game 2.

The trained models are saved to the `output` folder, which can be changed in `settings.py`.

### AutoMATE algorithms
| Algorithm       | Label                  |
|-----------------|------------------------|
| AutoMATE w. Synchronized Consensus       | `MATE-TD`                   |
| AutoMATE w. Sovereign Consensus                | `MATE-TD-NOSYNC`       |
| AutoMATE without Consensus  | `MATE-TD-NC`       |


### Other available MARL algorithms


| Algorithm       | Label                  |
|-----------------|------------------------|
| MATE-TD (fixed token)        | `MATE-TD-T{0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 8}`       |
| Naive Learner      | `IAC`                   |
| LOLA                | `LOLA`       |
| Gifting (Zero-Sum) | `Gifting-ZEROSUM`       |
| Gifting (Budget)   | `Gifting-BUDGET`       |
| LIO                | `LIO`       |

### Other available discrete token Algorithms based on MATE
All algorithms sample from [0.25, 0.5, 1, 2, 4].

| Algorithm       | Label                  | Description   |
|-----------------|------------------------|-------------------------|
| MATE w. UCB (Centralized)    | `MATE-TD-UCB-CENT`       | |
| MATE w. UCB (Decentralized)    | `MATE-TD-UCB-DEC`       |  |
| MATE Random (per epoch)    | `MATE-TD-RANDOM`       | Random Token at each epoch|
| MATE Random (per time step)   | `MATE-TD-RANDOM-TS`   | Random Token at each time-step (centralized)|
| MATE Random (Reflecting)    | `MATE-TD-REFLECTING`       | Random Token at each time-step (decentralized)|
| MATE Random (Holding)    | `MATE-TD-HOLDING`       | Random Token at each time-step (decentralized)|


### Available domains

| Domain   		| Label            | Description                                                       |
|---------------|------------------|-------------------------------------------------------------------|
| IPD           | `Matrix-IPD`     | Iterated Prisoner's Dilemma                 					   |
| Coin[2]       | `CoinGame-2`     | 2-player version of Coin                   					   |
| Coin[2] x01   | `CoinGame-2x01`   | 2-player version of Coin with rewards of (0.1/-0.2)                   					   |
| Coin[4]       | `CoinGame-4`     | 4-player version of Coin                   					   |
| Coin[6]       | `CoinGame-6`     | 6-player version of Coin                   					   |
| Harvest[6]    | `Harvest-6`      | Harvest domain with 6 agents 				                       |
| Harvest[12]    | `Harvest-12`      | Harvest domain with 12 agents 				                       |



## Experiment parameters

The experiment parameters are specified in `settings.py` and can be adjusted.  

| Parameter   		| Default Value    |
|---------------|------------------|
|`params["episodes_per_epoch"]` |  10  |
|`params["nr_epochs"]` |  5000  |
|`params["nr_hidden_units"]` |  64  |
|`params["clip_norm"]` |  1  |
|`params["learning_rate"]` |  0.001  |
|`params["output_folder"]` |  "output"  |
|`params["data_prefix_pattern"]` |  {}-agents_domain-{}_{}"  |


## References

- [1] T. Phan et al., ["Emergent Cooperation from Mutual Acknowledgment Exchange"](https://ifaamas.org/Proceedings/aamas2022/pdfs/p1047.pdf), in AAMAS 2022
