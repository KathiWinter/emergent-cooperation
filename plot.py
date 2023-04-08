from settings import params
import mate.plotting as plotting
import mate.domains as domains
import sys
import matplotlib.pyplot as plot
from os.path import join

ALGORITHM_NAMES = {
    "IAC_": "Naive Learning",
    "Gifting-ZEROSUM_": "Gifting (Zero-Sum)",
    "Gifting-BUDGET_": "Gifting (Budget)",
    #"MATE-TD-T0_": "MATE T-0 / Naive Learning",
    "MATE-TD_": "MATE T-1",
    "MATE-TD-T2": "MATE T-2",
    "MATE-TD-T4": "MATE T-4",
    "MATE-TD-T0.25": "MATE T-0.25",
    "MATE-TD-T0.5": "MATE T-0.5",
    "MATE-TD-T0.75": "MATE T-0.75",
    "MATE-TD-T8": "MATE T-8",
    "MATE-TD-T1.5": "MATE T-1.5",
    "MATE-TD-T2.5": "MATE T-2.5",
    "MATE-TD-T3": "MATE T-3",
    "MATE-TD-RANDOM": "MATE (random)",
    "MATE-TD-EPSGREEDY": "MATE (epsilon-greedy)",
    "MATE-TD-UCB": "MATE (UCB)",
   "MATE-TD-EARNING": "MATE (Earning)",
    "MATE-REWARD_": "MATE (reward-based)",
    "LIO_": "LIO",
    "MATE-TD-DEFECT_COMPLETE_": "MATE (defect=Complete)",
    "MATE-TD-DEFECT_REQUEST_": "MATE (defect=Request)",
    "MATE-TD-DEFECT_RESPONSE_": "MATE (defect=Response)",
    "Random_": "Random"
}

Y_LABEL = {
    "undiscounted_returns": "Efficiency (U)",
    "discounted_returns": "collective discounted return",
    "own_coin_prob": "P(own coin)",
    "peace": "Peace (P)",
    "sustainability": "Sustainability (S)"
}

LEGEND_CONFIG = {
    "Matrix-IPD":7,
    "CoinGame-2":7,
    "Harvest-6":4
}
path = "plots"
plot.figure(figsize=(4, 3))
ax = plot.gca()
baseline_comparison = sys.argv[1] == "True"
params["domain_name"] = sys.argv[2]
env = domains.make(params)
params["filter_size"] = 10
params["nr_runs"] = 20
params["directory"] = "output"
metric = "undiscounted_returns"
params["metric_index"] = None
params["norm_index"] = None
params["stats_label"] = "domain_values"
params["y_label"] = Y_LABEL["undiscounted_returns"]
params["data_length"] = params["nr_epochs"]
if len(sys.argv) > 3:
    params["stats_label"] = "domain_values"
    metric = sys.argv[3]
if metric == "equality":
    params["stats_label"] = metric
    params["y_label"] = "Equality (E)"
elif "_returns" not in metric:
    params["metric_index"], params["norm_index"] = env.get_metric_indices(metric)
    params["norm_nr_agents"] = 1
    params["y_label"] = Y_LABEL[metric]
if len(sys.argv) > 4:
    params["norm_nr_agents"] = env.nr_agents
    params["y_label"] += " per agent"
if params["domain_name"].startswith("Matrix-"):
    params["stats_label"] = "domain_values"
    params["norm_index"] = 0
    params["y_label"] = Y_LABEL["undiscounted_returns"]
    params["norm_nr_agents"] = 1
params["x_label"] = "epoch"
filename_prefix = "{}_{}".format(params["domain_name"].lower(), metric.lower())
if not baseline_comparison:
    filename_prefix += "_defectors"
filename = filename_prefix + ".svg"
filename_png = filename_prefix + ".png"
filename_pdf = filename_prefix + ".pdf"

data_prefix_pattern = params["data_prefix_pattern"]

if baseline_comparison:
    algorithm_info = [("sienna","MATE-TD_"), ("sienna","MATE-TD-T1.5"), ("sienna","MATE-TD-T2"), ("sienna","MATE-TD-T2.5"), ("deeppink","MATE-TD-RANDOM"), ("turquoise","MATE-TD-EARNING"), ("gray","IAC_")]
else:
    algorithm_info = [("sienna","MATE-TD_"), ("sienna","MATE-TD-T1.5"), ("sienna","MATE-TD-T2"), ("sienna","MATE-TD-T2.5"), ("deeppink","MATE-TD-RANDOM"), ("turquoise","MATE-TD-EARNING"), ("gray","IAC_")]

for color, algorithm_name in algorithm_info:
    params["data_prefix_pattern"] = data_prefix_pattern.format(
        params["nr_agents"],\
        params["domain_name"],\
        algorithm_name)
    params["label"] = ALGORITHM_NAMES[algorithm_name]
    params["color"] = color
    params["plot_mean"] = algorithm_name == "Random_"
    params["plot_mean_length"] = 5000
    if params["domain_name"].startswith("Matrix-IPD") and algorithm_name == "LIO_":
        # Number from the original paper
        plot.plot([0,params["data_length"]-1], [-2.25, -2.25], linestyle="dashed", color=color, label="LIO")
    if params["domain_name"].startswith("Matrix-IPD") and algorithm_name == "Random_":
        # Analytically determined performance
        plot.plot([0,params["data_length"]-1], [-3, -3], linestyle="dashed", color=color, label="Random")
    else:
        plotting.plot_runs(params)
if params["domain_name"].startswith("Matrix-IPD"):
    # Number from the original paper
    plot.plot([0,params["data_length"]-1], [-2.4, -2.4], linestyle="dashed", color="purple", label="LOLA-PG")
if params["domain_name"] == "CoinGame-2":
    if params["y_label"] == Y_LABEL["own_coin_prob"]:
        # Number from the original paper
        plot.plot([0,params["data_length"]-1], [0.82, 0.82], linestyle="dashed", color="purple", label="LOLA-PG")
    if params["y_label"] == Y_LABEL["undiscounted_returns"]:
        # Number from the original paper
        plot.plot([0,params["data_length"]-1], [16, 16], linestyle="dashed", color="purple", label="LOLA-PG")
legend_position = None
if params["domain_name"] in LEGEND_CONFIG and params["y_label"] == Y_LABEL["undiscounted_returns"]:
    legend_position = LEGEND_CONFIG[params["domain_name"]]
    legend = plot.legend(loc=legend_position)
if not baseline_comparison and params["y_label"] == Y_LABEL["undiscounted_returns"]:
    legend = plot.legend()
plot.tight_layout()
ax.legend(bbox_to_anchor=(1.1, 1.05))
ax.grid(which='both', linestyle='--')
plot.savefig(join(path, filename), bbox_inches='tight')
plot.savefig(join(path, filename_png), bbox_inches='tight')
plot.savefig(join(path, filename_pdf), bbox_inches='tight')