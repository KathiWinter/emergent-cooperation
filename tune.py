from settings import params
import mate.domains as domains
import mate.controllers.mate as mate
import mate.experiments as experiments
import mate.data as data
import ray
from ray import tune

def objective(config):
  params["domain_name"] = "CoinGame-2"

  env = domains.make(params)
  env.reset()

  params["mate_mode"] = "td_error"
  params["token_value"] = config["a"]
  controller = mate.MATE(params)

  params["directory"] = params["output_folder"] + "/" + params["data_prefix_pattern"].\
      format(
          params["nr_agents"],\
          params["domain_name"],\
          params["algorithm_name"])
  params["directory"] = data.mkdir_with_timestap(params["directory"])
  result = experiments.run_training(env, controller, params)
  combined_rewards = [sum(x) for x in zip(result["undiscounted_returns"][0][-2:], result["undiscounted_returns"][1][-2:])]
  
  score = sum(combined_rewards) / 2
  return {"score": score}

#Ray Tune
search_space = {
    "a": tune.uniform(0.5, 3),
}

#objective(search_space)
ray.init(runtime_env={"working_dir": "C:/Users/Kathi/Desktop/MasterThesis/MATE/emergent-cooperation"})
tuner = tune.Tuner(objective, param_space=search_space)
results = tuner.fit()
print(results.get_best_result(metric="score", mode="max").config)

