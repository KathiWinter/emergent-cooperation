from settings import params
import mate.domains as domains
import mate.algorithms as algorithms
import mate.experiments as experiments
import mate.data as data
import sys
import ray 

@ray.remote
def f(token):
    params["domain_name"] = "CoinGame-4"
    params["algorithm_name"] = "MATE-TD-T" + str(token)
    env = domains.make(params)
    env.reset()
    controller = algorithms.make(params)

    params["directory"] = params["output_folder"] + "/" + params["data_prefix_pattern"].\
        format(
            params["nr_agents"],\
            params["domain_name"],\
            params["algorithm_name"])
    params["directory"] = data.mkdir_with_timestap(params["directory"])
    result = experiments.run_training(env, controller, params)
    return result 

#ray.init()
x_id = f.remote(0.25)
y_id = f.remote(0.5)
z_id = f.remote(1)
v_id = f.remote(2)
q_id = f.remote(4)

x, y, z, v, q = ray.get([x_id, y_id, z_id, v_id, q_id])