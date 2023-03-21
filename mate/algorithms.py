import mate.controllers.controller as controller
import mate.controllers.actor_critic as actor_critic
import mate.controllers.lola as lola
import mate.controllers.gifting as gifting
import mate.controllers.mate as mate
import mate.controllers.lio as lio

def make(params):
    algorithm_name = params["algorithm_name"]
    if algorithm_name == "Random":
        return controller.Controller(params)
    if algorithm_name == "IAC":
        return actor_critic.ActorCritic(params)
    if algorithm_name == "LOLA":
        return lola.LOLA(params)
    if algorithm_name == "LIO":
        params["no_ppo"] = False
        return lio.LIO(params)
    if algorithm_name == "LIO-0.1":
        params["no_ppo"] = False
        params["comm_failure_prob"] = 0.1
        return lio.LIO(params)
    if algorithm_name == "LIO-0.2":
        params["no_ppo"] = False
        params["comm_failure_prob"] = 0.2
        return lio.LIO(params)
    if algorithm_name == "LIO-0.4":
        params["no_ppo"] = False
        params["comm_failure_prob"] = 0.4
        return lio.LIO(params)
    if algorithm_name == "LIO-0.8":
        params["no_ppo"] = False
        params["comm_failure_prob"] = 0.8
        return lio.LIO(params)
    if algorithm_name == "Gifting-BUDGET":
        params["gifting_mode"] = gifting.BUDGET_MODE
        return gifting.Gifting(params)
    if algorithm_name == "Gifting-BUDGET-0.1":
        params["comm_failure_prob"] = 0.1
        params["gifting_mode"] = gifting.BUDGET_MODE
        return gifting.Gifting(params)
    if algorithm_name == "Gifting-BUDGET-0.2":
        params["comm_failure_prob"] = 0.2
        params["gifting_mode"] = gifting.BUDGET_MODE
        return gifting.Gifting(params)
    if algorithm_name == "Gifting-BUDGET-0.4":
        params["comm_failure_prob"] = 0.4
        params["gifting_mode"] = gifting.BUDGET_MODE
        return gifting.Gifting(params)
    if algorithm_name == "Gifting-BUDGET-0.8":
        params["comm_failure_prob"] = 0.8
        params["gifting_mode"] = gifting.BUDGET_MODE
        return gifting.Gifting(params)
    if algorithm_name == "Gifting-ZEROSUM":
        params["gifting_mode"] = gifting.ZERO_SUM_MODE
        return gifting.Gifting(params)
    if algorithm_name == "Gifting-ZEROSUM-0.1":
        params["comm_failure_prob"] = 0.1
        params["gifting_mode"] = gifting.ZERO_SUM_MODE
        return gifting.Gifting(params)
    if algorithm_name == "Gifting-ZEROSUM-0.2":
        params["comm_failure_prob"] = 0.2
        params["gifting_mode"] = gifting.ZERO_SUM_MODE
        return gifting.Gifting(params)
    if algorithm_name == "Gifting-ZEROSUM-0.4":
        params["comm_failure_prob"] = 0.4
        params["gifting_mode"] = gifting.ZERO_SUM_MODE
        return gifting.Gifting(params)
    if algorithm_name == "Gifting-ZEROSUM-0.8":
        params["comm_failure_prob"] = 0.8
        params["gifting_mode"] = gifting.ZERO_SUM_MODE
        return gifting.Gifting(params)
    if algorithm_name == "MATE-REWARD":
        return mate.MATE(params)
    if algorithm_name == "MATE-TD":
        params["mate_mode"] = "td_error"
        return mate.MATE(params)
    if algorithm_name == "MATE-TD-0.1":
        params["mate_mode"] = "td_error"
        params["comm_failure_prob"] = 0.1
        return mate.MATE(params)
    if algorithm_name == "MATE-TD-0.2":
        params["mate_mode"] = "td_error"
        params["comm_failure_prob"] = 0.2
        return mate.MATE(params)
    if algorithm_name == "MATE-TD-0.4":
        params["mate_mode"] = "td_error"
        params["comm_failure_prob"] = 0.4
        return mate.MATE(params)
    if algorithm_name == "MATE-TD-0.8":
        params["mate_mode"] = "td_error"
        params["comm_failure_prob"] = 0.8
        return mate.MATE(params)
    if algorithm_name == "MATE-TD-T0.5":
        params["mate_mode"] = "td_error"
        params["token_value"] = 0.5
        return mate.MATE(params)
    if algorithm_name == "MATE-TD-T0.25":
        params["mate_mode"] = "td_error"
        params["token_value"] = 0.25
        return mate.MATE(params)
    if algorithm_name == "MATE-TD-T2":
        params["mate_mode"] = "td_error"
        params["token_value"] = 2
        return mate.MATE(params)
    if algorithm_name == "MATE-TD-T4":
        params["mate_mode"] = "td_error"
        params["token_value"] = 4
        return mate.MATE(params)
    if algorithm_name == "MATE-TD-T8":
        params["mate_mode"] = "td_error"
        params["token_value"] = 8
        return mate.MATE(params)
    if algorithm_name == "MATE-TD-T0":
        params["mate_mode"] = "td_error"
        params["token_value"] = 0
        return mate.MATE(params)
    if algorithm_name == "MATE-TD-RANDOM":
        params["mate_mode"] = "td_error"
        params["token_mode"] = "random"
        return mate.MATE(params)
    if algorithm_name == "MATE-TD-DYNAMIC_TOKEN":
        params["mate_mode"] = "td_error"
        params["token_mode"] = "dynamic-token"
        return mate.MATE(params)
    if algorithm_name == "MATE-TD-RANDOM-B":
        params["mate_mode"] = "td_error"
        params["token_mode"] = "random"
        params["token_range"] = [0.5, 1, 1.5]
        return mate.MATE(params)
    if algorithm_name == "MATE-TD-RANDOM-C":
        params["mate_mode"] = "td_error"
        params["token_mode"] = "random"
        params["token_range"] = [0.25, 0.5, 1, 2, 4]
        return mate.MATE(params)
    if algorithm_name == "MATE-TD-RANDOM-D":
        params["mate_mode"] = "td_error"
        params["token_mode"] = "random"
        params["token_range"] = [0.25, 0.5, 1, 2, 4, 8]
        return mate.MATE(params)
    if algorithm_name == "MATE-TD-EPSGREEDY":
        params["mate_mode"] = "td_error"
        params["token_mode"] = "epsilon-greedy"
        return mate.MATE(params)
    if algorithm_name == "MATE-TD-EPSGREEDY-E2":
        params["mate_mode"] = "td_error"
        params["token_mode"] = "epsilon-greedy"
        params["epsilon"] = 0.2
        return mate.MATE(params)
    if algorithm_name == "MATE-TD-EPSGREEDY-E3":
        params["mate_mode"] = "td_error"
        params["token_mode"] = "epsilon-greedy"
        params["epsilon"] = 0.3
        return mate.MATE(params)
    if algorithm_name == "MATE-TD-EPSGREEDY-E5":
        params["mate_mode"] = "td_error"
        params["token_mode"] = "epsilon-greedy"
        params["epsilon"] = 0.5
        return mate.MATE(params)
    if algorithm_name == "MATE-TD-EPSGREEDY-I025":
        params["mate_mode"] = "td_error"
        params["token_mode"] = "epsilon-greedy"
        params["initial_value"] = 0.25
        return mate.MATE(params)
    if algorithm_name == "MATE-TD-EPSGREEDY-I4":
        params["mate_mode"] = "td_error"
        params["token_mode"] = "epsilon-greedy"
        params["initial_value"] = 4
        return mate.MATE(params)
    if algorithm_name == "MATE-TD-EPSGREEDY-I025-E2":
        params["mate_mode"] = "td_error"
        params["token_mode"] = "epsilon-greedy"
        params["initial_value"] = 0.25
        params["epsilon"] = 2
        return mate.MATE(params)
    if algorithm_name == "MATE-TD-EPSGREEDY-CONT":
        params["mate_mode"] = "td_error"
        params["token_mode"] = "epsilon-greedy-cont"
        return mate.MATE(params)
    if algorithm_name == "MATE-TD-META":
        params["mate_mode"] = "td_error"
        params["token_mode"] = "meta-policy"
        return mate.MATE(params)
    if algorithm_name == "MATE-TD-DEFECT_COMPLETE":
        params["mate_mode"] = "td_error"
        params["defect_mode"] = mate.DEFECT_ALL
        return mate.MATE(params)
    if algorithm_name == "MATE-TD-DEFECT_REQUEST":
        params["mate_mode"] = "td_error"
        params["defect_mode"] = mate.DEFECT_SEND
        return mate.MATE(params)
    if algorithm_name == "MATE-TD-DEFECT_RESPONSE":
        params["mate_mode"] = "td_error"
        params["defect_mode"] = mate.DEFECT_RECEIVE
        return mate.MATE(params)
    raise ValueError("Unknown algorithm '{}'".format(algorithm_name))