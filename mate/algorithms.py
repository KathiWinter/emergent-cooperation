import random
import mate.controllers.controller as controller
import mate.controllers.actor_critic as actor_critic
import mate.controllers.gifting as gifting
import mate.controllers.mate as mate
import mate.controllers.lio as lio

def make(params):
    algorithm_name = params["algorithm_name"]
    
    #Baselines
    if algorithm_name == "IAC":
        return actor_critic.ActorCritic(params)
    if algorithm_name == "LIO":
        params["no_ppo"] = False
        return lio.LIO(params) 
    if algorithm_name == "Gifting-BUDGET":
        params["gifting_mode"] = gifting.BUDGET_MODE
        return gifting.Gifting(params)
    if algorithm_name == "Gifting-ZEROSUM":
        params["gifting_mode"] = gifting.ZERO_SUM_MODE
        return gifting.Gifting(params)
    
    #MATE
    if algorithm_name == "MATE-TD-T1":
        params["mate_mode"] = "td_error"
        params["fixed_token_mode"] = True
        params["fixed_token"] = True
        params["token_value"] = 1
        return mate.MATE(params)
    
    #AUTOMATE
    if algorithm_name == "MATE-TD-SYNC":
        params["mate_mode"] = "td_error"
        params["consensus_on"] = True
        return mate.MATE(params)
    if algorithm_name == "MATE-TD-AUTOMATE":
        params["mate_mode"] = "td_error"
        params["consensus_on"] = False
        return mate.MATE(params)
    if algorithm_name == "MATE-TD-SOV":
        params["mate_mode"] = "td_error"
        params["no_sync"] = True
        return mate.MATE(params)
    
    
    #Standard Token Range:
    if algorithm_name == "MATE-TD-T0.25":
        params["mate_mode"] = "td_error"
        params["fixed_token_mode"] = True
        params["fixed_token"] = True
        params["token_value"] = 0.25
        return mate.MATE(params)
    if algorithm_name == "MATE-TD-T0.5":
        params["mate_mode"] = "td_error"
        params["fixed_token_mode"] = True
        params["fixed_token"] = True
        params["token_value"] = 0.5
        return mate.MATE(params)
    if algorithm_name == "MATE-TD-T2":
        params["mate_mode"] = "td_error"
        params["fixed_token_mode"] = True
        params["fixed_token"] = True
        params["token_value"] = 2
        return mate.MATE(params)
    if algorithm_name == "MATE-TD-T4":
        params["mate_mode"] = "td_error"
        params["fixed_token_mode"] = True
        params["fixed_token"] = True
        params["token_value"] = 4
        return mate.MATE(params)
    
    #Extended Token Range:
    if algorithm_name == "MATE-TD-T0":
        params["mate_mode"] = "td_error"
        params["fixed_token_mode"] = True
        params["fixed_token"] = True
        params["token_value"] = 0
        return mate.MATE(params)
    if algorithm_name == "MATE-TD-T0.75":
        params["mate_mode"] = "td_error"
        params["fixed_token_mode"] = True
        params["fixed_token"] = True
        params["token_value"] = 0.75
        return mate.MATE(params)
    if algorithm_name == "MATE-TD-T1.5":
        params["mate_mode"] = "td_error"
        params["fixed_token_mode"] = True
        params["fixed_token"] = True
        params["token_value"] = 1.5
        return mate.MATE(params)
    if algorithm_name == "MATE-TD-T2.5":
        params["mate_mode"] = "td_error"
        params["fixed_token_mode"] = True
        params["fixed_token"] = True
        params["token_value"] = 2.5
        return mate.MATE(params)    
    if algorithm_name == "MATE-TD-T3":
        params["mate_mode"] = "td_error"
        params["fixed_token_mode"] = True
        params["fixed_token"] = True
        params["token_value"] = 3
        return mate.MATE(params)    
    if algorithm_name == "MATE-TD-T8":
        params["mate_mode"] = "td_error"
        params["fixed_token_mode"] = True
        params["fixed_token"] = True
        params["token_value"] = 8
        return mate.MATE(params)

    
    #Individual Decentralized Tokens for 2 agents only
    if algorithm_name == "MATE-TD-T0.25-0.5":
        params["mate_mode"] = "td_error"
        params["fixed_token_mode"] = True
        params["fixed_token"] = True
        params["token_value"] = [0.25, 0.5]
        return mate.MATE(params)
    if algorithm_name == "MATE-TD-T0.25-1":
        params["mate_mode"] = "td_error"
        params["fixed_token_mode"] = True
        params["fixed_token"] = True
        params["token_value"] = [0.25, 1]
        return mate.MATE(params)
    if algorithm_name == "MATE-TD-T0.25-2":
        params["mate_mode"] = "td_error"
        params["fixed_token_mode"] = True
        params["fixed_token"] = True
        params["token_value"] = [0.25, 2]
        return mate.MATE(params)
    if algorithm_name == "MATE-TD-T0.25-4":
        params["mate_mode"] = "td_error"
        params["fixed_token_mode"] = True
        params["fixed_token"] = True
        params["token_value"] = [0.25, 4]
        return mate.MATE(params)
    if algorithm_name == "MATE-TD-T0.5-1":
        params["mate_mode"] = "td_error"
        params["fixed_token_mode"] = True
        params["fixed_token"] = True
        params["token_value"] = [0.5, 1]
        return mate.MATE(params)
    if algorithm_name == "MATE-TD-T0.5-2":
        params["mate_mode"] = "td_error"
        params["fixed_token_mode"] = True
        params["fixed_token"] = True
        params["token_value"] = [0.5, 2]
        return mate.MATE(params)
    if algorithm_name == "MATE-TD-T0.5-4":
        params["mate_mode"] = "td_error"
        params["fixed_token_mode"] = True
        params["fixed_token"] = True
        params["token_value"] = [0.5, 4]
        return mate.MATE(params)
    if algorithm_name == "MATE-TD-T1-2":
        params["mate_mode"] = "td_error"
        params["fixed_token_mode"] = True
        params["fixed_token"] = True
        params["token_value"] = [1, 2]
        return mate.MATE(params)
    if algorithm_name == "MATE-TD-T1-4":
        params["mate_mode"] = "td_error"
        params["fixed_token_mode"] = True
        params["fixed_token"] = True
        params["token_value"] = [1, 4]
        return mate.MATE(params)
    if algorithm_name == "MATE-TD-T2-4":
        params["mate_mode"] = "td_error"
        params["fixed_token_mode"] = True
        params["fixed_token"] = True
        params["token_value"] = [2, 4]
        return mate.MATE(params)
    
    #Random (episode-wise)
    if algorithm_name == "MATE-TD-RANDOM":
        params["mate_mode"] = "td_error"
        params["random_mode"] = "epoch"
        return mate.MATE(params)
    #Random (step-wise) centralized
    if algorithm_name == "MATE-TD-RANDOM-TS":
        params["mate_mode"] = "td_error"
        params["architecture"] = "centralized"
        params["random_mode"] = "time_step"
        return mate.MATE(params)
    #Reflecting 
    if algorithm_name == "MATE-TD-REFLECTING":
        params["mate_mode"] = "td_error"
        params["architecture"] = "reflecting"
        params["random_mode"] = "time_step"
        return mate.MATE(params)     
    #Holding
    if algorithm_name == "MATE-TD-HOLDING":
        params["mate_mode"] = "td_error"
        params["architecture"] = "holding"
        params["random_mode"] = "time_step"
        return mate.MATE(params)  
    
    #UCB 
    if algorithm_name == "MATE-TD-UCB-CENT":
        params["mate_mode"] = "td_error"
        params["ucb_mode"] = "centralized"
        params["token_value"] = random.choice([0.25, 0.5, 1.0, 2.0, 4.0])
        return mate.MATE(params)
    if algorithm_name == "MATE-TD-UCB-DEC":
        params["mate_mode"] = "td_error"
        params["ucb_mode"] = "decentralized"
        params["token_value"] = random.choice([0.25, 0.5, 1.0, 2.0, 4.0])
        return mate.MATE(params)    
    
    #Miscellaneous
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
    
    if algorithm_name == "Random":
        return controller.Controller(params)
    
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
    
    raise ValueError("Unknown algorithm '{}'".format(algorithm_name))
