from os.path import join
import mate.data as data
import numpy

def run_episode(env, controller, params, training_mode=True):
    done = False
    time_step = 0
    observations = env.reset()
    joint_probs_history = []
    request_messages_sent = 0
    response_messages_sent = 0
    token_values = []
    value_gradients = []
    values = []
    while not done:
        joint_action, joint_probs = controller.policy(observations)
        next_observations, rewards, done, info = env.step(joint_action)
        joint_probs_history.append(joint_probs)
        time_step += 1
        if training_mode:
            transition = controller.update(observations, joint_action, rewards, next_observations, done, info)
            request_messages_sent += transition["request_messages_sent"]
            response_messages_sent += transition["response_messages_sent"]
            token_values = transition["token_values"]
            value_gradients = transition["value_gradients"]
            values = transition["values"]
        observations = next_observations
    return {
        "discounted_returns": env.discounted_returns,
        "undiscounted_returns": env.undiscounted_returns,
        "domain_value": env.domain_values(),
        "sent_gifts": env.sent_gifts,
        "joint_probs_history": joint_probs_history,
        "request_messages_sent": request_messages_sent*1.0/time_step,
        "response_messages_sent": response_messages_sent*1.0/time_step,
        "messages_sent": (request_messages_sent+response_messages_sent)*1.0/time_step,
        "token_values": token_values,
        "value_gradients": value_gradients,
        "values": values
    }

def run_episodes(nr_episodes, env, controller, params, training_mode=True):
    discounted_returns = numpy.zeros(env.nr_agents)
    undiscounted_returns = numpy.zeros(env.nr_agents)
    domain_values = numpy.zeros(env.domain_values().shape)
    sent_gifts = numpy.zeros(env.nr_agents)
    request_messages_sent = 0
    response_messages_sent = 0
    messages_sent = 0
    token_values = []
    value_gradients = []
    values = []
    for _ in range(nr_episodes):
        result = run_episode(env, controller, params, training_mode)
        for i, dR, uR in zip(range(env.nr_agents), result["discounted_returns"], result["undiscounted_returns"]):
            discounted_returns[i] += (dR*1.0)/nr_episodes
            undiscounted_returns[i] += (uR*1.0)/nr_episodes
        domain_values += (result["domain_value"]*1.0)/nr_episodes
        sent_gifts += (result["sent_gifts"]*1.0)/nr_episodes
        request_messages_sent += (result["request_messages_sent"]*1.0)/nr_episodes
        response_messages_sent += (result["response_messages_sent"]*1.0)/nr_episodes
        messages_sent += (result["messages_sent"]*1.0)/nr_episodes
        token_values = result["token_values"]
        value_gradients = result["value_gradients"]
        values = result["values"]
    return {
        "discounted_returns": discounted_returns.tolist(),
        "undiscounted_returns": undiscounted_returns.tolist(),
        "domain_values": domain_values.tolist(),
        "sent_gifts": sent_gifts.tolist(),
        "request_messages_sent": request_messages_sent,
        "response_messages_sent": response_messages_sent,
        "messages_sent": messages_sent,
        "token_values": token_values,
        "value_gradients": value_gradients,
        "values": values
    }

def run_training(env, controller, params):
    episodes_per_epoch = params["episodes_per_epoch"]
    discounted_returns = [[] for _ in range(env.nr_agents)]
    undiscounted_returns = [[] for _ in range(env.nr_agents)]
    domain_values = []
    sent_gifts = []
    request_messages_sent = []
    response_messages_sent = []
    messages_sent = []
    token_values =[]
    value_gradients = []
    values = []
    for i in range(params["nr_epochs"]):
        result = run_episodes(episodes_per_epoch, env, controller, params, training_mode=True)
        print("Finished epoch {} ({}, {}, {} agents):".format(i, params["algorithm_name"], params["domain_name"], params["nr_agents"]))
        print("- Discounted return:  ", result["discounted_returns"], "->", numpy.sum(result["discounted_returns"]))
        print("- Undiscounted return:", result["undiscounted_returns"], "->", numpy.sum(result["undiscounted_returns"]))
        mean_domain_values = result["domain_values"]
        assert len(mean_domain_values) == len(env.domain_values())
        a, b = env.domain_value_debugging_indices()
        if abs(mean_domain_values[b]) > 1e-10:
            domain_value = mean_domain_values[a]/mean_domain_values[b]
        else:
            domain_value = 0
        domain_values.append(result["domain_values"])
        request_messages_sent.append(result["request_messages_sent"])
        response_messages_sent.append(result["response_messages_sent"])
        messages_sent.append(result["messages_sent"])
        token_values.append(result["token_values"])
        value_gradients.append(result["value_gradients"])
        values.append(result["values"])
        sent = result["sent_gifts"]
        sent_gifts.append(sent)
        print("- Domain value:", domain_value)
        print("- Sent gifts:", sent)
        for return_list, new_return in zip(discounted_returns, result["discounted_returns"]):
            return_list.append(float(new_return))
        for return_list, new_return in zip(undiscounted_returns, result["undiscounted_returns"]):
            return_list.append(float(new_return))
    result = {
        "discounted_returns": discounted_returns,
        "undiscounted_returns": undiscounted_returns,
        "domain_values": domain_values,
        "sent_gifts": sent_gifts,
        "request_messages_sent": request_messages_sent,
        "response_messages_sent": response_messages_sent,
        "messages_sent": messages_sent,
        "token_values": token_values,
        "value_gradients": value_gradients,
        "values": values
    }
    if "directory" in params:
        data.save_json(join(params["directory"], "results.json"), result)
        controller.save_model_weights(params["directory"])
    return result