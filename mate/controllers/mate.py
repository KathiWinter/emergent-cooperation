from mate.utils import get_param_or_default
from mate.controllers.actor_critic import ActorCritic
import torch
import numpy
import random

STATIC_MODE = "static"
TD_ERROR_MODE = "td_error"
VALUE_DECOMPOSE_MODE = "value_decompose"
MATE_MODES = [STATIC_MODE, TD_ERROR_MODE, VALUE_DECOMPOSE_MODE]

NO_DEFECT = 0
DEFECT_ALL = 1 # Does not send or receive any acknowledgment messages
DEFECT_RESPONSE = 2 # Sends acknowledgment requests but does not respond to incoming requests 
DEFECT_RECEIVE = 3 # Sends acknowledgment requests but does not receive any responses
DEFECT_SEND = 4 # Receives acknowledgment requests but does send any requests itself

DEFECT_MODES = [NO_DEFECT, DEFECT_ALL, DEFECT_RESPONSE, DEFECT_RECEIVE, DEFECT_SEND]

FIXED = "fixed"
RANDOM = "random"
UCB = "ucb"
EPSGREEDY = "epsilon-greedy"
TOKEN_MODES = [FIXED, RANDOM, UCB, EPSGREEDY]

"""
 Mutual Acknowledgment Token Exchange (MATE)
"""
class MATE(ActorCritic):

    def __init__(self, params):
        super(MATE, self).__init__(params)
        self.last_rewards_observed = [[] for _ in range(self.nr_agents)]
        self.mate_mode = get_param_or_default(params, "mate_mode", STATIC_MODE)
        self.token_value = get_param_or_default(params, "token_value", 1.0)
        self.token_range = get_param_or_default(params, "token_range", [0.25, 0.5, 1.0, 2.0, 4.0])
        self.trust_request_matrix = numpy.zeros((self.nr_agents, self.nr_agents), dtype=numpy.float32)
        self.trust_response_matrix = numpy.zeros((self.nr_agents, self.nr_agents), dtype=numpy.float32)
        self.defect_mode = get_param_or_default(params, "defect_mode", NO_DEFECT)
        self.token_mode = get_param_or_default(params, "token_mode", FIXED)
        
        ##UCB 
        self.tokens_dict = {}
        self.episode = 0
        self.episode_return = 0
        self.epsilon = 0.1
        self.best_value = 0.0
        self.last_token_value = self.token_value

    def can_rely_on(self, agent_id, reward, history, next_history):
        if self.mate_mode == STATIC_MODE:
            is_empty = self.last_rewards_observed[agent_id]
            if is_empty:
                self.last_rewards_observed[agent_id].append(reward)
                return True
            last_reward = numpy.mean(self.last_rewards_observed[agent_id])
            self.last_rewards_observed[agent_id].append(reward)
            return reward >= last_reward
        if self.mate_mode == TD_ERROR_MODE:
            history = torch.tensor(numpy.array([history]), dtype=torch.float32, device=self.device)
            next_history = torch.tensor(numpy.array([next_history]), dtype=torch.float32, device=self.device)
            value = self.get_values(agent_id, history)[0].item()
            next_value = self.get_values(agent_id, next_history)[0].item()
            return reward + self.gamma*next_value - value >= 0
        if self.mate_mode == VALUE_DECOMPOSE_MODE:
            return False


    def prepare_transition(self, joint_histories, joint_action, rewards, next_joint_histories, done, info):
        transition = super(MATE, self).prepare_transition(joint_histories, joint_action, rewards, next_joint_histories, done, info)
        
        if self.token_mode == FIXED:
            token_value = self.token_value
        if self.token_mode == RANDOM:
            token_value = random.choice(self.token_range)
        if self.token_mode == UCB:
            self.episode_return += rewards
            token_value = self.last_token_value
            if done:
                self.episode += 1
                max_upper_bound = 0
                if(str(token_value) not in self.tokens_dict):
                    self.tokens_dict[str(token_value)] = {'rewards': []} 
                self.tokens_dict[str(token_value)]['rewards'].append(self.episode_return)
                if(len(self.tokens_dict[str(token_value)]['rewards']) > 50):
                    self.tokens_dict[str(token_value)]['rewards'].pop(0) 
                
                for token, stats in self.tokens_dict.items():
                    if(len(stats['rewards']) > 0):
                        mean_reward = numpy.sum(stats['rewards']) / len(stats['rewards'])
                        di = numpy.sqrt((3/2 * numpy.log(self.episode + 1)) / len(stats['rewards']))
                        upper_bound = mean_reward + di
                    else:
                        upper_bound = 1e400
                    if(upper_bound > max_upper_bound):
                        max_upper_bound = upper_bound
                        self.best_value = token
                p = random.uniform(0, 1) 
                if p < self.epsilon: 
                    token_value = random.choice([0.25, 0.5, 1.0, 2.0, 4.0])
                else:                 
                    token_value = self.best_value       
                self.last_token_value = token_value
    
                self.episode_return = 0
                transition["token_value"] = token_value

        original_rewards = [r for r in rewards]
        self.trust_request_matrix[:] = 0
        self.trust_response_matrix[:] = 0
        # 1. Send trust requests
        defector_id = -1
        if self.defect_mode != NO_DEFECT:
            defector_id = numpy.random.randint(0, self.nr_agents)
        request_receive_enabled = [self.sample_no_comm_failure() for _ in range(self.nr_agents)]
        for i, reward, history, next_history in zip(range(self.nr_agents), original_rewards, joint_histories, next_joint_histories):
            requests_enabled = i != defector_id or self.defect_mode not in [DEFECT_ALL, DEFECT_SEND]
            requests_enabled = requests_enabled and self.sample_no_comm_failure()
            if requests_enabled and self.can_rely_on(i, reward, history, next_history): # Analyze the "winners" of that step
                neighborhood = info["neighbor_agents"][i]
                for j in neighborhood:
                    assert i != j
                    self.trust_request_matrix[j][i] += numpy.float32(token_value)
                    transition["request_messages_sent"] += 1
        # 2. Send trust responses
        for i, history, next_history in zip(range(self.nr_agents), joint_histories, next_joint_histories):
            neighborhood = info["neighbor_agents"][i]
            respond_enabled = i != defector_id or self.defect_mode not in [DEFECT_ALL, DEFECT_RESPONSE]
            respond_enabled = respond_enabled and self.sample_no_comm_failure()
            if request_receive_enabled[i]:
                trust_requests = [self.trust_request_matrix[i][x] for x in neighborhood]
                if len(trust_requests) > 0:
                    transition["rewards"][i] += numpy.max(trust_requests)
            if respond_enabled and len(neighborhood) > 0:
                if self.can_rely_on(i, transition["rewards"][i], history, next_history):
                    accept_trust = 1
                else:
                    accept_trust = -1
                for j in neighborhood:
                    assert i != j
                    if self.trust_request_matrix[i][j] > 0:
                        self.trust_response_matrix[j][i] = accept_trust * numpy.float32(token_value)
                        if accept_trust > 0:
                            transition["response_messages_sent"] += 1
        # 3. Receive trust responses
        for i, trust_responses in enumerate(self.trust_response_matrix):
            neighborhood = info["neighbor_agents"][i]
            receive_enabled = i != defector_id or self.defect_mode not in [DEFECT_ALL, DEFECT_RECEIVE]
            receive_enabled = receive_enabled and self.sample_no_comm_failure()
            if receive_enabled and len(neighborhood) > 0 and trust_responses.any():
                filtered_trust_responses = [trust_responses[x] for x in neighborhood if abs(trust_responses[x]) > 0]
                if len(filtered_trust_responses) > 0:
                    transition["rewards"][i] += min(filtered_trust_responses)
        if done:
            self.last_rewards_observed = [[] for _ in range(self.nr_agents)]
            print(token_value)
        return transition
