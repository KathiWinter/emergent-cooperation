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

FIXED_TOKEN = "fixed"
RANDOM_TOKEN = "random"
EPSILON_GREEDY = "epsilon-greedy"
UCB = "ucb"
TOKEN_MODES = [FIXED_TOKEN, RANDOM_TOKEN, EPSILON_GREEDY, UCB]

"""
 Mutual Acknowledgment Token Exchange (MATE)
"""
class MATE(ActorCritic):

    def __init__(self, params):
        super(MATE, self).__init__(params)
        self.last_rewards_observed = [[] for _ in range(self.nr_agents)]
        self.mate_mode = get_param_or_default(params, "mate_mode", STATIC_MODE)
        self.token_value = get_param_or_default(params, "token_value", 1)
        self.token_range = get_param_or_default(params, "token_range", [0.25, 0.5, 1.0, 2.0, 4.0])
        self.epsilon = get_param_or_default(params, "epsilon", 0.2)
        self.trust_request_matrix = numpy.zeros((self.nr_agents, self.nr_agents), dtype=numpy.float32)
        self.trust_response_matrix = numpy.zeros((self.nr_agents, self.nr_agents), dtype=numpy.float32)
        self.defect_mode = get_param_or_default(params, "defect_mode", NO_DEFECT)
        self.token_mode = get_param_or_default(params, "token_mode", FIXED_TOKEN)
        self.greedy_value = [1, 1]
        self.tmp_token_value = [1, 1]
        self.best_value = [0 for _ in range(self.nr_agents)]
        self.tokens_dict = [{} for _ in range(self.nr_agents)]
        self.episode = 0
        self.episode_return = [0 for _ in range(self.nr_agents)]
        self.last_token_value = [random.choice([0.25, 0.5, 1.0, 2.0, 4.0]) for _ in range(self.nr_agents)]
  
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

        if self.token_mode == FIXED_TOKEN:
            token_value = self.token_value
        if self.token_mode == RANDOM_TOKEN:
            token_value = [0,0]
            for i in range(self.nr_agents):
                token_value[i] = random.choice([0.25, 0.5, 1, 2, 4, 5, 6, 7, 8, 9])
        if self.token_mode == EPSILON_GREEDY:  
            original_rewards = [r for r in rewards]
            for i, reward, history, next_history in zip(range(self.nr_agents), original_rewards, joint_histories, next_joint_histories):
                if self.can_rely_on(i, reward, history, next_history):
                    self.greedy_value[i] = self.tmp_token_value[i] 
            p = random.uniform(0, 1)  
            if p < self.epsilon:
                token_value = [random.choice([0.25, 0.5, 1, 2, 4]), random.choice([0.25, 0.5, 1.0, 2.0, 4.0])]
            else:                 
                token_value = self.greedy_value
            self.tmp_token_value = token_value
        
        # if self.token_mode == UCB: #individual
        #     token_value = [random.choice([0.25, 0.5, 1.0, 2.0, 4.0]) for _ in range(self.nr_agents)]
        #     for i in range(self.nr_agents):
        #         self.episode_return[i] += rewards[i]
        #         token_value[i] = self.last_token_value[i]
        #     if done:
        #         self.episode += 1
        #         if self.episode % 10 == 1:
        #             for i in range(self.nr_agents):
        #                 max_upper_bound = -numpy.inf
        #                 if(str(self.last_token_value[i]) not in self.tokens_dict[i]):
        #                     self.tokens_dict[i][str(self.last_token_value[i])] = {'rewards': [],} 
        #                 self.tokens_dict[i][str(self.last_token_value[i])]['rewards'].append([self.episode, self.episode_return[i]])
                                                          
        #                 for token, stats in self.tokens_dict[i].items():
        #                     if(len(stats['rewards']) > 0):
        #                         sum_rewards = sum([x[1] for x in stats['rewards']])
        #                         mean_reward = sum_rewards / len(stats['rewards'])
        #                         di = numpy.sqrt((2 * numpy.log(self.episode)) / len(stats['rewards']))
        #                         upper_bound = mean_reward + di
        #                         #print(token, "upper bound: ", upper_bound )
        #                     else:
        #                         upper_bound = 1e400
        #                     if(upper_bound > max_upper_bound):
        #                         max_upper_bound = upper_bound
        #                         self.best_value[i] = float(token)

        #                 token_value[i] = self.best_value[i]   
        #                 if self.episode-1 < len(self.token_range)*10:
        #                     index = int(self.episode / 10)
        #                     token_value[i] = self.token_range[index]
                            
        #                 self.episode_return[i] = 0
        #                 self.last_token_value[i] = token_value[i]
        #                 print(token_value)    
        #         transition["token_values"].append(token_value)
                        
        if self.token_mode == UCB: #central
            token_value = random.choice([0.25, 0.5, 1.0, 2.0, 4.0])
            for i in range(self.nr_agents):
                self.episode_return[0] += rewards[i]
            token_value = self.last_token_value[0]
            if done:
                self.episode += 1
                if self.episode % 10 == 1:
                    max_upper_bound = -numpy.inf
                    if(str(self.last_token_value[0]) not in self.tokens_dict[0]):
                        self.tokens_dict[0][str(self.last_token_value[0])] = {'rewards': [],} 
                    self.tokens_dict[0][str(self.last_token_value[0])]['rewards'].append([self.episode, self.episode_return[0]])
                    
                    #NOT WORKING. Just for the RECORD:
                    # for token, stats in self.tokens_dict[0].items():
                    #     for x in stats['rewards']:
                    #         if x[0] < self.episode - 50:
                    #             self.tokens_dict[0][str(token)]['rewards'].remove(x)
                                
                    for token, stats in self.tokens_dict[0].items():
                        if(len(stats['rewards']) > 0):
                            sum_rewards = sum([x[1] for x in stats['rewards']])
                            mean_reward = sum_rewards / len(stats['rewards'])
                            di = numpy.sqrt((2 * numpy.log(self.episode)) / len(stats['rewards']))
                            upper_bound = mean_reward + di
                            #print(token, "upper bound: ", upper_bound )
                        else:
                            upper_bound = 1e400
                        if(upper_bound > max_upper_bound):
                            max_upper_bound = upper_bound
                            self.best_value[0] = float(token)

                    token_value = self.best_value[0]   
                    if self.episode-1 < len(self.token_range)*10:
                        index = int(self.episode / 10)
                        token_value = self.token_range[index]
                        
                    self.episode_return[0] = 0
                        
                    
                    print(token_value)
                    self.last_token_value[0] = token_value
                transition["token_values"].append(token_value)
                 
 
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
                    self.trust_request_matrix[j][i] += token_value#[i]
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
                        self.trust_response_matrix[j][i] = accept_trust * token_value#[i]
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
        return transition
