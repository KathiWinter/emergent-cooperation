import uuid
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

"""
 Mutual Acknowledgment Token Exchange (MATE)
"""
class MATE(ActorCritic):

    def __init__(self, params):
        super(MATE, self).__init__(params)
        self.last_rewards_observed = [[] for _ in range(self.nr_agents)]
        self.mate_mode = get_param_or_default(params, "mate_mode", STATIC_MODE)
        self.token_value = get_param_or_default(params, "token_value", [0 for _ in range(self.nr_agents)])#numpy.zeros(self.nr_agents, dtype=float))
        self.trust_request_matrix = numpy.zeros((self.nr_agents, self.nr_agents), dtype=float)
        self.trust_response_matrix = numpy.zeros((self.nr_agents, self.nr_agents), dtype=float)
        self.defect_mode = get_param_or_default(params, "defect_mode", NO_DEFECT)
        self.token_send_matrix = numpy.zeros((self.nr_agents, self.nr_agents), dtype=float)
        self.token_response_matrix = numpy.zeros((self.nr_agents, self.nr_agents), dtype=float)
        self.token_shares = [[] for _ in range(self.nr_agents)]
        self.share_list = [[] for _ in range(self.nr_agents)]
        self.values = numpy.zeros(self.nr_agents, dtype=float)
        self.epoch_values = [[] for _ in range(self.nr_agents)]
        self.last_values = [[] for _ in range(self.nr_agents)]
        self.episode_step = 0
        self.consensus_on = get_param_or_default(params, "consensus_on", True)
        self.mean_reward = [0 for _ in range(self.nr_agents)]
        self.max_reward = [-numpy.inf for _ in range(self.nr_agents)]
        self.rewards = [[] for _ in range(self.nr_agents)]
        self.episode_return = numpy.zeros(self.nr_agents, dtype=float)
        self.update_rate = [[] for _ in range(self.nr_agents)]
        self.new_value = [False for _ in range(self.nr_agents)]
        self.share_id = [0 for _ in range(self.nr_agents)]
        self.episode = 0

 
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
            history = torch.tensor(numpy.array([history]), dtype=torch.float, device=self.device)
            next_history = torch.tensor(numpy.array([next_history]), dtype=torch.float, device=self.device)
            value = self.get_values(agent_id, history)[0].item()
            next_value = self.get_values(agent_id, next_history)[0].item()
            return reward + self.gamma*next_value - value >= 0
        if self.mate_mode == VALUE_DECOMPOSE_MODE:
            return False
        
    def generate_token_shares(self, neighborhood_size, total):
        lower_bound = -total
        upper_bound = +total
        shares = [random.uniform(lower_bound, upper_bound) for _ in range(neighborhood_size)]
        last_share = total - sum(shares)
        shares.append(last_share)
        return shares

    def generate_id(self):
        id = uuid.uuid4()
        return id

    def prepare_transition(self, joint_histories, joint_action, rewards, next_joint_histories, done, info):
        transition = super(MATE, self).prepare_transition(joint_histories, joint_action, rewards, next_joint_histories, done, info)
        original_rewards = [r for r in rewards]
        
        self.trust_request_matrix[:] = 0
        self.trust_response_matrix[:] = 0
        self.episode_step += 1
        self.episode_return += rewards
        
        if self.consensus_on == False:
            self.token_value = [1 for _ in range(self.nr_agents)] 
       
        for i in range(self.nr_agents):
 
            for r in rewards:
                if r != 0 and not r in self.rewards[i]:
                    self.rewards[i].append(r)
            if len(self.rewards[i]) > 0:
                if numpy.max(self.rewards[i]) > self.max_reward[i]:
                    self.max_reward[i] = numpy.max(self.rewards[i])
                self.mean_reward[i] = abs(numpy.min(self.rewards[i]))/2

        if done and self.consensus_on:
            self.episode += 1
            for i in range(self.nr_agents):
                self.epoch_values[i].append(self.values[i])
                self.values[i] = 0
            
                if self.episode % 10 == 0:
                    # derive token value from value function
                    self.token_send_matrix[:] = 0
                    self.token_response_matrix[:] = 0
                    self.token_shares[i] = []
                    self.share_list = [[] for _ in range(self.nr_agents)]
                    if self.episode > 9:
                        if len(self.last_values[i]) > 0:
                            value_gradient = float(numpy.median(self.epoch_values[i]) - numpy.median(self.last_values[i])) / abs(numpy.median(self.last_values[i]))
                        else:
                            value_gradient = 0
                        transition["value_gradients"][i] = value_gradient
                        transition["values"][i] = numpy.median(self.epoch_values[i])
                        print("value: ", numpy.median(self.epoch_values[i]) , "last value: ",numpy.median(self.last_values[i]) )
                
                        token_update = value_gradient 
                        update_rate = 0.1 * self.mean_reward[i] 
                        
                        # if value change is too small
                        if abs(token_update) == numpy.inf:
                            token_update = 0.0 

                        if self.max_reward[i] > 0:
                            sign = 1
                        else:
                            sign = -1
                        self.token_value[i] = self.token_value[i] + token_update * update_rate * sign
                      
                        
                        # prevent negative token values
                        self.token_value[i] = numpy.maximum(0.0, self.token_value[i])
                        self.new_value[i] = True
                    
                    #reset episode parameters
                    self.last_values[i] = self.epoch_values[i]
                    self.epoch_values[i] = []
        
            self.episode_step = 0   
            self.rewards = [[] for _ in range(self.nr_agents)] 
            
        # 1. Send trust requests
        defector_id = -1
        if self.defect_mode != NO_DEFECT:
            defector_id = numpy.random.randint(0, self.nr_agents)
        request_receive_enabled = [self.sample_no_comm_failure() for _ in range(self.nr_agents)]
        for i, reward, history, next_history in zip(range(self.nr_agents), original_rewards, joint_histories, next_joint_histories):
            self.values[i] += self.get_values(i, torch.tensor(numpy.array([history]), dtype=torch.float, device=self.device))[0].item()
            neighborhood = info["neighbor_agents"][i]
            if done and self.consensus_on and self.new_value[i]:
                self.token_shares[i] = self.generate_token_shares(len(neighborhood), self.token_value[i])
                self.token_send_matrix[i][i] = self.token_shares[i][0]
            requests_enabled = i != defector_id or self.defect_mode not in [DEFECT_ALL, DEFECT_SEND]
            requests_enabled = requests_enabled and self.sample_no_comm_failure()
            next_index = 1
            for j in neighborhood:
                if requests_enabled:
                    if done and self.consensus_on and self.new_value[i]: 
                        self.token_send_matrix[j][i] = self.token_shares[i][next_index]
                        next_index += 1
                    assert i != j
                    if self.can_rely_on(i, reward, history, next_history): # Analyze the "winners" of that step
                        self.trust_request_matrix[j][i] += self.token_value[i]
                        transition["request_messages_sent"] += 1

        # 2. Send trust responses
        for i, history, next_history in zip(range(self.nr_agents), joint_histories, next_joint_histories):
            if done and self.consensus_on and self.new_value[i]:
                summed_token_shares = sum(self.token_send_matrix[i])                                         
                share_id = self.generate_id()
                self.share_list[i].append((summed_token_shares, share_id))
                self.new_value[i] = False
            neighborhood = info["neighbor_agents"][i]
            respond_enabled = i != defector_id or self.defect_mode not in [DEFECT_ALL, DEFECT_RESPONSE]
            respond_enabled = respond_enabled and self.sample_no_comm_failure()
            if request_receive_enabled[i]:
                trust_requests = [self.trust_request_matrix[i][x] for x in neighborhood]
                if len(trust_requests) > 0:
                    transition["rewards"][i] += numpy.max(trust_requests)
            if respond_enabled and len(neighborhood) > 0:
                if self.can_rely_on(i, transition["rewards"][i], history, next_history):
                    accept_trust = self.token_value[i]
                else:
                    accept_trust = -self.token_value[i]
                   
                for j in neighborhood:
                    assert i != j
                    if self.consensus_on and len(self.share_list[i])>0:
                        for x in self.share_list[i]:
                            if x not in self.share_list[j]:
                                self.share_list[j].append(x)
                    if self.trust_request_matrix[i][j] > 0:
                        self.trust_response_matrix[j][i] = accept_trust
                        if accept_trust > 0:
                            transition["response_messages_sent"] += 1
        # 3. Receive trust responses
        for i, trust_responses in enumerate(self.trust_response_matrix):
            neighborhood = info["neighbor_agents"][i]
            receive_enabled = i != defector_id or self.defect_mode not in [DEFECT_ALL, DEFECT_RECEIVE]
            receive_enabled = receive_enabled and self.sample_no_comm_failure()
            if receive_enabled and len(neighborhood) > 0:
                if self.consensus_on:
                    if len(self.share_list[i]) > 0:
                        self.token_value[i] = sum([x[0] for x in self.share_list[i]])/len(self.share_list[i])
                if trust_responses.any():
                    filtered_trust_responses = [trust_responses[x] for x in neighborhood if abs(trust_responses[x]) > 0]
                    if len(filtered_trust_responses) > 0:
                        transition["rewards"][i] += min(filtered_trust_responses)
        if done:
            

            self.last_rewards_observed = [[] for _ in range(self.nr_agents)]
            self.episode_return = numpy.zeros(self.nr_agents, dtype=float)
            print(self.token_value)
            for i in range(self.nr_agents):
                transition["token_values"][i].append(self.token_value[i])
        return transition
