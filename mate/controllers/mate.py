from mate.utils import get_param_or_default
from mate.controllers.actor_critic import ActorCritic
import copy
import torch
import torch.nn
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
DYNAMIC_TOKEN = "dynamic-token"
META = "meta-policy"
TOKEN_MODES = [FIXED_TOKEN, RANDOM_TOKEN, EPSILON_GREEDY, UCB, META, DYNAMIC_TOKEN]

"""
 Mutual Acknowledgment Token Exchange (MATE)
"""
class MATE(ActorCritic):

    def __init__(self, params):
        super(MATE, self).__init__(params)
        self.last_rewards_observed = [[] for _ in range(self.nr_agents)]
        self.mate_mode = get_param_or_default(params, "mate_mode", STATIC_MODE)
        self.token_value = get_param_or_default(params, "token_value", 1)
        self.token_range = get_param_or_default(params, "token_range", [0.5, 1, 2])
        self.trust_request_matrix = numpy.zeros((self.nr_agents, self.nr_agents), dtype=int)
        self.trust_response_matrix = numpy.zeros((self.nr_agents, self.nr_agents), dtype=int)
        self.defect_mode = get_param_or_default(params, "defect_mode", NO_DEFECT)
        self.token_mode = get_param_or_default(params, "token_mode", FIXED_TOKEN)
        self.epsilon = get_param_or_default(params, "epsilon", 0.9)
        self.initial_value = get_param_or_default(params, "initial_value", random.choice([0.25, 0.5, 1.0, 2.0, 4.0]))
        self.best_value = copy.copy(self.initial_value)
        self.last_token_value = random.choice([0.25, 0.5, 1.0, 2.0, 4.0])#numpy.zeros(self.nr_agents)
        self.tokens_dict = {}
        self.Q_table = numpy.zeros((2, 36, 128)) 
        self.Q_table_joint = numpy.zeros((10, 10, 10, 10, 5)) 
        self.alpha = 0.5
        self.step = 0
        self.min = 0.0


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
        
    def get_token(self, agent_id, reward, history, next_history):
        history = torch.tensor(numpy.array([history]), dtype=torch.float32, device=self.device)
        next_history = torch.tensor(numpy.array([next_history]), dtype=torch.float32, device=self.device)
        value = self.get_values(agent_id, history)[0].item()
        next_value = self.get_values(agent_id, next_history)[0].item()
        token_value = abs(reward + self.gamma*next_value - value)
        return token_value 
    
    def updateQTable(self, agent_id, reward, token, new_state, last_state, new_value):        
        maxQ = max(self.Q_table[agent_id, new_state, :])
        token = int(token*4)
        self.Q_table[agent_id, last_state, token] = (1-self.alpha) * self.Q_table[agent_id,last_state, token] + self.alpha*(reward + self.gamma * new_value + self.gamma * maxQ)
        #print(self.Q_table[agent_id, last_state, token] )
        #print(self.alpha*(reward + self.gamma * new_value + self.gamma * maxQ))
        
    def updateQTable_joint(self, rewards, token, agent0_new, agent1_new, coin0_new, coin1_new, agent0_last, agent1_last, coin0_last, coin1_last, new_value): 
        maxQ = max(self.Q_table_joint[agent0_new, agent1_new, coin0_new, coin1_new, :])
        if(token == 0.25):
            token = 0
        elif(token == 0.5):
            token = 1
        elif(token == 1.0):
            token = 2
        elif(token == 2.0):
            token = 3
        elif(token == 4.0):
            token = 4
        self.Q_table_joint[agent0_last, agent1_last, coin0_last, coin1_last, token] = (1-self.alpha) * self.Q_table_joint[agent0_last, agent1_last, coin0_last, coin1_last, token] + self.alpha*(rewards + self.gamma * maxQ)
        )

    def prepare_transition(self, joint_histories, joint_action, rewards, next_joint_histories, done, info):
        transition = super(MATE, self).prepare_transition(joint_histories, joint_action, rewards, next_joint_histories, done, info)

        if self.token_mode == FIXED_TOKEN:
            token_value = self.token_value
        if self.token_mode == RANDOM_TOKEN:
            token_value = random.choice(self.token_range)
        if self.token_mode == DYNAMIC_TOKEN:
            token_value = numpy.zeros(self.nr_agents)
            for i, history, next_history in zip(range(self.nr_agents), joint_histories, next_joint_histories):
                if self.can_rely_on(i, transition["rewards"][i], history, next_history):
                    token_value[i] = self.get_token(i, transition["rewards"][i], history, next_history) 
        ## INDIVIDUAL QLEARNING ##         
                    #print(token_value[i])
        # if self.token_mode == META:
        #     self.step += 1
        #     token_value = numpy.zeros(self.nr_agents)
        #     if(self.epsilon > 0.2 and self.step % 5000 == 0):
        #         self.epsilon -= 0.1
        #     for i, history, next_history in zip(range(self.nr_agents), joint_histories, next_joint_histories):
        #         history = torch.tensor(numpy.array([history]), dtype=torch.float32, device=self.device)
        #         next_history = torch.tensor(numpy.array([next_history]), dtype=torch.float32, device=self.device)
        #         new_value = self.get_values(i, next_history).item()
        #         new_state = 0
        #         if(1.0 in next_history.tolist()[0][0][18:27]):
        #             new_state = 1
        #         last_state = 0
        #         if(1.0 in history.tolist()[0][0][18:27]):
        #             last_state = 1
        #         self.updateQTable(i, transition["rewards"][i], self.last_token_value[i], new_state, last_state, new_value)
        #         p = random.uniform(0, 1)  
        #         if p < self.epsilon:
        #             token_value[i] = random.choice([0.25, 0.5, 1.0, 2.0, 4.0])
        #         else:
        #             token_value[i] = 0.25 * numpy.argmax(self.Q_table[i, new_state,:])
                
        #     self.last_token_value = token_value
        
        if self.token_mode == META:
            self.step += 1
            if(self.epsilon > 0.2 and self.step % 5000 == 0):
                self.epsilon -= 0.1
            
            ##TODO: Hier korrekt machen, nicht nur vom ersten Agenten nehmen
            history = torch.tensor(numpy.array([joint_histories[0]]), dtype=torch.float32, device=self.device)
            next_history = torch.tensor(numpy.array([next_joint_histories[0]]), dtype=torch.float32, device=self.device)
            new_value = self.get_values(0, next_history)[0].item()+self.get_values(1, next_history)[0].item()

            agent0_new= next_history.tolist()[0][0][0:9].index(1.0)
            agent1_new= next_history.tolist()[0][0][9:18].index(1.0)
            coin0_new = 9
            coin1_new = 9
            if(1.0 in next_history.tolist()[0][0][18:27]):
                coin0_new = next_history.tolist()[0][0][18:27].index(1.0)
            if(1.0 in next_history.tolist()[0][0][27:36]):
                coin1_new = next_history.tolist()[0][0][27:36].index(1.0)
            
            agent0_last= history.tolist()[0][0][0:9].index(1.0)
            agent1_last= history.tolist()[0][0][9:18].index(1.0)
            coin0_last = 9
            coin1_last = 9
            if(1.0 in history.tolist()[0][0][18:27]):
                coin0_last = history.tolist()[0][0][18:27].index(1.0)
            if(1.0 in history.tolist()[0][0][27:36]):
                coin1_last = history.tolist()[0][0][27:36].index(1.0)
            
       
                
            #print(self.token_values_dict[(str(self.last_token_value))])
            self.updateQTable_joint(sum(transition["rewards"]), self.last_token_value, agent0_new, agent1_new, coin0_new, coin1_new, agent0_last, agent1_last, coin0_last, coin1_last, new_value)
            p = random.uniform(0, 1)  
            if p < self.epsilon:
                token_value = random.choice([0.25, 0.5, 1.0, 2.0, 4.0])
            else:
                value = numpy.argmax(self.Q_table_joint[agent0_new, agent1_new, coin0_new, coin1_new, :])
                if(value == 0):
                    token_value = 0.25
                elif(value == 1):
                    token_value = 0.5
                elif(value == 2):
                    token_value = 1.0
                elif(value == 3):
                    token_value = 2.0
                elif(value == 4):
                    token_value = 4.0

                
            self.last_token_value = token_value
        if self.token_mode == UCB:
            self.step += 1
            max_upper_bound = 0
            if(str(self.last_token_value) not in self.tokens_dict):
                self.tokens_dict[str(self.last_token_value)] = {'sum_rewards': 0, 'count': 0} 
            self.tokens_dict[str(self.last_token_value)]['sum_rewards'] += sum(rewards)
            self.tokens_dict[str(self.last_token_value)]['count'] += 1    
            token_value = self.last_token_value
            if True:
                for token, stats in self.tokens_dict.items():
                    if(stats['count'] != 0):
                        mean_reward = stats['sum_rewards'] / stats['count']
                        di = numpy.sqrt((3/2 * numpy.log(self.step + 1)) / stats['count'])
                        upper_bound = mean_reward + di
                    else:
                        upper_bound = 1e400
                    if(upper_bound > max_upper_bound):
                        max_upper_bound = upper_bound
                        self.best_value = float(token)
                p = random.uniform(0, 1)  
                if self.step < 1000:
                    token_value = random.choice([0.25, 0.5, 1.0, 2.0, 4.0])
                else:                 
                    token_value = self.best_value       
                self.last_token_value = token_value
                transition["token_value"] = token_value
        if self.token_mode == EPSILON_GREEDY:
            self.step += 1
            if(str(self.last_token_value) not in self.tokens_dict):
                self.tokens_dict[str(self.last_token_value)] = {'sum_rewards': 0, 'count': 0} 
            self.tokens_dict[str(self.last_token_value)]['sum_rewards'] += sum(rewards)
            self.tokens_dict[str(self.last_token_value)]['count'] += 1    
            token_value = self.last_token_value
            if(self.epsilon > 0.2 and self.step % 5000 == 0):
                self.epsilon -= 0.1       
            if done:
                max_mean = float("-inf")
                for token, stats in self.tokens_dict.items():
                    if(stats['count'] != 0):
                        mean_reward = stats['sum_rewards'] / stats['count']                       
                        if mean_reward > max_mean:  
                            max_mean = mean_reward
                            self.best_value = float(token)
                p = random.uniform(0, 1)  
                if p < self.epsilon:
                    token_value = random.choice([0.25, 0.5, 1.0, 2.0, 4.0])
                else:                 
                    token_value = self.best_value 
                self.last_token_value = token_value
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
                        self.trust_response_matrix[j][i] = accept_trust * token_value#[j] 
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
