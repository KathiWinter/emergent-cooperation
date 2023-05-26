from mate.utils import assertEquals, get_param_or_default
from mate.controllers.actor_critic import ActorCritic, preprocessing_module
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy

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

class IncentiveNet(nn.Module):
    def __init__(self, agent_id, input_dim, nr_agents, nr_actions, nr_hidden_units, learning_rate):
        super(IncentiveNet, self).__init__()
        self.agent_id = agent_id
        self.other_agents_id = [i for i in range(nr_agents) if i != self.agent_id]
        self.nr_actions = nr_actions
        self.nr_input_features = input_dim + (nr_agents-1)*nr_actions
        self.nr_hidden_units = nr_hidden_units
        self.nr_outputs = nr_agents
        self.fc_net = preprocessing_module(self.nr_input_features, self.nr_hidden_units)
        self.reward_head = nn.Linear(self.nr_hidden_units, nr_agents-1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, observations, joint_actions):
        batch_size = observations.size(0)
        x = observations.view(batch_size, -1)
        y = []
        assertEquals(batch_size, len(joint_actions))
        for joint_action in joint_actions:
            other_joint_action = [joint_action[i] for i in self.other_agents_id]
            one_hot_actions = torch.zeros(len(self.other_agents_id)*self.nr_actions, dtype=torch.float32)
            for i, action in enumerate(other_joint_action):
                one_hot_actions[int(i*self.nr_actions + action)] = 1
            y.append(one_hot_actions)
        y = torch.stack(y)
        x = self.fc_net(torch.cat([x, y], dim=-1))
        output = torch.zeros((batch_size, self.nr_outputs), dtype=torch.float32)
        output[:,self.other_agents_id] = self.reward_head(x)
        return torch.sigmoid(output)

"""
 Learning to Incentivice Other agents (LIO)
"""
class LIO(ActorCritic):

    def __init__(self, params):
        super(LIO, self).__init__(params)
        self.cost_weight = get_param_or_default(params, "cost_weight", 0.001)
        self.incentive_nets = []
        self.R_max = get_param_or_default(params, "R_max", 3)
        for i in range(self.nr_agents):
            self.incentive_nets.append(IncentiveNet(i, self.input_dim, self.nr_agents,\
                self.nr_actions, params["nr_hidden_units"], self.learning_rate))
        self.update_policy = True
        #mate
        self.last_rewards_observed = [[] for _ in range(self.nr_agents)]
        self.mate_mode = get_param_or_default(params, "mate_mode", STATIC_MODE)
        self.defect_mode = get_param_or_default(params, "defect_mode", NO_DEFECT)
        self.trust_requests = []
        self.trust_responses = []
 

    def update_step(self):
        preprocessed_data = self.preprocess()
        if not self.update_policy:
            for i, memory, incentive_net in\
                zip(range(self.nr_agents), self.memories, self.incentive_nets):
                self.local_incentive_update(i, memory, incentive_net, preprocessed_data)
        for i, memory, actor_net, critic_net in\
            zip(range(self.nr_agents), self.memories, self.actor_nets, self.critic_nets):
            self.local_update(i, memory, actor_net, critic_net, preprocessed_data)
        for memory in self.memories:
            memory.clear()
        self.update_policy = not self.update_policy

    def preprocess(self):
        incentives = []
        abs_incentive_returns = [torch.zeros(1, dtype=torch.float32, device=self.device) for _ in range(self.nr_agents)]
        current_incentive_returns = torch.zeros(self.nr_agents, dtype=torch.float32, device=self.device)
        incentive_returns = [[] for _ in range(self.nr_agents)]
        receive_enabled = None
        for memory, incentive_net in zip(self.memories, self.incentive_nets):
            current_abs_incentive_return = torch.zeros(1, dtype=torch.float32, device=self.device)
            histories, _, _, _, _, _, dones, _ = memory.get_training_data()
  
            if receive_enabled is None:
                receive_enabled = [[self.sample_no_comm_failure() for _ in range(self.nr_agents)] for _ in histories]
            joint_actions = memory.get_joint_actions()
            incentives_ = incentive_net(histories, joint_actions)
            incentives.append(incentives_)
            reward_incentives = incentives_*self.R_max # h_length, n_agents
            for t, done, incentive in zip(range(len(dones)), dones, reward_incentives):
                send_enabled = self.sample_no_comm_failure()
                if done:
                    abs_incentive_returns[memory.agent_id] += current_abs_incentive_return
                    current_incentive_returns.fill_(0.0)
                rew_incentive = torch.zeros(self.nr_agents, dtype=torch.float32, device=self.device)
                rew_incentive_other = torch.zeros(self.nr_agents, dtype=torch.float32, device=self.device)
                
                #current_incentive_returns_other = self.gamma*current_incentive_returns_other
                rew_incentive_other[memory.agent_id] += max(self.trust_requests[t][memory.agent_id])
                filtered_trust_responses = [self.trust_responses[t][memory.agent_id][x] for x in range(self.nr_agents) if abs(self.trust_responses[t][memory.agent_id][x]) > 0]
                if len(filtered_trust_responses) > 0:
                                rew_incentive_other[memory.agent_id] += min(filtered_trust_responses)
                if len(incentive_returns[memory.agent_id]) <= t:
                        incentive_returns[memory.agent_id].append(0)
                incentive_returns[memory.agent_id][t] = sum(rew_incentive_other)
                                
                current_incentive_returns = self.gamma*current_incentive_returns
                if send_enabled:
                    for a in range(self.nr_agents):
                        if receive_enabled[t][a]:
                            rew_incentive[a] = incentive[a] * self.trust_requests[t][a][memory.agent_id] + incentive[a] * self.trust_responses[t][a][memory.agent_id]

                    current_incentive_returns += rew_incentive
                current_abs_incentive_return = incentive.abs().sum() + self.gamma*current_abs_incentive_return

        return incentives, abs_incentive_returns, [torch.tensor(R, dtype=torch.float32, device=self.device) for R in incentive_returns]

    def update_critic(self, agent_id, training_data, critic_net, preprocessed_data):
        _, _, incentive_returns = preprocessed_data
        incentive_returns = incentive_returns[agent_id]
        histories, _, _, _, returns, _, _, _ = training_data
        assertEquals(returns.size(), incentive_returns.size())
        values = critic_net(histories).squeeze()
        assertEquals(values.size(), returns.size())
        returns += incentive_returns
        critic_loss = F.mse_loss(returns.detach(), values)
        critic_net.optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(critic_net.parameters(), self.clip_norm)
        critic_net.optimizer.step()

    def update_actor(self, agent_id, training_data, actor_net, preprocessed_data):
        _, _, incentive_returns = preprocessed_data
        incentive_returns = incentive_returns[agent_id]
        histories, _, actions, _, returns, old_probs, _, _ = training_data
        assertEquals(returns.size(), incentive_returns.size())
        returns += incentive_returns
        values = self.get_values(agent_id, histories).squeeze().detach()
        action_probs = actor_net(histories)
        advantages = returns.detach() - values.detach()
        policy_losses = []
        for action, old_prob, probs, advantage in zip(actions, old_probs, action_probs, advantages):
            policy_losses.append(self.policy_loss(advantage.item(), probs, action, old_prob))
        actor_loss = torch.stack(policy_losses).sum()
        actor_net.optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor_net.parameters(), self.clip_norm)
        actor_net.optimizer.step()

    def local_incentive_update(self, agent_id, memory, incentive_net, preprocessed_data):
        R_incentives, abs_incentive_cost, _ = preprocessed_data
        R_incentives = R_incentives[agent_id]
        extrinsic_returns = memory.get_extrinsic_returns()
        partial_losses = []
        for j in incentive_net.other_agents_id:
            assert j != agent_id
            losses = R_incentives[:,j].view(-1)*extrinsic_returns.detach()
            partial_losses.append(losses.sum())
        loss = torch.stack(partial_losses).sum() + self.cost_weight*self.R_max*abs_incentive_cost[agent_id]
        incentive_net.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(incentive_net.parameters(), self.clip_norm)
        incentive_net.optimizer.step()


    def can_rely_on(self, agent_id, reward, history, next_history):
        history = torch.tensor(numpy.array([history]), dtype=torch.float32, device=self.device)
        next_history = torch.tensor(numpy.array([next_history]), dtype=torch.float32, device=self.device)
        value = self.get_values(agent_id, history)[0].item()
        next_value = self.get_values(agent_id, next_history)[0].item()
        return reward + self.gamma*next_value - value >= 0



    def prepare_transition(self, joint_histories, joint_action, rewards, next_joint_histories, done, info):
        transition = super(LIO, self).prepare_transition(joint_histories, joint_action, rewards, next_joint_histories, done, info)
        token_value = [1, 1]#self.token_value

        original_rewards = [r for r in rewards]
        trust_request_matrix = numpy.zeros((self.nr_agents, self.nr_agents), dtype=numpy.float32)
        trust_response_matrix = numpy.zeros((self.nr_agents, self.nr_agents), dtype=numpy.float32)
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
                    trust_request_matrix[j][i] += token_value[i]
                    
        # 2. Send trust responses
        for i, history, next_history in zip(range(self.nr_agents), joint_histories, next_joint_histories):
            neighborhood = info["neighbor_agents"][i]
            respond_enabled = i != defector_id or self.defect_mode not in [DEFECT_ALL, DEFECT_RESPONSE]
            respond_enabled = respond_enabled and self.sample_no_comm_failure()
            if request_receive_enabled[i]:
                trust_requests = [trust_request_matrix[i][x] for x in neighborhood]
                if len(trust_requests) > 0:
                    #transition["rewards"][i] += numpy.max(trust_requests)
                    pass
            if respond_enabled and len(neighborhood) > 0:
                if self.can_rely_on(i, original_rewards[i], history, next_history):
                    accept_trust = 1
                else:
                    accept_trust = -1
                for j in neighborhood:
                    assert i != j
                    if trust_request_matrix[i][j] > 0:
                        trust_response_matrix[j][i] = accept_trust * token_value[i]
  
        # 3. Receive trust responses
        for i, trust_responses in enumerate(trust_response_matrix):
            neighborhood = info["neighbor_agents"][i]
            receive_enabled = i != defector_id or self.defect_mode not in [DEFECT_ALL, DEFECT_RECEIVE]
            receive_enabled = receive_enabled and self.sample_no_comm_failure()
            if receive_enabled and len(neighborhood) > 0 and trust_responses.any():
                filtered_trust_responses = [trust_responses[x] for x in neighborhood if abs(trust_responses[x]) > 0]
                if len(filtered_trust_responses) > 0:
                    #transition["rewards"][i] += min(filtered_trust_responses)
                    pass
        if done:
            self.last_rewards_observed = [[] for _ in range(self.nr_agents)]
        
        self.trust_requests.append(trust_request_matrix)
        self.trust_responses.append(trust_response_matrix)
        return transition