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
        return F.sigmoid(output)

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
        self.preprocessed_data = None
        #mate
        self.last_rewards_observed = [[] for _ in range(self.nr_agents)]
        self.mate_mode = get_param_or_default(params, "mate_mode", TD_ERROR_MODE)
        self.defect_mode = get_param_or_default(params, "defect_mode", NO_DEFECT)
        self.step = 0
        self.incentive_rewards = [[] for _ in range(self.nr_agents)]
        
    def update_step(self):
        self.preprocessed_data = self.preprocess()
        if not self.update_policy:
            for i, memory, incentive_net in\
                zip(range(self.nr_agents), self.memories, self.incentive_nets):
                self.local_incentive_update(i, memory, incentive_net, self.preprocessed_data)
        for i, memory, actor_net, critic_net in\
            zip(range(self.nr_agents), self.memories, self.actor_nets, self.critic_nets):
            self.local_update(i, memory, actor_net, critic_net)
        for memory in self.memories:
            memory.clear()
        self.update_policy = not self.update_policy

    def preprocess(self):
        incentives = []
        abs_incentive_returns = [torch.zeros(1, dtype=torch.float32, device=self.device) for _ in range(self.nr_agents)]
        for memory, incentive_net in zip(self.memories, self.incentive_nets):
            current_abs_incentive_return = torch.zeros(1, dtype=torch.float32, device=self.device)
            histories, _, _, _, _, _, _, dones, _ = memory.get_training_data()
            joint_actions = memory.get_joint_actions()
            incentives_ = incentive_net(histories, joint_actions)
            incentives.append(incentives_)
            reward_incentives = incentives_*self.R_max # h_length, n_agents
            for t, done, incentive in zip(range(len(dones)), dones, reward_incentives):
                if done:
                    abs_incentive_returns[memory.agent_id] += current_abs_incentive_return
                current_abs_incentive_return = incentive.abs().sum() + self.gamma*current_abs_incentive_return
            self.incentive_rewards[memory.agent_id] = reward_incentives
        return incentives, abs_incentive_returns

    def update_critic(self, agent_id, training_data, critic_net):
        histories, _, _, rewards, _, returns, _, _, _ = training_data
        assertEquals(rewards.size(), returns.size())
        values = critic_net(histories).squeeze()
        assertEquals(values.size(), returns.size())
        critic_loss = F.mse_loss(returns.detach(), values)
        critic_net.optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(critic_net.parameters(), self.clip_norm)
        critic_net.optimizer.step()

    def update_actor(self, agent_id, training_data, actor_net):
        histories, _, actions, _, _, returns, old_probs, _, _ = training_data
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
        R_incentives, abs_incentive_cost = preprocessed_data
        R_incentives = R_incentives[agent_id]
        extrinsic_returns = memory.get_extrinsic_returns()
        partial_losses = []
        for j in incentive_net.other_agents_id:
            assert j != agent_id
            losses = torch.log(R_incentives[:,j].view(-1))*extrinsic_returns.detach()
            partial_losses.append(losses.sum())
   
        loss = torch.stack(partial_losses).sum() + self.cost_weight*self.R_max*abs_incentive_cost[agent_id]
        incentive_net.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(incentive_net.parameters(), self.clip_norm)
        incentive_net.optimizer.step()


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
        transition = super(LIO, self).prepare_transition(joint_histories, joint_action, rewards, next_joint_histories, done, info)

        token_value = self.incentive_rewards

        if len(self.incentive_rewards[0]) > 0:
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
                        trust_request_matrix[j][i] += token_value[i][self.step][j]
                        
            # 2. Send trust responses
            for i, history, next_history in zip(range(self.nr_agents), joint_histories, next_joint_histories):
                neighborhood = info["neighbor_agents"][i]
                respond_enabled = i != defector_id or self.defect_mode not in [DEFECT_ALL, DEFECT_RESPONSE]
                respond_enabled = respond_enabled and self.sample_no_comm_failure()
                if request_receive_enabled[i]:
                    trust_requests = [trust_request_matrix[i][x] for x in neighborhood]
                    if len(trust_requests) > 0:
                        transition["incentive_rewards"][i] += numpy.max(trust_requests)
                        
                if respond_enabled and len(neighborhood) > 0:
                    if self.can_rely_on(i, transition["rewards"][i]+transition["incentive_rewards"][i], history, next_history):
                        accept_trust = 1
                    else:
                        accept_trust = -1
                    for j in neighborhood:
                        assert i != j
                        if trust_request_matrix[i][j] > 0:
                            trust_response_matrix[j][i] = accept_trust * token_value[i][self.step][j]
    
            # 3. Receive trust responses
            for i, trust_responses in enumerate(trust_response_matrix):
                neighborhood = info["neighbor_agents"][i]
                receive_enabled = i != defector_id or self.defect_mode not in [DEFECT_ALL, DEFECT_RECEIVE]
                receive_enabled = receive_enabled and self.sample_no_comm_failure()
                if receive_enabled and len(neighborhood) > 0 and trust_responses.any():
                    filtered_trust_responses = [trust_responses[x] for x in neighborhood if abs(trust_responses[x]) > 0]
                    if len(filtered_trust_responses) > 0:
                        transition["incentive_rewards"][i] += min(filtered_trust_responses)
        
            if done:
                print(token_value[0][self.step][1])
                print(token_value[1][self.step][0])
                
        self.step += 1              
        if done:
            self.last_rewards_observed = [[] for _ in range(self.nr_agents)]
            self.step = 0
        return transition