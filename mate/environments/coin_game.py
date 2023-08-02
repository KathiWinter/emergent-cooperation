from mate.environments.environment import Environment
import numpy
import random
import math

MOVE_NORTH = 0
MOVE_SOUTH = 1
MOVE_WEST = 2
MOVE_EAST = 3

COIN_GAME_ACTIONS = [MOVE_NORTH, MOVE_SOUTH, MOVE_WEST, MOVE_EAST]

NORMAL = 1
PENALTY = 0
LEVEL_UP = 0
VERSION = [NORMAL, PENALTY, LEVEL_UP]

class MovableAgent:

    def __init__(self, agent_id, width, height, view_range):
        self.agent_id = agent_id
        self.position = None
        self.width = width
        self.height = height
        self.view_range = view_range
    def move(self, action):
        x, y = self.position
        if action == MOVE_NORTH and y + 1 < self.height:
            self.position = (x, y + 1)
        if action == MOVE_SOUTH and y - 1 >= 0:
            self.position = (x, y - 1)
        if action == MOVE_EAST and x + 1 < self.width:
            self.position = (x + 1, y)
        if action == MOVE_WEST and x - 1 >= 0:
            self.position = (x - 1, y)
    def reset(self, position):
        self.position = position
    def visible_positions(self):
        x0, y0 = self.position
        x_center = int(self.view_range/2)
        y_center = int(self.view_range/2)
        positions = [(x,y) for x in range(-x_center+x0, x_center+1+x0)\
            for y in range(-y_center+y0, y_center+1+y0)]
        return positions
    def relative_position_to(self, other_position):
        dx = other_position[0] - self.position[0]
        dy = other_position[1] - self.position[1]
        return dx, dy
class Coin:
    def __init__(self, nr_agents):
        self.agent_ids = list(range(nr_agents))
        self.agent_id = None # Indicates color of coin
        self.position = None
    def reset(self, position):
        self.position = position
        self.agent_id = random.choice(self.agent_ids)
class CoinGameEnvironment(Environment):
    def __init__(self, params):
        params["domain_value_labels"] = ["time_steps", "coins_collected", "own_coins_collected", "coin_1_generated"]
        super(CoinGameEnvironment, self).__init__(params)
        self.width = params["width"]
        self.height = params["height"]
        self.view_range = params["view_range"]
        self.observation_shape = (4, self.width, self.height)
        self.agents = [MovableAgent(i, self.width, self.height, self.view_range) for i in range(self.nr_agents)]
        self.positions = [(x, y) for x in range(self.width) for y in range(self.height)]
        self.coin = Coin(self.nr_agents)
        self.episode_step = 0
        self.episode = 0
        self.upgrade = 0.1
        self.penalty = [1 for _ in range(self.nr_agents)]
        self.coin_count = 0
        self.level = 1

    def generate_drift(self, step, center, amplitude, frequency):
        return center + amplitude * math.sin(2 * math.pi * frequency * step)

    def perform_step(self, joint_action):
        self.episode_step += 1
        rewards, infos = super(CoinGameEnvironment, self).perform_step(joint_action)
        assert not self.is_done(), "Episode terminated at time step {}. Please, reset before calling 'step'."\
            .format(self.time_step)
        coin_collected = False
        agent_actions = list(zip(self.agents, joint_action))
        random.shuffle(agent_actions)
        for agent, action in agent_actions:
            agent.move(action)
            self.penalty[agent.agent_id] += 1
            if agent.position == self.coin.position:
                self.domain_counts[1] += 1
                coin_collected = True
                self.coin_count += 1
                if self.coin_count == 3:
                    self.coin_count = 0
                    self.level += 2
                rewards[agent.agent_id] += 1
                if agent.agent_id != self.coin.agent_id:
                    if VERSION[0] == 1: #NORMAL
                        rewards[self.coin.agent_id] -= 2
                    elif VERSION[1] == 1: #PENALTY
                        rewards[self.coin.agent_id] -= self.penalty[agent.agent_id]
                    elif VERSION[2] == 1: #LEVEL UP
                        rewards[self.coin.agent_id] -= self.level
                    self.domain_counts[2] += 1    
                else:
                    self.penalty[agent.agent_id] = 1
        if coin_collected:
            old_position = self.coin.position
            new_position = random.choice([pos for pos in self.positions if pos != old_position])
            self.coin.reset(new_position)
        if self.episode_step == 150:
            self.level = 1
            self.episode += 1
            self.episode_step = 0            
        return rewards, infos

    def get_metric_indices(self, metric):
        if metric == "own_coin_prob":
            return self.get_index("own_coins_collected"), self.get_index("coins_collected")
        return None, self.get_index("time_steps")
    
    def domain_value_debugging_indices(self):
        return self.get_index("own_coins_collected"), self.get_index("coins_collected")
    def local_observation(self, agent_id):
        observation = numpy.zeros(self.observation_shape)
        focus_agent = self.agents[agent_id]
        x, y = focus_agent.position
        observation[0][x][y] = 1
        for agent in self.agents:
            if agent.agent_id != focus_agent.agent_id:
                x, y = agent.position
                observation[1][x][y] += 1
        index = 2
        if self.coin.agent_id != agent_id:
            index = 3
        x, y = self.coin.position
        observation[index][x][y] = 1
        return observation.reshape(-1)
    def reset(self):
        positions = random.sample(self.positions, k=(self.nr_agents+1))
        for i, agent in enumerate(self.agents):
            agent.reset(positions[i])
        self.coin.reset(positions[-1])
        return super(CoinGameEnvironment, self).reset()
def make(params):
    domain_name = params["domain_name"]
    params["gamma"] = 0.95
    params["time_limit"] = 150
    params["nr_actions"] = len(COIN_GAME_ACTIONS)
    params["history_length"] = 1
    params["view_range"] = 5
    if domain_name == "CoinGame-2":
        params["nr_agents"] = 2
        params["width"] = 3
        params["height"] = 3
    if domain_name == "CoinGame-4":
        params["nr_agents"] = 4
        params["width"] = 5
        params["height"] = 5
    if domain_name == "CoinGame-6":
        params["nr_agents"] = 6
        params["width"] = 7
        params["height"] = 7
    if domain_name == "CoinGame-8":
        params["nr_agents"] = 8
        params["width"] = 9
        params["height"] = 9   
    params["observation_dim"] = int(params["width"]*params["height"]*4)
    return CoinGameEnvironment(params)