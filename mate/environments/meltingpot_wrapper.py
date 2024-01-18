import numpy
import meltingpot
from meltingpot.meltingpot import substrate as meltingpot_substrate
from meltingpot.examples.gym import utils
import functools

PLAYER_STR_FORMAT = 'player_{index}'

class MeltingPot_Environment:

    def __init__(self, params) -> None:
        substrate_name = "territory__rooms"
        self.config = meltingpot_substrate.get_config(substrate_name)
        self.env = meltingpot_substrate.build(substrate_name, roles=self.config.default_player_roles)
        self.actions = len(self.config.action_set)

        self.time_step = 0
        self.gamma = params["gamma"]
        self.time_limit = params["time_limit"]
        self.nr_agents = params["nr_agents"]
        self.sent_gifts = numpy.zeros(self.nr_agents)
        self.discounted_returns = numpy.zeros(self.nr_agents)
        self.undiscounted_returns = numpy.zeros(self.nr_agents)
        self.domain_counts = numpy.zeros(4) #TODO
        self.last_joint_action = -numpy.ones(self.nr_agents, dtype=int)
        self.observation_dim = params["observation_dim"]
        self.num_players = len(self.env.observation_spec())
        self.possible_agents = [
            PLAYER_STR_FORMAT.format(index=index) for index in range(self.num_players)
        ]
        observation_space = utils.remove_world_observations_from_space(utils.spec_to_space(self.env.observation_spec()[0]))
        self.observation_space = functools.lru_cache(maxsize=None)(lambda agent_id: observation_space)
        action_space = utils.spec_to_space(self.env.action_spec()[0])
        self.action_space = functools.lru_cache(maxsize=None)(lambda agent_id: action_space)
        self.state_space = utils.spec_to_space(
        self.env.observation_spec()[0]['WORLD.RGB'])
    
    def domain_values(self):
        return self.domain_counts
    
    def reset(self):
        self.time_step = 0
        self.discounted_returns[:] = 0
        self.undiscounted_returns[:] = 0
        timestep = self.env.reset()
        self.agents = self.possible_agents[:]

        joint_observation = [numpy.array(utils.timestep_to_observations(timestep)[i]['RGB'][:][:]).reshape(self.observation_dim) for i in self.agents]
        #joint_observation = utils.timestep_to_observations(timestep)
        #print("observations:",(utils.timestep_to_observations(timestep)))
    
        return joint_observation
     

    def step(self, action):
        self.time_step += 1
        actions = [action[index] for index, agent in enumerate(self.agents)]
        #print("actions:", actions)
        timestep = self.env.step(actions)
        #print("timestep:", timestep.observation[0]['COLLECTIVE_REWARD'])
        #print("Timestep reward: ", timestep.reward)
        reward_arrays = [timestep.reward[index] for index, agent in enumerate(self.agents)]
        rewards = [float(arr) for arr in reward_arrays]

        #print("rewards: ", rewards)
        info = {"neighbor_agents": [[1,2,3,4,5,6,7,8], [0,2,3,4,5,6,7,8], [0,1,3,4,5,6,7,8], [0,1,2,4,5,6,7,8], [0,1,2,3,5,6,7,8], [0,1,2,3,4,6,7,8], [0,1,2,3,4,5,7,8], [0,1,2,3,4,5,6,8], [0,1,2,3,4,5,6,7]]} #TODO
        
        self.undiscounted_returns += rewards
        #print("time step:", (self.gamma**self.time_step)*(rewards))
        self.discounted_returns += [(self.gamma**self.time_step)*reward for reward in (rewards)]
        observations = [numpy.array(utils.timestep_to_observations(timestep)[i]['RGB'][:][:]).reshape(self.observation_dim) for i in self.agents] 

    
        done = self.is_done()
        if done:
            self.agents = []
        return observations, rewards, done, info

    def is_done(self):
        return self.time_step >= self.time_limit

    def domain_value_debugging_indices(self):
        return 0, 0 #TODO

def make(params):
    domain_name = params["domain_name"]
    if domain_name == "Territory-9":
        substrate_name = "territory__rooms"
        config = meltingpot_substrate.get_config(substrate_name)
        params["nr_actions"] = len(config.action_set)
        params["nr_agents"] = 9
        params["gamma"] = 0.95
        params["history_length"] = 1
        params["view_range"] = [5,9,5,1]
        params["observation_dim"] = int(23232)
        params["time_limit"]=150
        return MeltingPot_Environment(params)