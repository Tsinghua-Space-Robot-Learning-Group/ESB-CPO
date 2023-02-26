import numpy as np
from typing import Tuple, Dict, List 
from bullet_safety_gym.envs.builder import EnvironmentBuilder
from gym.utils import seeding
from esbcpo.env.wrappers.saute_env import saute_env

saute_env_cfg = dict(
        action_dim=2,
        action_range=[-1, 1],
        saute_discount_factor=0.99,
        safety_budget=15,
        unsafe_reward=-1.0,        
        max_ep_len=250, # check?
        min_rel_budget=1.,
        max_rel_budget=1.,
        test_rel_budget=1.,   
        mode="train",
        use_reward_shaping=False,
        use_state_augmentation=False             
)

class BaselineBuilder(EnvironmentBuilder):
    """
    Base class for the bullet safety gym environments 
    """
    def __init__(
        self, 
        max_ep_len:int=200,
        mode:str="train",
        builder_cfg:Dict=None,
    ):
        super(BaselineBuilder, self).__init__(**builder_cfg)
        assert mode == "train" or mode == "test" or mode == "deterministic", "mode can be deterministic, test or train"
        assert max_ep_len > 0
        self.max_episode_steps = max_ep_len
        self._mode = mode

    def seed(self, seed:int=None) -> List[int]:
        super(BaselineBuilder, self).seed(seed)
        self.np_random, seed = seeding.np_random(self._seed)
        return [seed]

    def step(self, action:np.ndarray) -> Tuple[np.ndarray, int, bool, Dict]:
        obs, reward, done, info = super(BaselineBuilder, self).step(action)
        # info['pos_com'] = self.world.robot_com() # saving position of the robot to plot
        return obs, reward, done, info
   
    
@saute_env
class AugmentedSafeBuilder(BaselineBuilder):
    """Sauted pendulum using a wrapper"""


if __name__ == '__main__':
    # import bullet_safety_gym.envs
    import gym
    env = gym.make('SafetyBallReach-v0')
    sauted_env = AugmentedSafeBuilder(safety_budget=saute_env_cfg['safety_budget'], 
                saute_discount_factor=saute_env_cfg['saute_discount_factor'],
                max_ep_len=saute_env_cfg['max_ep_len'],
                mode="train",
                unsafe_reward=saute_env_cfg['unsafe_reward'],
                min_rel_budget=saute_env_cfg['min_rel_budget'],
                max_rel_budget=saute_env_cfg['max_rel_budget'],
                test_rel_budget=saute_env_cfg['test_rel_budget'],
                use_reward_shaping=saute_env_cfg['use_reward_shaping'],
                use_state_augmentation=saute_env_cfg['use_state_augmentation'],
                builder_cfg=env.spec.kwargs
    )
    pass
