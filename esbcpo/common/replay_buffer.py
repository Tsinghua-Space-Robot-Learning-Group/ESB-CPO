import torch
import numpy as np
from esbcpo.common.core import combined_shape,discount_cumsum
from esbcpo.common.vtrace import calculate_v_trace
import esbcpo.common.mpi_tools as mpi_tools


class ReplayBuffer:
    def __init__(self,
                 obs_dim: tuple,
                 act_dim: tuple,
                 size: int,
                 ):         
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.cost_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.safety_state_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
       
    def store(self, obs, act, rew, next_obs, done, cost=0., safety_state=0.):
        
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.cost_buf[self.ptr] = cost
        self.safety_state_buf[self.ptr] = safety_state
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)
    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs],
                     cost= self.cost_buf[idxs],
                     safety_state=self.safety_state_buf)

        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}