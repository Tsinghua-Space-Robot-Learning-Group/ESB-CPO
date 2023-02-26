import torch
import torch.nn as nn
import numpy as np
from gym.spaces import Box, Discrete
from esbcpo.common.online_mean_std import OnlineMeanStd
from esbcpo.models.MLP_Categorical_Actor import MLPCategoricalActor
from esbcpo.models.MLP_Gaussian_Actor import MLPGaussianActor
from esbcpo.models.Critic import Critic
from esbcpo.models.model_utils import build_mlp_network
class ActorCritic(nn.Module):
    def __init__(self,
                 actor_type,
                 observation_space,
                 action_space,
                 use_standardized_obs,
                 use_scaled_rewards,
                 use_shared_weights,
                 ac_kwargs,
                 weight_initialization='kaiming_uniform'
                 ):
        super().__init__()
        self.obs_shape = observation_space.shape
        self.action_space = action_space.shape
        self.obs_oms = OnlineMeanStd(shape=self.obs_shape) \
            if use_standardized_obs else None

        ## fix
        # self.obs_shape = observation_space.shape
        # self.obs_oms = OnlineMeanStd(shape=self.obs_shape) \
        #     if use_standardized_obs else None

        self.ac_kwargs = ac_kwargs

        # policy builder depends on action space
        if isinstance(action_space, Box):
            
            actor_fn = MLPGaussianActor
            act_dim = action_space.shape[0]
        elif isinstance(action_space, Discrete):
            # distribution_type = 'categorical'
            actor_fn = MLPCategoricalActor
            act_dim = action_space.n
        else:
            raise ValueError

        obs_dim = observation_space.shape[0]
        layer_units = [obs_dim] + list(ac_kwargs['pi']['hidden_sizes'])
        act = ac_kwargs['pi']['activation']
        if use_shared_weights:
            shared = build_mlp_network(
                layer_units,
                activation=act,
                weight_initialization=weight_initialization,
                output_activation=act
            )
        else:
            shared = None

        # actor_fn = get_registered_actor_fn(actor_type, distribution_type)
        self.pi = actor_fn(obs_dim=obs_dim,
                           act_dim=act_dim,
                           shared=shared,
                           weight_initialization=weight_initialization,
                           **ac_kwargs['pi'])
        self.v = Critic(obs_dim,
                           shared=shared,
                           **ac_kwargs['val'])

        self.ret_oms = OnlineMeanStd(shape=(1,)) if use_scaled_rewards else None

    def forward(self,
                obs: torch.Tensor
                ) -> tuple:
        return self.step(obs)

    def step(self,
             obs: torch.Tensor
             ):
        """ 
            If training, this includes exploration noise!
            Expects that obs is not pre-processed.

            Returns:
                action, value, log_prob(action)
            Note:
                Training mode can be activated with ac.train()
                Evaluation mode is activated by ac.eval()
        """
        with torch.no_grad():
            if self.obs_oms:
                # Note: Update RMS in Algorithm.running_statistics() method
                # self.obs_oms.update(obs) if self.training else None
                obs = self.obs_oms(obs)
            v = self.v(obs)
            if self.training:
                a, logp_a = self.pi.sample(obs)
            else:
                a, logp_a = self.pi.predict(obs)

        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self,
            obs: torch.Tensor
            ) -> np.ndarray:
        return self.step(obs)[0]

    def update(self, frac):
        """update internals of actors

            1) Updates exploration parameters
            + for Gaussian actors update log_std

        frac: progress of epochs, i.e. current epoch / total epochs
                e.g. 10 / 100 = 0.1

        """
        if hasattr(self.pi, 'set_log_std'):
            self.pi.set_log_std(1 - frac)
