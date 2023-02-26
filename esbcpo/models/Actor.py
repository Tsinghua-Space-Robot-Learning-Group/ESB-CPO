import torch.nn as nn
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, weight_initialization, shared=None):
        super(Actor, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.shared = shared
        self.weight_initialization = weight_initialization

    def dist(self, obs):
        '''
            Returns:
                torch.distributions.Distribution
        '''
        raise NotImplementedError

    def log_prob_from_dist(self, pi, act):
        '''
            Returns:
                torch.Tensor
        '''
        raise NotImplementedError

    def forward(self, obs, act=None) -> tuple:
        '''
            Returns:
                the distributions for given obs and the log likelihood of given actions under the distributions.
        '''
        pi = self.dist(obs)
        logp_a = None
        if act is not None:
            logp_a = self.log_prob_from_dist(pi, act)
        return pi, logp_a

    def sample(self, obs) -> tuple:
        raise NotImplementedError

    def predict(self, obs) -> tuple:
        """ 
            Returns:
                Predict action based on observation without exploration noise.
                Use this method for evaluation purposes. 
        """
        return self.sample(obs)