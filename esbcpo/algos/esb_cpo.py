import torch
import numpy as np
import json
import os
import torch.optim as optim
from esbcpo.algos.trpo import TRPO
import esbcpo.common.mpi_tools as mpi_tools
# from esbcpo.algos.lagrangian_base import Lagrangian
from esbcpo.common.utils import get_flat_params_from, set_param_values_to_model,\
                                set_param_values_to_model,get_flat_gradients_from,\
                                conjugate_gradients
class ESB_CPO(TRPO):
    """
        Paper Name: Lyapunov-based Constrained Policy Optimization
        Paper author: ElegentControl
        Paper URL: 

        This implementation 
    """
    def __init__(
            self,
            algo: str = 'esb_cpo',
            cost_limit: float = 25.,
            use_safety_state: bool = True,
            constrain_coef: float = 0.01,
            lagrangian_multiplier_init: float = 0.001,
            lambda_lr: float = 0.035,
            lambda_optimizer: str = 'Adam',
            use_lagrangian_penalty: bool = True,
            **kwargs
    ):
        saute = kwargs.pop('saute')
        if not saute:
            env_lib = kwargs.pop("env_lib")
            env_lib = "pure"
        if use_safety_state:
            saute = True

        TRPO.__init__(
            self,
            algo=algo,
            use_standardized_reward=True, 
            use_standardized_cost=True,
            use_standardized_obs=True,
            use_kl_early_stopping=False,
            use_cost_value_function=True,
            saute = saute,
            env_lib = env_lib,
            cost_limit = cost_limit,
            **kwargs
        )

        # Lagrangian.__init__(
        #     self,
        #     cost_limit=cost_limit,
        #     use_lagrangian_penalty=use_lagrangian_penalty,
        #     lagrangian_multiplier_init=lagrangian_multiplier_init,
        #     lambda_lr=lambda_lr,
        #     lambda_optimizer=lambda_optimizer
        # )

        self.cost_limit = cost_limit
        self.constrain_coeff = torch.tensor(1 - 1e-10)
        self.constrain_coeff_limit = torch.tensor(constrain_coef)
        # with open(os.path.join(self.logger.log_dir, "config.json"), 'r') as f:
        #     data = json.load(f)
        # data.update({"use_safety_state": use_safety_state,
        #              "cost_limit": self.cost_limit,
        #              "constrain_coeff": self.constrain_coeff})
        # output = json.dumps(data, separators=(',', ':\t'), indent=4, sort_keys=True)
        # with open(os.path.join(self.logger.log_dir, "config.json"), 'w') as f:
        #     f.write(output)
        self.loss_pi_cost_before = 0.
        
        # Log Lagrangian init
        self.SCALE_lambda_MIN_MAX = (0, 1)
        self.lambda_lr = lambda_lr
        # self.use_lagrangian_penalty = use_lagrangian_penalty

        init_value = max(lagrangian_multiplier_init, 1e-5)
        self.log_lagrangian_multiplier = torch.nn.Parameter(
            torch.as_tensor(np.log(init_value)),   
            requires_grad=True)
        self.lambda_range_projection = torch.nn.ReLU()
        # fetch optimizer from PyTorch optimizer package
        assert hasattr(optim, lambda_optimizer), \
            f'Optimizer={lambda_optimizer} not found in torch.'
        torch_opt = getattr(optim, lambda_optimizer)
        self.lambda_optimizer = torch_opt([self.log_lagrangian_multiplier, ],
                                          lr=lambda_lr)

    def search_step_size(
            self,
            step_dir,
            g_flat,
            c,
            optim_case,
            p_dist,
            data,
            total_steps: int = 25,
            decay: float = 0.8
    ):
        """
            SPPO algorithm performs line-search to ensure constraint satisfaction for rewards and costs.
        """
        step_frac = 1.0
        _theta_old = get_flat_params_from(self.ac.pi.net)
        _, old_log_p = self.ac.pi(data['obs'], data['act'])
        expected_rew_improve = g_flat.dot(step_dir)

        # while not within_trust_region:
        for j in range(total_steps):
            new_theta = _theta_old + step_frac * step_dir
            set_param_values_to_model(self.ac.pi.net, new_theta)
            acceptance_step = j + 1

            with torch.no_grad():
                loss_pi_rew, _ = self.compute_loss_pi(data=data)
                loss_pi_cost, _, _, _ = self.compute_loss_cost_performance(data=data)
                # determine KL div between new and old policy
                q_dist = self.ac.pi.dist(data['obs'])
                torch_kl = torch.distributions.kl.kl_divergence(
                    p_dist, q_dist).mean().item()
            loss_rew_improve = self.loss_pi_before - loss_pi_rew.item()
            cost_diff = loss_pi_cost.item() - self.loss_pi_cost_before

            # Average across MPI processes...
            torch_kl = mpi_tools.mpi_avg(torch_kl)
            loss_rew_improve = mpi_tools.mpi_avg(loss_rew_improve)
            cost_diff = mpi_tools.mpi_avg(cost_diff)

            self.logger.log("Expected Improvement: %.3f Actual: %.3f" % (
                expected_rew_improve, loss_rew_improve))

            if not torch.isfinite(loss_pi_rew) and not torch.isfinite(
                    loss_pi_cost):
                self.logger.log('WARNING: loss_pi not finite')
            elif loss_rew_improve < 0 if optim_case > 1 else False:
                self.logger.log('INFO: did not improve improve <0')

            elif cost_diff > max(-c, 0):
                self.logger.log(f'INFO: no improve {cost_diff} > {max(-c, 0)}')
            elif torch_kl > self.target_kl * 1.5:
                self.logger.log(
                    f'INFO: violated KL constraint {torch_kl} at step {j + 1}.')
            else:
                # step only if surrogate is improved and we are
                # within the trust region
                self.logger.log(f'Accept step at i={j + 1}')
                break
            step_frac *= decay
        else:
            self.logger.log('INFO: no suitable step found...')
            step_dir = torch.zeros_like(step_dir)
            acceptance_step = 0

        set_param_values_to_model(self.ac.pi.net, _theta_old)
        return step_frac * step_dir, acceptance_step

    def algorithm_specific_logs(self):
        TRPO.algorithm_specific_logs(self)
        self.logger.log_tabular('Misc/cost_gradient_norm')
        self.logger.log_tabular('Misc/A')
        self.logger.log_tabular('Misc/B')
        self.logger.log_tabular('Misc/q')
        self.logger.log_tabular('Misc/r')
        self.logger.log_tabular('Misc/s')
        self.logger.log_tabular('Misc/Lambda_star')
        self.logger.log_tabular('Misc/Nu_star')
        self.logger.log_tabular('Misc/OptimCase')
        self.logger.log_tabular('Misc/Constrain_coeff')
        self.logger.log_tabular('Misc/Gap1')
        self.logger.log_tabular('Misc/Gap2')

    def compute_loss_cost_performance(self, data, grad = True):
        """
        tips: 
        1 - epsilon : represents the degree of cost if ep_costs> cost_limit, 1-epsilon close to 0
        constraints improve
        ( 1 - self.buf.gamma ): a constant can be modified (adjust performance)
        a possible matching with saute rl is to replace (1 - epsilon) with a z_t's formulation 
        """

        if self.saute:
            safety_states = data['safety_state']
            epsilon = np.clip(-np.tanh(safety_states[1:]), 0, 1)
        else:
            ep_costs = self.logger.get_stats('EpCosts')[0]
            epsilon = np.clip (np.tanh( ep_costs - self.cost_limit ), 0, 1) # scale from 0 to 1
        
        if grad:
            dist, _log_p = self.ac.pi(data['obs'], data['act'])
        else:
            dist, _log_p = self.ac.pi(data['obs'], data['act'])
            _log_p = _log_p.detach()
        ratio = torch.exp(_log_p - data['log_p'])
        constraint_func = (data['cost_val'][1:]-data['cost_val'][:-1]) + self.constrain_coeff * ((data['cost_val'][:-1] ) - (1 - epsilon) * data['cost_val'][1:])
        
        alpha = np.clip(self.constrain_coeff.numpy(), 0.01, 1-1e-10)
        # self.constrain_coeff = torch.tensor(alpha)
        # constraint_func /= (1 - alpha)

        B1 = (1 - self.gamma) * data['cost_val'][1:] - data['cost'][:-1]
        B2 = alpha * (1 - (1 - epsilon)) / (1 - alpha) * data['cost_val'][1:]
        # data['cost_adv'][:-1] + B1
        # constraint_func = (data['cost_val'][1:]-data['cost_val'][:-1]) + B2

        mask = [True] * len(data['cost_val'])
        for idx in data['start_ptr_list']:
            mask[idx - 1] = False
        cost_loss = (ratio[:-1] *(constraint_func))[mask[:-1]].mean()
        G1 = ((ratio[:-1] - 1) * B1)[mask[:-1]].mean() / (1 - self.gamma)
        G2 = ((ratio[:-1] - 1) * B2)[mask[:-1]].mean() / (1 - self.gamma)

        # ent = dist.entropy().mean().item()
        info = {}
        return cost_loss, info, G1.detach().numpy(), G2.detach().numpy()

    def update_policy_net(self, data):
        # Get loss and info values before update
        theta_old = get_flat_params_from(self.ac.pi.net)
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data=data)
        self.loss_pi_before = loss_pi.item()
        self.loss_v_before = self.compute_loss_v(data['obs'],
                                                 data['target_v']).item()
        self.loss_c_before = self.compute_loss_c(data['obs'],
                                                 data['target_c']).item()
        # get prob. distribution before updates
        p_dist = self.ac.pi.dist(data['obs'])
        # Train policy with multiple steps of gradient descent
        loss_pi.backward()
        # average grads across MPI processes
        mpi_tools.mpi_avg_grads(self.ac.pi.net)
        g_flat = get_flat_gradients_from(self.ac.pi.net)

        # flip sign since policy_loss = -(ration * adv)
        g_flat *= -1

        x = conjugate_gradients(self.Fvp, g_flat, self.cg_iters)
        assert torch.isfinite(x).all()
        eps = 1.0e-8
        # Note that xHx = g^T x, but calculating xHx is faster than g^T x
        xHx = torch.dot(x, self.Fvp(x))  # equivalent to : g^T x
        alpha = torch.sqrt(2 * self.target_kl / (xHx + eps))
        assert xHx.item() >= 0, 'No negative values'

        # update constrain coefficient
        loss_cost, _, _, _ = self.compute_loss_cost_performance(data=data, grad = False)
        # update Lagrange multiplier parameter
        self.update_lagrange_multiplier(loss_cost)
        self.constrain_coeff = max(torch.tanh(0.05 / self.lambda_l_op().detach()), self.constrain_coeff_limit)
        # self.constrain_coeff = max(torch.tanh(self.lambda_l_op().detach()), self.constrain_coeff_limit)
        # self.constrain_coeff = 1 - self.lambda_l_op().detach()
        self.logger.log(f'constrain_coeff = {self.constrain_coeff}')

        # get the policy cost performance gradient b (flat as vector)
        self.pi_optimizer.zero_grad()
        loss_cost, _, G1, G2 = self.compute_loss_cost_performance(data=data)
        loss_cost.backward()
        # average grads across MPI processes
        mpi_tools.mpi_avg_grads(self.ac.pi.net)
        self.loss_pi_cost_before = loss_cost.item()
        b_flat = get_flat_gradients_from(self.ac.pi.net)

        ep_costs = self.logger.get_stats('EpCosts')[0]
        c = ep_costs - self.cost_limit
        c = mpi_tools.mpi_avg(c)
        c /= (self.logger.get_stats('EpLen')[0] + eps)  # rescale
        self.logger.log(f'c = {c}')
        self.logger.log(f'b^T b = {b_flat.dot(b_flat).item()}')

        # set variable names as used in the paper
        p = conjugate_gradients(self.Fvp, b_flat, self.cg_iters)
        q = xHx
        r = g_flat.dot(p)  # g^T H^{-1} b
        s = b_flat.dot(p)  # b^T H^{-1} b

        if b_flat.dot(b_flat) <= 1e-6 and c < 0:
            # feasible step and cost grad is zero: use plain TRPO update...
            A = torch.zeros(1)
            B = torch.zeros(1)
            optim_case = 4
        else:

            self.logger.log(f'q={q.item()}')
            self.logger.log(f'r={r.item()}')
            self.logger.log(f's={s.item()}')
            self.logger.log(f'r/c={(r / c).item()}')
            assert torch.isfinite(r).all()
            assert torch.isfinite(s).all()

            A = q - r ** 2 / s  # must be always >= 0 (Cauchy-Schwarz inequality)
            B = 2 * self.target_kl - c ** 2 / s  # safety line intersects trust-region if B > 0

            if c < 0 and B < 0:
                # point in trust region is feasible and safety boundary doesn't intersect
                # ==> entire trust region is feasible
                optim_case = 3
            elif c < 0 and B >= 0:
                # x = 0 is feasible and safety boundary intersects
                # ==> most of trust region is feasible
                optim_case = 2
            elif c >= 0 and B >= 0:
                # x = 0 is infeasible and safety boundary intersects
                # ==> part of trust region is feasible, recovery possible
                optim_case = 1
                self.logger.log('Alert! Attempting feasible recovery!',
                                'yellow')
            else:
                # x = 0 infeasible, and safety halfspace is outside trust region
                # ==> whole trust region is infeasible, try to fail gracefully
                optim_case = 0
                self.logger.log('Alert! Attempting infeasible recovery!', 'red')

        if optim_case in [3, 4]:
            alpha = torch.sqrt(2 * self.target_kl / (xHx + 1e-8))
            nu_star = torch.zeros(1)
            lambda_star = 1 / alpha
            step_dir = alpha * x

        elif optim_case in [1, 2]:
            def project_on_set(t: torch.Tensor,
                               low: float,
                               high: float
                               ) -> torch.Tensor:
                return torch.Tensor([max(low, min(t, high))])

            lambda_a = torch.sqrt(A / B)
            lambda_b = torch.sqrt(q / (2 * self.target_kl))
            if c < 0:
                lambda_a_star = project_on_set(lambda_a, 0., r / c)
                lambda_b_star = project_on_set(lambda_b, r / c, np.inf)
            else:
                lambda_a_star = project_on_set(lambda_a, r / c, np.inf)
                lambda_b_star = project_on_set(lambda_b, 0., r / c)

            def f_a(lam):
                return -0.5 * (A / (lam + eps) + B * lam) - r * c / (s + eps)

            def f_b(lam):
                return -0.5 * (q / (lam + eps) + 2 * self.target_kl * lam)

            lambda_star = lambda_a_star \
                if f_a(lambda_a_star) >= f_b(lambda_b_star) else lambda_b_star

            # Discard all negative values with torch.clamp(x, min=0)
            nu_star = torch.clamp(lambda_star * c - r, min=0) / (s + eps)
            step_dir = 1. / (lambda_star + eps) * (x - nu_star * p)

        else:  # case == 0
            # purely decrease costs
            lambda_star = torch.zeros(1)
            nu_star = np.sqrt(2 * self.target_kl / (s + eps))
            step_dir = -nu_star * p

        final_step_dir, accept_step = self.search_step_size(
            step_dir,
            g_flat,
            c=c,
            optim_case=optim_case,
            p_dist=p_dist,
            data=data,
            total_steps=20
        )
        # update actor network parameters
        new_theta = theta_old + final_step_dir
        set_param_values_to_model(self.ac.pi.net, new_theta)

        q_dist = self.ac.pi.dist(data['obs'])
        torch_kl = torch.distributions.kl.kl_divergence(
            p_dist, q_dist).mean().item()

        self.logger.store(**{
            'Values/Adv': data['act'].numpy(),
            'Entropy': pi_info['ent'],
            'KL': torch_kl,
            'PolicyRatio': pi_info['ratio'],
            'Loss/Pi': self.loss_pi_before,
            'Loss/DeltaPi': loss_pi.item() - self.loss_pi_before,
            'Misc/StopIter': 1,
            'Misc/AcceptanceStep': accept_step,
            'Misc/Alpha': alpha.item(),
            'Misc/FinalStepNorm': final_step_dir.norm().numpy(),
            'Misc/xHx': xHx.numpy(),
            'Misc/H_inv_g': x.norm().item(),  # H^-1 g
            'Misc/gradient_norm': torch.norm(g_flat).numpy(),
            'Misc/cost_gradient_norm': torch.norm(b_flat).numpy(),
            'Misc/Lambda_star': lambda_star.item(),
            'Misc/Nu_star': nu_star.item(),
            'Misc/OptimCase': int(optim_case),
            'Misc/A': A.item(),
            'Misc/B': B.item(),
            'Misc/q': q.item(),
            'Misc/r': r.item(),
            'Misc/s': s.item(),
            'Misc/Constrain_coeff': self.constrain_coeff.numpy(),
            'Misc/Gap1': G1,
            'Misc/Gap2': G2,
        })
    
    def compute_lambda_loss(self, cost_constrain):
        """Penalty loss for Lagrange multiplier."""
        return -self.log_lagrangian_multiplier * cost_constrain

    def update_lagrange_multiplier(self, ep_costs):
        """ Update Lagrange multiplier (lambda)
            Note: ep_costs obtained from: self.logger.get_stats('EpCosts')[0]
            are already averaged across MPI processes.
        """
        self.lambda_optimizer.zero_grad()
        lambda_loss = self.compute_lambda_loss(ep_costs)
        lambda_loss.backward()
        self.lambda_optimizer.step()
        # self.log_lagrangian_multiplier.data.clamp_(0)  # enforce: lambda in [0, inf]
    
    def lambda_l_op(self):
        '''
            return : lambda_l  using learned lambda parameter
        '''
        return torch.clamp(torch.exp(self.log_lagrangian_multiplier), *self.SCALE_lambda_MIN_MAX)
    
