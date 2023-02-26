import torch
from esbcpo.algos.natural_pg import NPG
import esbcpo.common.mpi_tools as mpi_tools
from esbcpo.common.utils import get_flat_params_from, set_param_values_to_model

class TRPO(NPG):
    def __init__(self, algo='trpo', **kwargs):
        NPG.__init__(self, algo=algo, **kwargs)

    def search_step_size(self, step_dir, g_flat, p_dist, data, total_steps=15, decay=0.8):
        """ 
            TRPO performs line-search until constraint satisfaction.
            main idea: search around for a satisfied step of policy update to improve loss and reward performance
            :param step_dir:direction theta changes towards
            :param g_flat:  gradient tensor of reward ,informs about how rewards improve with change of step direction 
            :param c:how much epcost goes above limit
            :param p_dist: inform about old policy, how the old policy p performs on observation this moment
            :param optim_case: the way to optimize
            :param data: data buffer,mainly with adv, costs, values, actions, and observations
            :param decay: how search-step reduces in line-search 
        """
        # How far to go in a single update
        step_frac = 1.0
        # Get old parameterized policy expression
        _theta_old = get_flat_params_from(self.ac.pi.net)
        # Change expected objective function gradient = expected_imrpove best this moment
        expected_improve = g_flat.dot(step_dir)

        # While not within_trust_region and not out of total_steps:
        for j in range(total_steps):
            # Update theta params
            new_theta = _theta_old + step_frac * step_dir
            # Set new params as params of net
            set_param_values_to_model(self.ac.pi.net, new_theta)
            # The stepNo this update accept 
            acceptance_step = j + 1

            with torch.no_grad():
                loss_pi, pi_info = self.compute_loss_pi(data=data)
                # Compute KL distance between new and old policy
                q_dist = self.ac.pi.dist(data['obs'])
                # KL-distance of old p-dist and new q-dist, applied in KLEarlyStopping
                torch_kl = torch.distributions.kl.kl_divergence(
                    p_dist, q_dist).mean().item()
            # Real loss improve: old policy loss - new policy loss
            loss_improve = self.loss_pi_before - loss_pi.item()
            # Average processes.... multi-processing style like: mpi_tools.mpi_avg(xxx)
            torch_kl = mpi_tools.mpi_avg(torch_kl)
            loss_improve = mpi_tools.mpi_avg(loss_improve)

            self.logger.log("Expected Improvement: %.3f Actual: %.3f" % (
                expected_improve, loss_improve))
            if not torch.isfinite(loss_pi):
                self.logger.log('WARNING: loss_pi not finite')
            elif loss_improve < 0:
                self.logger.log('INFO: did not improve improve <0')
            elif torch_kl > self.target_kl * 1.5:
                self.logger.log('INFO: violated KL constraint.')
            else:
                # step only if surrogate is improved and when within trust reg.
                self.logger.log(f'Accept step at i={acceptance_step}')
                break
            step_frac *= decay
        else:
            self.logger.log('INFO: no suitable step found...')
            step_dir = torch.zeros_like(step_dir)
            acceptance_step = 0

        set_param_values_to_model(self.ac.pi.net, _theta_old)

        return step_frac * step_dir, acceptance_step