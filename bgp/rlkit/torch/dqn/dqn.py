from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import bgp.rlkit.torch.pytorch_util as ptu
from bgp.rlkit.exploration_strategies.base import PolicyWrappedWithExplorationStrategy
from bgp.rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
from bgp.rlkit.core.eval_util import create_stats_ordered_dict
from bgp.rlkit.policies.argmax import ArgmaxDiscretePolicy
from bgp.rlkit.torch.torch_rl_algorithm import TorchRLAlgorithm


class DQN(TorchRLAlgorithm):
    def __init__(
        self,
        env,
        qf,
        policy=None,
        learning_rate=1e-3,
        use_hard_updates=False,
        hard_update_period=1000,
        tau=0.001,
        epsilon=0.1,
        qf_criterion=None,
        weight_decay=0,
        gradient_max_value=None,
        **kwargs
    ):
        """

        :param env: Env.
        :param qf: QFunction. Maps from state to action Q-values.
        :param learning_rate: Learning rate for qf. Adam is used.
        :param use_hard_updates: Use a hard rather than soft update.
        :param hard_update_period: How many gradient steps before copying the
        parameters over. Used if `use_hard_updates` is True.
        :param tau: Soft target tau to update target QF. Used if
        `use_hard_updates` is False.
        :param epsilon: Probability of taking a random action.
        :param kwargs: kwargs to pass onto TorchRLAlgorithm
        """
        exploration_strategy = EpsilonGreedy(
            action_space=env.action_space,
            prob_random_action=epsilon,
        )
        self.policy = policy or ArgmaxDiscretePolicy(qf)
        exploration_policy = PolicyWrappedWithExplorationStrategy(
            exploration_strategy=exploration_strategy,
            policy=self.policy,
        )
        super().__init__(env, exploration_policy, eval_policy=self.policy, **kwargs)
        self.qf = qf
        self.target_qf = self.qf.copy()
        self.learning_rate = learning_rate
        self.use_hard_updates = use_hard_updates
        self.hard_update_period = hard_update_period
        self.tau = tau
        self.qf_optimizer = optim.Adam(
            self.qf.parameters(), lr=self.learning_rate, weight_decay=weight_decay
        )
        if qf_criterion is None:
            self.qf_criterion = nn.MSELoss()
        else:
            try:
                self.qf_criterion = qf_criterion()
            except:
                self.qf_criterion = qf_criterion  # Messy
        self.gradient_max_value = gradient_max_value

    def _do_training(self):
        batch = self.get_batch()
        rewards = batch["rewards"]
        terminals = batch["terminals"]
        obs = batch["observations"]
        actions = batch["actions"]
        next_obs = batch["next_observations"]

        """
        Compute loss
        """

        target_q_values = self.target_qf(next_obs).detach().max(1, keepdim=True)[0]
        y_target = rewards + (1.0 - terminals) * self.discount * target_q_values
        y_target = y_target.detach()
        # actions is a one-hot vector
        y_pred = torch.sum(self.qf(obs) * actions, dim=1, keepdim=True)
        qf_loss = self.qf_criterion(y_pred, y_target)

        """
        Update networks
        """
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        if self.gradient_max_value is not None:
            torch.nn.utils.clip_grad_value_(
                self.qf.parameters(), self.gradient_max_value
            )
        self.qf_optimizer.step()
        self._update_target_network()

        """
        Save some statistics for eval using just one batch.
        """
        if self.need_to_update_eval_statistics:
            self.need_to_update_eval_statistics = False
            self.eval_statistics["QF Loss"] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Y Predictions",
                    ptu.get_numpy(y_pred),
                )
            )
            grads = torch.tensor([], device=self.qf.device)
            for param in self.qf.parameters():
                grads = torch.cat((grads, torch.abs(param.grad.data.flatten())))
            self.eval_statistics["Gradient"] = grads.mean().item()

    def _update_target_network(self):
        if self.use_hard_updates:
            if self._n_train_steps_total % self.hard_update_period == 0:
                ptu.copy_model_params_from_to(self.qf, self.target_qf)
        else:
            ptu.soft_update_from_to(self.qf, self.target_qf, self.tau)

    def offline_evaluate(self, epoch):
        raise NotImplementedError()

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot.update(
            exploration_policy=self.exploration_policy,
            policy=self.policy,
        )
        return snapshot

    @property
    def networks(self):
        return [
            self.qf,
            self.target_qf,
        ]
