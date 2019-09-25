import numpy as np

from bc.model.base import MetaPolicy


class FlatPolicy(MetaPolicy):
    "Vanilla Architecture"
    "One head prediciting current action and future actions using concatenated Standard Blocks"

    def __init__(self, **network_args):
        super(FlatPolicy, self).__init__(**network_args)

    def compute_loss(self, obs, actions, signals, skills, eval=False):
        pred = self.__call__(obs, signals)
        actions = actions.to(self.device)

        # index of the beginning of each basic blocs
        # when multiple actions are predicted
        idx_net = np.arange(0, self.dim_prediction, self.dim_action + 1)
        idx_action = np.arange(0, self.dim_gt, self.dim_action)
        loss_grip = 0
        loss_move = 0
        # Each basic bloc is separated in two parts
        # First two columns is the gripper state classification : apply CE loss
        # Columns 2 to 7 are the linear and angular velocity regression :
        # apply L2 loss
        for idx_n, idx_a in zip(idx_net, idx_action):
            loss_grip += self.ce_loss(pred[:, idx_n:idx_n + 2],
                                      (actions[:, idx_a] < 0).long())
            loss_move += self.l2_loss(
                pred[:, idx_n + 2:idx_n + self.dim_action + 1],
                actions[:, idx_a + 1:idx_a + self.dim_action])
        loss = self.lam_grip * loss_grip + (1 - self.lam_grip) * loss_move
        if not eval:
            loss.backward()
        return {'l2_move': loss_move, 'ce_grip': loss_grip}
