import numpy as np
import torch

from bc.model.base import MetaPolicy


class FilmPolicy(MetaPolicy):
    """Skills Architecture
    One head prediciting subpolicy to perform and one head for each subpolicy which predicts
    the current action and future actions using concatenated Standard Blocks"""

    def __init__(self, lam_master=0.0, num_skills=2, **network_args):
        # dim_action+1 for predicted actions as the outputs proba p0,p1 of the gripper begin open,close
        self.lam_master = lam_master
        assert num_skills >= 2
        self.num_skills = num_skills

        super(FilmPolicy, self).__init__(
            network_extra_args=dict(num_skills=self.num_skills),
            **network_args)

    def compute_loss(self, obs, actions, signals, skills, eval=False):
        # pred contains the master prediction of which subpolicy to choose
        # and the control output by every subpolicy
        # target contains the master gt and the control gt for the gt subpolicy
        pred = self.__call__(obs, skills)
        # pred_master = pred[:, -self.num_skills:]
        # pred = pred[:, :-self.num_skills]
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

    def get_action(self, obs, signals=None, skill=None):
        assert obs.shape[0] == 1
        if skill is not None:
            skill = torch.tensor([skill])
        pred = self.__call__(obs, skill, compute_grad=False)
        pred = pred[0].cpu().numpy()[:self.dim_action + 1]
        return pred
