import numpy as np
import torch

from bc.model.base import MetaPolicy


class SkillsPolicy(MetaPolicy):
    """Skills Architecture
    One head prediciting subpolicy to perform and one head for each subpolicy which predicts
    the current action and future actions using concatenated Standard Blocks"""

    def __init__(self, lam_master=0.0, num_skills=2, **network_args):
        # dim_action+1 for predicted actions as the outputs proba p0,p1 of the gripper begin open,close
        self.lam_master = lam_master
        assert num_skills >= 2
        self.num_skills = num_skills

        super(SkillsPolicy, self).__init__(
            network_extra_args=dict(num_skills=self.num_skills),
            **network_args)

    def compute_loss(self, obs, actions, signals, skills, eval=False):
        # pred contains the master prediction of which subpolicy to choose
        # and the control output by every subpolicy
        # target contains the master gt and the control gt for the gt subpolicy
        pred = self.__call__(obs)
        pred_master = pred[:, -self.num_skills:]
        pred_subpolicies = pred[:, :-self.num_skills]
        actions = actions.to(self.device)
        skills = skills.to(self.device)
        loss_master = self.ce_loss(pred_master, skills)

        # extract pred on subpolicies picked according to ground truth
        # this is the only subpolicy where ground truth is available
        mask = torch.zeros(pred_subpolicies.shape, dtype=torch.uint8)
        for i, gt_subpolicy in enumerate(skills):
            mask[i, gt_subpolicy * self.dim_prediction:(gt_subpolicy + 1) *
                 self.dim_prediction] = 1
        pred_subpolicies_gt = pred_subpolicies[mask].view(-1, self.dim_prediction)

        # index of the beginning of each basic block when multiple actions are predicted
        idx_net = np.arange(0, self.dim_prediction, self.dim_action + 1)
        idx_action = np.arange(0, self.dim_gt, self.dim_action)
        loss_grip, loss_move = 0, 0

        # each basic block is separated in two parts
        # first two columns is the gripper state classification : apply CE loss
        # columns 2 to 7 are the linear and angular velocity regression : apply L2 loss
        for idx_n, idx_a in zip(idx_net, idx_action):
            loss_grip += self.ce_loss(pred_subpolicies_gt[:, idx_n:idx_n + 2],
                                      (actions[:, idx_a] < 0).long())
            loss_move += self.l2_loss(
                pred_subpolicies_gt[:, idx_n + 2:idx_n + self.dim_action + 1],
                actions[:, idx_a + 1:idx_a + self.dim_action])
        # debugging the skills
        loss_skills_move = torch.zeros(self.num_skills).to(self.device)
        loss_skills_grip = torch.zeros(self.num_skills).to(self.device)
        skills_count_dict = {}
        for skill in range(self.num_skills):
            if (skills == skill).sum() > 0:
                for idx_n, idx_a in zip(idx_net, idx_action):
                    loss_skills_grip[skill] += self.ce_loss(
                        pred_subpolicies_gt[skills == skill, idx_n:idx_n + 2],
                        (actions[skills == skill, idx_a] < 0).long())
                    loss_skills_move[skill] += self.l2_loss(
                        pred_subpolicies_gt[skills == skill, idx_n + 2:idx_n +
                                            self.dim_action + 1],
                        actions[skills == skill, idx_a + 1:idx_a +
                                self.dim_action])
            skills_count_dict['ratio/skill{}'.format(skill)] = torch.tensor(
                (skills == skill).sum().item() / skills.shape[0])
        loss_skills_move_dict = {
            'l2_move/skill{}'.format(k): v
            for k, v in enumerate(loss_skills_move)
        }
        loss_skills_grip_dict = {
            'ce_grip/skill{}'.format(k): v
            for k, v in enumerate(loss_skills_grip)
        }
        loss_subpolicy = self.lam_grip * loss_grip + (
            1 - self.lam_grip) * loss_move
        loss = self.lam_master * loss_master + (
            1 - self.lam_master) * loss_subpolicy
        if not eval:
            loss.backward()
        loss_dict = {
            'l2_move': loss_move,
            'ce_grip': loss_grip,
            'ce_master': loss_master
        }
        loss_dict.update(loss_skills_move_dict)
        loss_dict.update(loss_skills_grip_dict)
        loss_dict.update(skills_count_dict)
        return loss_dict

    def get_action(self, obs, signals=None, skill=None):
        assert obs.shape[0] == 1
        pred = self.__call__(obs, signals, compute_grad=False)
        pred = pred[0].cpu().numpy()
        if skill is None:
            # index of the subpolicy to choose predicted by the master
            skill = pred[-self.num_skills:].argmax()
        # controls predicted by subpolicies
        pred_subpolicies = pred[:-self.num_skills]
        # control predicted by the master chosen subpolicy
        pred_subpolicy = pred_subpolicies[
            skill * self.dim_prediction: (skill + 1) * self.dim_prediction]
        pred = pred_subpolicy[:self.dim_action + 1]
        return pred
