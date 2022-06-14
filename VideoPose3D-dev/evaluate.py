# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -*- coding: utf-8 -*-
"""
Class for model evaluation

@author: Denis Tome'

"""
import numpy as np

__all__ = [
    'EvalBody',
    'EvalUpperBody',
    'EvalLowerBody'
]
def set_action():
    """Clastering specialized actions in groups

    Returns:
        dict -- action names mapping
    """

    _NAMES = [
        'Gesticuling', 'Reacting', 'Greeting',
        'Talking', 'UpperStretching', 'Gaming',
        'LowerStretching', 'Patting', 'Walking', 'All'
    ]

    _ACTION = {
        'anim_Clip1': 8, 'Opening_A_Lid': 0, 'Dribble': 5, 'Boxing': 5,
        'Standing_Arguing__1_': 3, 'Happy': 3, 'Plotting': 3, 'Counting': 4,
        'Standing_Arguing': 0, 'Standing_2H_Cast_Spell_01': 4, 'Shooting_Gun': 5,
        'Two_Hand_Spell_Casting': 0, 'Shaking_Hands_2': 2, 'Hands_Forward_Gesture': 2,
        'Rifle_Punch': 1, 'Baseball_Umpire': 5, 'Angry_Gesture': 0, 'Waving_Gesture': 0,
        'Taunt_Gesture': 0, 'Golf_Putt_Failure': 5, 'Rejected': 1, 'Shake_Fist': 2,
        'Revealing_Dice': 5, 'Golf_Putt_Failure__1_': 5, 'No': 3, 'Angry_Point': 1,
        'Agreeing': 3, 'Sitting_Thumbs_Up': 6, 'Standing_Thumbs_Up': 4, 'Patting': 7,
        'Petting': 7, 'Petting_Animal': 7, 'Taking_Punch': 0,
        'Standing_1H_Magic_Attack_01': 4, 'Talking': 3, 'Standing_Greeting': 2,
        'Happy_Hand_Gesture': 0, 'Dismissing_Gesture': 1, 'Strong_Gesture': 1,
        'Pointing_Gesture': 1, 'Golf_Putt_Victory': 5, 'Pointing': 0,
        'Thinking': 4, 'Loser': 1, 'Reaching_Out': 3, 'Crazy_Gesture': 0,
        'Golf_Putt_Victory__1_': 5, 'Insult': 3, 'Arm_Gesture': 0,
        'Beckoning': 1, 'Charge': 5, 'Weight_Shift_Gesture': 8,
        'Pain_Gesture': 1, 'Fist_Pump': 0, 'Terrified': 1, 'Surprised': 1,
        'Clapping': 1, 'Rallying': 1, 'Hand_Raising': 0, 'Sitting_Disapproval': 6,
        'Quick_Formal_Bow': 2, 'Counting__1_': 0, 'Tpose_Take_001': 4,
        'upper_stretching': 4, 'lower_stretching': 6, 'walking': 8
    }

    action_map = {}
    for k, v in _ACTION.items():
        action_map[k] = _NAMES[v]

    return action_map

def compute_error(pred, gt):
    """Compute error

    Arguments:
        pred {np.ndarray} -- format (N x 3)
        gt {np.ndarray} -- format (N x 3)

    Returns:
        float -- error
    """

    if pred.shape[1] != 3:
        pred = np.transpose(pred, [1, 0])

    if gt.shape[1] != 3:
        gt = np.transpose(gt, [1, 0])

    assert pred.shape == gt.shape
    error = np.sqrt(np.sum((pred - gt)**2, axis=1))

    return np.mean(error)  # MPJPE


class EvalBody():
    """Eval entire body"""

    def eval(self, pred, gt, actions=None):
        """Evaluate

        Arguments:
            pred {np.ndarray} -- predictions, format (bs x N x 3)
            gt {np.ndarray} -- ground truth, format (bs x N x 3)

        Keyword Arguments:
            action {str} -- action name (default: {None})
        """
        self.error = {'All': []}
        self.action_map = set_action
        for pid, (pose_in, pose_target) in enumerate(zip(pred, gt)):
            err = compute_error(pose_in*1000, pose_target*1000)

            if actions:
                act_name = self._map_action_name(actions[pid])

                # add element to dictionary if not there yet
                if not self._is_action_stored(act_name):
                    self._init_action(act_name)
                self.error[act_name].append(err)

            # add to all
            act_name = 'All'
            self.error[act_name].append(err)

    def desc(self):
        return 'Average3DError'
    def get_results(self):
        """Get results

        Returns:
            dict -- results per action
        """

        results = {}
        for k, v in self.error.items():
            results.update({k: float(np.mean(v))})  # v is a list

        return results


class EvalUpperBody():
    """Eval upper body"""

    _SEL = [0, 1, 2, 3, 4, 5, 6, 7]

    def eval(self, pred, gt, actions=None):
        """Evaluate

        Arguments:
            pred {np.ndarray} -- predictions, format (bs x N x 3)
            gt {np.ndarray} -- ground truth, format (bs x N x 3)

        Keyword Arguments:
            action {str} -- action name (default: {None})
        """
        self.error = {'All': []}
        self.action_map = set_action
        for pid, (pose_in, pose_target) in enumerate(zip(pred, gt)):
            err = compute_error(pose_in[self._SEL]*1000, pose_target[self._SEL]*1000)

            if actions:
                act_name = self._map_action_name(actions[pid])

                # add element to dictionary if not there yet
                if not self._is_action_stored(act_name):
                    self._init_action(act_name)
                self.error[act_name].append(err)

            # add to all
            act_name = 'All'
            self.error[act_name].append(err)
    def get_results(self):
        """Get results

        Returns:
            dict -- results per action
        """

        results = {}
        for k, v in self.error.items():
            results.update({k: float(np.mean(v))})  # v is a list

        return results

    def desc(self):
        return 'UpperBody_Average3DError'
    def get_results(self):
        """Get results

        Returns:
            dict -- results per action
        """

        results = {}
        for k, v in self.error.items():
            results.update({k: float(np.mean(v))})  # v is a list

        return results


class EvalLowerBody():
    """Eval lower body"""

    _SEL = [8, 9, 10, 11, 12, 13, 14, 15,16]

    def eval(self, pred, gt, actions=None):
        """Evaluate

        Arguments:
            pred {np.ndarray} -- predictions, format (bs x N x 3)
            gt {np.ndarray} -- ground truth, format (bs x N x 3)

        Keyword Arguments:
            action {str} -- action name (default: {None})
        """
        self.error = {'All': []}
        self.action_map = set_action
        for pid, (pose_in, pose_target) in enumerate(zip(pred, gt)):
            err = compute_error(pose_in[self._SEL]*1000, pose_target[self._SEL]*1000)

            if actions:
                act_name = self._map_action_name(actions[pid])

                # add element to dictionary if not there yet
                if not self._is_action_stored(act_name):
                    self._init_action(act_name)
                self.error[act_name].append(err)

            # add to all
            act_name = 'All'
            self.error[act_name].append(err)

    def desc(self):
        return 'LowerBody_Average3DError'
    def get_results(self):
        """Get results

        Returns:
            dict -- results per action
        """

        results = {}
        for k, v in self.error.items():
            results.update({k: float(np.mean(v))})  # v is a list

        return results
