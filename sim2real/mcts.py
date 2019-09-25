import numpy as np
import pickle as pkl
import os

from sim2real.transformations import ImageTransform


class MonteCarloNode:
    """
    transform: current image transformation of the node
    attribute: attribute of the transformation determined by this node children,
    the action space of the node is the range of the attribute
    state: index chosen within attribute range of parent
    """

    def __init__(self, parent, transform, attribute, state, c_exp):
        self.parent = parent
        self.transform = transform
        self.attribute = attribute
        self.state = state
        self.num_visits = 0
        self.total_score = 0
        self.children = []
        self.c_exp = c_exp
        self.unexplored_actions = list(transform.attribute2range(attribute))

    def __getitem__(self, idx):
        assert idx < len(self.children)
        return self.children[idx]

    def add_child(self, child):
        assert isinstance(child, MonteCarloNode)
        self.children.append(child)
        self.unexplored_actions = [
            a for a in self.unexplored_actions if a != child.state
        ]

    @property
    def uct(self):
        """Returns UCT score, None if the node has never been explored."""
        # each created node is visited while backproping the final score
        # this value should then automically be positive
        assert self.num_visits > 0
        exploration = np.sqrt(np.log(self.parent.num_visits) / self.num_visits)
        return self.score_avg + float(self.c_exp) * exploration

    @property
    def score_avg(self):
        assert self.num_visits > 0
        return self.total_score / self.num_visits

    def __str__(self):
        """Return attribute of the node, state of the children and their score"""
        description = '\n'.join([
            'State {}, avg score {:.3f}'.format(child.state, child.score_avg)
            for child in self.children
        ])
        return description


class MonteCarloTree(object):
    """max_transforms: maximum number of transformations than can be applied consecutively
    3 * max_transformations is the maximal height of the tree.
    attributes_node: number of attributes definind a node : name, magnitude, probability"""

    def __init__(self,
                 max_transforms,
                 c_exp=None,
                 pickle_path='',
                 backprop_score_name='median_score'):
        self.max_transforms = max_transforms
        if c_exp is None:
            c_exp = np.sqrt(2)
        self._c_exp = c_exp
        self._backprop_score_name = backprop_score_name
        self._attributes_node = 3
        self.iterations = 0
        self.load(pickle_path)

        self._transform_names = self.root.transform.attribute2range('name')

    def load(self, pickle_path):
        # setting the transform name of image transform has no effect on root behavior
        self.root = MonteCarloNode(
            parent=None,
            transform=ImageTransform('identity'),
            attribute='name',
            state=0,
            c_exp=self._c_exp)
        self._history = []

        if pickle_path:
            dic = pkl.load(open(pickle_path, 'rb'))
            history = dic['history']
            max_transforms = len(history[0][0]) // self._attributes_node

            self.max_transforms = max_transforms
            if 'c_exp' in dic:
                self._c_exp = dic['c_exp']
            if 'backprop_score_name' in dic:
                self._backprop_score_name = dic['backprop_score_name']

            for path, score in history:
                self.add_path(path, score)

    def get_uct_action(self, node):
        """Returns best actions according to UCT score.
        If some action have never been visited, randomly sample among them
        and create a new node."""

        # remove redundancies with respect to previous node transformations
        # only remove_object is considered as non-redundant
        if node.attribute == 'name':
            parent_transforms = []
            parent_node = node.parent
            while parent_node is not None:
                if parent_node.state in self._transform_names:
                    parent_transforms.append(parent_node.state)
                parent_node = parent_node.parent
            unexplored_actions = [
                action for action in node.unexplored_actions
                if action not in parent_transforms or action == 'identity'
            ]
        else:
            unexplored_actions = node.unexplored_actions

        # expand the tree if some branches are unexplored
        if len(unexplored_actions) > 0:
            idx_action = np.random.randint(len(unexplored_actions))
            action = unexplored_actions[idx_action]
            child_node = self.expand_node(node, action)
            return action, child_node

        # else got to the child with best uct score
        scores = [child.uct for child in node.children]
        best_child = node.children[np.argmax(scores)]
        best_action = best_child.state

        return best_action, best_child

    def expand_node(self, node, action):
        if node.attribute == 'name':
            child_transform = ImageTransform(action)
        else:
            child_transform = node.transform
        child_attribute = child_transform.attribute2child(node.attribute)
        child_node = MonteCarloNode(node, child_transform, child_attribute,
                                    action, self._c_exp)
        node.add_child(child_node)
        return child_node

    def add_path(self, path, policy_scores):
        """
        given a path of the form (name0, magn0, proba0, name1, ...) and a score,
        browse and expand the tree then backprop the socre
        """
        node = self.root
        for action in path:
            if action in node.unexplored_actions:
                node = self.expand_node(node, action)
            else:
                node = [
                    child for child in node.children if child.state == action
                ][0]

        self.backprop_score(
            node, float(policy_scores[str(self._backprop_score_name)]))
        self._history.append((path, policy_scores))
        self.iterations += 1

    def sample_path(self, path=None, relative_root=None):
        """
        path is defined as a sequence of integer in node action_space
        which defines the transformation.
        [1, 3, 4, 2, 4, 0] will choose to apply image transform 1 with value 3
        of magnitude range and value 4 of probability range etc.
        """
        if path is None:
            path = []

        node = self.root if relative_root is None else relative_root

        # if the path is of maximal length, return it
        if len(path) // self._attributes_node >= self.max_transforms:
            return path, relative_root

        best_action, best_child = self.get_uct_action(node)
        path.append(best_action)
        return self.sample_path(path, best_child)

    def backprop_score(self, node, score):
        node.num_visits += 1
        node.total_score += score
        if node.parent is not None:
            self.backprop_score(node.parent, score)

    def add_to_history(self, path, score):
        self._history.append({'path': path, 'score': score})

    def save(self, save_dir):
        dic_save = {
            'root': self.root,
            'history': self._history,
            'c_exp': self._c_exp,
            'backprop_score_name': self._backprop_score_name
        }
        path_mcts = os.path.join(save_dir, 'mcts.pkl')
        pkl.dump(dic_save, open(path_mcts, 'wb'))

    def __str__(self):
        return str(self.root)
