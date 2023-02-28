import copy

import pickle
import os.path
from typing import Dict, Union
import random
import numpy as np



def load_pickle(filepath: str):
    with open(filepath, 'rb') as pkl_file:
        obj = pickle.load(pkl_file)
    return obj


def save_pickle(obj, filepath: str):
    with open(filepath, 'wb') as pkl_file:
        pickle.dump(obj, pkl_file)


def cache_calc(func, path):
    if os.path.exists(path):
        obj = load_pickle(path)
    else:
        obj = func()
        save_pickle(obj, path)
    return obj




def opposite_dict(d: Dict):
    o = {}
    for k, v in d.items():
        o[v] = k
    return o





def pythagorean_fn(p, distance_dict):
    num_params = len(p)
    param_index_combinations = distance_dict.keys()
    equations = []
    for i, j in param_index_combinations:
        equations.append(p[i]**2 + p[j]**2 - distance_dict[(i, j)]**2)
    return equations


def label_to_onehot(labels):
    uniq_labels = list(set(labels))
    num_labels = len(uniq_labels)
    num_samples = len(labels)
    oh_labels = []
    label_to_oh = {}
    for i, l in enumerate(uniq_labels):
        tmp = np.zeros(num_labels)
        tmp[i] = 1
        label_to_oh[l] = tmp
    for label in labels:
        oh_labels.append(label_to_oh[label])
    return np.vstack(oh_labels)



def stable_matching(data:np.ndarray):
    data_copy = np.copy(data)
    initial_matches = np.argmax(data, axis=0)
    num_assignments = data.shape[1]

    a_assigned = []
    b_assigned = []
    a_unassigned = list(range(num_assignments))

    while len(a_assigned) < num_assignments and np.sum(data_copy[:, a_unassigned]) != 0:
        for i_a, i_b in enumerate(initial_matches):
            if i_b not in b_assigned and i_a not in a_assigned:
                b_assigned.append(i_b)
                a_assigned.append(i_a)

        data_copy[b_assigned] *= 0
        initial_matches = np.argmax(data_copy, axis=0)
        a_unassigned = [x for x in range(num_assignments) if x not in a_assigned]

    if len(a_assigned) < num_assignments:
        # assign at random
        # a_unassigned = [x for x in range(num_assignments) if x not in a_assigned]
        for i in a_unassigned:
            b_unassigned = [x for x in range(data.shape[0]) if x not in b_assigned]
            a_assigned.append(i)
            b_assigned.append(random.choice(b_unassigned))

    return dict(zip(a_assigned, b_assigned))


