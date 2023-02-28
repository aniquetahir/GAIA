import functools
import os
import chex
import numpy as onp
from utils.common import load_pickle, save_pickle
import jax
import haiku as hk
import jax.numpy as jnp
from typing import Dict, List, Tuple
import argparse
from jax.random import split as ksplit
from utils.jax.models.bnn import BNN, get_weight_sampler
import matplotlib.pyplot as plt
import seaborn as sns
import optax
import pandas as pd
import plotly.express as px
from utils.jax.common import restructure
from jax import jit, vmap
from utils.jax.common import destructure
from oryx.core.ppl import log_prob
from sklearn.metrics import roc_auc_score
from collections import defaultdict



plt.style.use('ggplot')
sns.set(rc={'figure.dpi':300, 'figure.figsize':(11.7,8.27)})
LEARNING_RATE = 0.005
onp.random.seed(10)
num_iters = 100000
SIGMA2 = 0.002

def unc_aware_data(bnn_combined: BNN, control_bnns: Tuple[BNN], train_data, protected_attrs: List[int], n=100):
    non_prot_attrs = [x for x in range(train_data.shape[1]) if x not in protected_attrs]
    train_data = train_data[:, non_prot_attrs]
    mean_preds_combined = bnn_combined.get_mean_predictions(train_data, n)
    # Get the uncertainty values
    alea2_combined = bnn_combined.get_aleatoric_uncertainty_v2(train_data, n).reshape(-1, 1)
    epist_combined = bnn_combined.get_epistemic_uncertainty(train_data, n).reshape(-1, 1)

    all_cont_preds = {}
    all_cont_epist = {}
    all_cont_alea = {}
    # For each control bnn
    for i, cont_bnn in enumerate(control_bnns):
        ## Get the predictions
        cont_preds = cont_bnn.get_mean_predictions(train_data, n)
        all_cont_preds[i] = cont_preds

        ## Get the epistemic uncertainty
        all_cont_epist[i] = cont_bnn.get_epistemic_uncertainty(train_data, n)
        ## Get the aleatoric uncertainty
        all_cont_alea[i] = cont_bnn.get_aleatoric_uncertainty_v2(train_data, n)

    all_cont_preds = jnp.hstack(list(all_cont_preds.values()))
    all_cont_epist = jnp.hstack([x.reshape(-1, 1) for x in list(all_cont_epist.values())])
    all_cont_alea = jnp.hstack([x.reshape(-1, 1) for x in list(all_cont_alea.values())])

    sg = jax.lax.stop_gradient
    # Stack everything horizontally
    combined_training_data = jnp.hstack((train_data,
                                         sg(mean_preds_combined),
                                         sg(all_cont_preds),
                                         sg(epist_combined),
                                         sg(all_cont_epist),
                                         sg(alea2_combined),
                                         sg(all_cont_alea) ))
    return combined_training_data


def f_preds_unc(key: chex.PRNGKey, bnn_combined: BNN, control_bnns: Tuple[BNN],
                train_data, train_labels, test_data, test_labels,
                protected_attrs: List[int], n=100):
    combined_training_data = unc_aware_data(bnn_combined, control_bnns, train_data, protected_attrs, n)
    combined_test_data = unc_aware_data(bnn_combined, control_bnns, test_data, protected_attrs, n)
    # Train a Simple MLP on the combined data
    classifier = SimpleMLPClassifierWithClassDistributionRandomization(
        key, combined_training_data[:10].shape, 3, 256, 0.1
    )

    classifier.fit(combined_training_data, train_labels,
                   num_iterations=100000, train_data_eval=combined_training_data,
                   test_data=combined_test_data, train_labels_eval=train_labels, test_labels=test_labels, verbose=True)

    return classifier


class MyBNN(BNN):
    def get_loss_fn(self, NUM_MINIBATCHES):
        # NUM_MINIBATCHES = NUM_MINIBATCHES
        log_pi_denom = onp.log((2. ** NUM_MINIBATCHES) - 1)
        @functools.partial(jit, static_argnames=['apply_fn', 'prior'])
        def loss_fn(b_params, key, data, labels, apply_fn, prior, i):
            mu, rho = b_params['mus'], b_params['rhos']
            logits, params = apply_fn(key, b_params, data)

            # get the logits for the mean predictor
            params_mlp = restructure(self._mlp_params, b_params['mus'], self._destructure_ranges, jax.tree_util.tree_structure(self._mlp_params))
            logits_mus = self._mlp.apply(params_mlp, data)

            # Regularization
            ## Make the means closer to zero
            l2_reg = jnp.sqrt((mu ** 2).mean())


            params = destructure(params, jax.tree_util.tree_structure(params))

            log_prior = log_prob(prior)(params).mean()
            log_variational_post = log_prob(get_weight_sampler(mu, rho))(params)
            nll = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
            nll_mus = optax.softmax_cross_entropy_with_integer_labels(logits_mus, labels).mean()

            # Some things are preprocessed outside jit because of large number calculations.
            pi = jnp.exp(((NUM_MINIBATCHES - i) * jnp.log(2)) - log_pi_denom)
            # pi = (2. ** (pi_param - i))/((2. ** pi_param) -1 )
            # jax.debug.print('{a}', a=pi)
            loss = ((log_variational_post - log_prior) * pi) + nll + 0.001 * l2_reg
            # + nll_mus * 0.5 #\
            # + 0.01 * l2_reg
            # loss = nll
            return loss
        return loss_fn


    def sample_random_batch(self, key, data_all, prot_attr_mask: chex.Array, batch_size):
        data, labels = data_all

        fav_mask = (labels == 1).astype(float).flatten()
        unfav_mask = (labels == 0).astype(float).flatten()
        num_fav, num_unfav = fav_mask.sum(), unfav_mask.sum()

        prot_mask = (prot_attr_mask == 1).astype(float).flatten() # TODO fix
        prot_alt_mask = (prot_attr_mask == 0).astype(float).flatten()
        num_prot, num_prot_alt = prot_mask.sum(), prot_alt_mask.sum()

        key, split = ksplit(key)
        perc_prot = jax.random.uniform(split)
        m_prot_mask = prot_mask * perc_prot
        m_protalt_mask = prot_alt_mask * (1 - perc_prot)
        selection_prot_probs = (m_prot_mask/num_prot) + (m_protalt_mask/num_prot_alt)

        key, split = ksplit(key)
        perc_fav = jax.random.uniform(split)
        m_fav_mask = fav_mask * perc_fav
        m_unfav_mask = unfav_mask * (1 - perc_fav)
        selection_probs = (m_fav_mask/num_fav) + (m_unfav_mask/num_unfav)

        key, split = ksplit(key)

        batch_idx = jax.random.choice(split, jnp.arange(data.shape[0]), (batch_size, ), replace=False, p=selection_probs)
        # batch_idx = jax.random.choice(split, jnp.arange(data.shape[0]), (batch_size, ), replace=False, p=selection_probs * selection_prot_probs)
        return data[batch_idx], labels[batch_idx]


    def get_trainable_param_components(self, frozen_keys: List[str] = []):
        if len(frozen_keys) == 0:
            return self._baysian_params, {'mus': jnp.array([]), 'rhos': jnp.array([])}

        # Get the parameters for the frozen layers
        frozen_layers = {}
        for k in frozen_keys:
            frozen_layers[k] = self._mlp_params[k]

        # Get the size of the parameters
        parameter_size = destructure(frozen_layers, jax.tree_util.tree_structure(frozen_layers)).shape[0]

        # split the baysian params based on the size
        tm = jax.tree_util.tree_map
        bp = self._baysian_params
        return tm(lambda x: x[parameter_size:], bp), tm(lambda x: x[:parameter_size], bp)

    def train_loop(self, num_iterations, data_iter: List[jnp.ndarray], valid_iter, update_step, loss_fn, extra):
        # Objectives
        # - Implement a jittable training loop
        # - Add printing for the loss values and the evaluation accuracies
        # - Sample the label distribution inside the loop
        # - Calculate the training dynamics i.e., confidence and variability
        prot_attrs: List[int] = extra['prot_attrs']

        # Filter the data based on the protected attributes and create a mask for the protected attributes
        non_prot_attrs = [x for x in list(range(data_iter[0].shape[1])) if x not in prot_attrs]
        prot_mask = data_iter[0][:, prot_attrs]
        prot_mask = jnp.array(prot_mask)

        # Change the data so that it does not contain the protected attributes
        data_iter = (data_iter[0][:, non_prot_attrs], data_iter[1], )

        print_freq = num_iterations // 10
        batch_size = extra['batch_size']
        NUM_MINIBATCHES = len(data_iter[0])/batch_size

        NUM_MINIBATCHES = NUM_MINIBATCHES * 1.5 # Inspired by Nyquist Theorem (since we use random sampling)

        @jax.jit
        def update_step(i, key, batch_data, batch_labels, params, opt_state):

            elbo, grads = jax.value_and_grad(loss_fn)(params, key, batch_data, batch_labels, self.predict_w_sample, self.prior,
                                                      (i % NUM_MINIBATCHES) + 1)
            updates, opt_state = self._optim.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            return params, opt_state, elbo


        sample_random_batch = jax.jit(functools.partial(self.sample_random_batch, batch_size=batch_size))

        def print_validation(print_args):
            i, elbo, acc = print_args

            if i % print_freq == 0 and self._verbose:
                print(f'iter: {i}, elbo: {elbo}, valid_acc: {acc}')


        def calc_valid_acc(key_params):
            key, params = key_params
            # jax.debug.print('{a}', a=key)
            valid_limit = 10000
            params_mlp = restructure(self._mlp_params, params['mus'], self._destructure_ranges, jax.tree_util.tree_structure(self._mlp_params))
            # jax.debug.print('mus head: {a}', a = params['mus'][:10])
            logits = self._mlp.apply(params_mlp, valid_iter[0][:valid_limit][:, 1:])
            train_logits = self._mlp.apply(params_mlp, data_iter[0][:valid_limit])
            # logits, _ = self.predict_w_sample(key, params, valid_iter[0][:valid_limit][:, 1:])
            valid_acc = (jnp.argmax(logits, axis=1) == valid_iter[1][:valid_limit]).astype(jnp.float32).mean()
            train_acc = (jnp.argmax(train_logits, axis=1) == data_iter[1][:valid_limit]).astype(jnp.float32).mean()

            if self._verbose:
                jax.debug.print('valid acc:{a}, train acc: {b}', a=valid_acc, b=train_acc)
            return valid_acc, train_acc


        def calc_train_acc(key_params):
            key, params = key_params
            # jax.debug.print('{a}', a=key)
            valid_limit = 10000
            params_mlp = restructure(self._mlp_params, params['mus'], self._destructure_ranges, jax.tree_util.tree_structure(self._mlp_params))
            # jax.debug.print('mus head: {a}', a = params['mus'][:10])
            # logits = self._mlp.apply(params_mlp, valid_iter[0][:valid_limit][:, 1:])
            train_logits = self._mlp.apply(params_mlp, data_iter[0][:valid_limit])
            # logits, _ = self.predict_w_sample(key, params, valid_iter[0][:valid_limit][:, 1:])
            # valid_acc = (jnp.argmax(logits, axis=1) == valid_iter[1][:valid_limit]).astype(jnp.float32).mean()
            train_acc = (jnp.argmax(train_logits, axis=1) == data_iter[1][:valid_limit]).astype(jnp.float32).mean()

            return train_acc


        @functools.partial(jax.jit, static_argnames=('num_confirmations'))
        def get_train_probs(key, params, num_confirmations = 50):
            average_probs = jnp.zeros((data_iter[0].shape[0], self._num_classes))
            for i in range(num_confirmations):
                key, split = ksplit(key)
                logits, _ = self.predict_w_sample(split, params, data_iter[0])
                logit_probs = jax.nn.softmax(logits)
                average_probs += logit_probs
            return average_probs/num_confirmations

        epoch_train_probs = []
        def add_train_probs(probs):
            epoch_train_probs.append(onp.array(probs))

        @jax.jit
        def loop_fn(carry, i):
            key, params, opt_state, data, best_model, last_model_stats = carry
            best_stats, best_params = best_model
            key, i_key = ksplit(key)
            key, split = ksplit(key)
            x, y = sample_random_batch(split, data, prot_mask)
            # update the params and optimizer state
            params, opt_state, elbo = update_step(i, i_key, x, y, params, opt_state)

            key, split = ksplit(key)
            valid_acc, train_acc = jax.lax.cond(i % print_freq == 0, calc_valid_acc, lambda k: (0., 0.), (split, params))
            key, split = ksplit(key)
            train_acc = calc_train_acc((split, params))
            key, split = ksplit(key)
            # jax.debug.print('loop {a}', a = i)
            jax.debug.callback(print_validation, (i, elbo, valid_acc))
            key, split = ksplit(key)
            train_probs = jax.lax.cond(i % len(data_iter[1]) == 0, # WARNING. This i already has mod applied
                                       lambda k: get_train_probs(k[0], k[1]),
                                       lambda k: jnp.zeros((len(data_iter[0]), self._num_classes)),
                                       (split, params))
            jax.lax.cond(i % len(data_iter[1]) == 0,
                         lambda k: jax.debug.callback(add_train_probs, k),
                         lambda k: [], train_probs)



            curr_stats = train_acc
            # jax.lax.cond(i % 100 == 0,
            #              lambda k: jax.debug.print('model running stats: {a}', a=k),
            #              lambda k: None, curr_stats)

            # Average out the current stats using the previous stats
            curr_stats = (curr_stats + last_model_stats) / 2

            # if current stats > best stats
            # operands = curr_stats, curr_params, best_stats, best_params
            best_stats, best_params = jax.lax.cond(
                curr_stats > best_stats,
                # if true
                lambda x: (x[0], x[1]),
                lambda x: (x[2], x[3]),
                (curr_stats, params, best_stats, best_params)
            )
            ## Update the best stats
            return (key, params, opt_state, data,
                    (best_stats, best_params), curr_stats) , x


        (_, self._baysian_params, self._opt_state, _, (best_stats, best_params), _), train_prob_iterations = \
            jax.lax.scan(loop_fn,
                         (self.next_key(), self._baysian_params, self._opt_state,
                          data_iter, (0., self._baysian_params), 0.),
                         jnp.arange(num_iterations)
                         )

        # Combine the trainable and the frozen params
        # WARNING Assumption is that the trainable params come last
        # self._baysian_params = jax.tree_util.tree_map(lambda a, b: jnp.concatenate((a, b)), frozen_params, trainable_params)
        self._baysian_params = best_params

        stacked_train_probs = jnp.stack(epoch_train_probs)
        mean_probs_all_classes = jnp.mean(stacked_train_probs, axis=0)
        std_probs_all_classes = jnp.std(stacked_train_probs, axis=0)
        conf_probs = []
        variability_probs = []
        for b, a, i in zip(std_probs_all_classes.tolist(), mean_probs_all_classes.tolist(), data_iter[1].tolist()):
            variability_probs.append(b[i])
            conf_probs.append(a[i])

        confidence_per_sample = onp.array(conf_probs)
        variability_per_sample = onp.array(variability_probs)

        ## Confidence calculation
        _, preds = jax.lax.scan(lambda c, i: (c, jnp.argmax(c[i], axis=1)), stacked_train_probs, jnp.arange(stacked_train_probs.shape[0]))
        preds = preds.T
        correctnesses = []
        for pred_list, gt_label in zip(preds.tolist(), data_iter[1]):
            correctnesses.append((pred_list == gt_label).astype(float).mean())
        correctnesses = onp.array(correctnesses)

        return {
            'confidence': confidence_per_sample,
            'variability': variability_per_sample,
            'correctness': correctnesses
        }


def get_prediction_categories_bnn(bnn_list: List[BNN], data, labels, n=100):
    prediction_list = []
    for bnn in bnn_list:
        preds = jnp.argmax(bnn.get_mean_predictions(data, n=n), axis=1)
        prediction_list.append((preds == labels).astype(int).flatten())

    intersections_mislabeled = jnp.where(prediction_list[0] == 0)[0]
    for l in prediction_list[1:]:
        l_mislabeled = jnp.where(l == 0)[0]
        intersections_mislabeled = jnp.intersect1d(intersections_mislabeled, l_mislabeled)

    intersections_represetative = jnp.where(prediction_list[0] == 1)[0]
    for l in prediction_list[1:]:
        l_representative = jnp.where(l == 1)[0]
        intersections_represetative = jnp.intersect1d(intersections_represetative, l_representative)

    low_confidence = jnp.setdiff1d(jnp.arange(0, len(data)), jnp.union1d(intersections_represetative, intersections_mislabeled))

    return intersections_represetative, intersections_mislabeled, low_confidence


def get_prediction_categories(key: chex.PRNGKey, param_list: List[chex.PyTreeDef], cls_fn, data, labels):
    prediction_list = []
    for m_params in param_list:
        key, split = ksplit(key)
        preds = jnp.argmax(cls_fn(m_params, split, data, False), axis=1)
        prediction_list.append((preds == labels).astype(int).flatten())

    intersections_mislabeled = jnp.where(prediction_list[0] == 0)[0]
    for l in prediction_list[1:]:
        l_mislabeled = jnp.where(l == 0)[0]
        intersections_mislabeled = jnp.intersect1d(intersections_mislabeled, l_mislabeled)

    intersections_represetative = jnp.where(prediction_list[0] == 1)[0]
    for l in prediction_list[1:]:
        l_representative = jnp.where(l == 1)[0]
        intersections_represetative = jnp.intersect1d(intersections_represetative, l_representative)

    low_confidence = jnp.setdiff1d(jnp.arange(0, len(data)), jnp.union1d(intersections_represetative, intersections_mislabeled))

    return intersections_represetative, intersections_mislabeled, low_confidence


def get_adult_data():
    df_adult = load_pickle('adult_df.pd')
    data_numpy = df_adult.to_numpy()
    data_numpy[:, [0, 1]] = data_numpy[:, [1, 0]]
    data_idx = onp.arange(len(data_numpy))
    onp.random.shuffle(data_idx)
    # onp.random.shuffle(data_numpy)
    data_numpy = data_numpy[data_idx]
    data = data_numpy[:, :-1]
    labels = data_numpy[:, -1:].astype(int).flatten()
    train_test_ratio = 0.8
    train_end_index = int(len(data_numpy) * train_test_ratio)

    train_data = data[:train_end_index]
    test_data = data[train_end_index:]
    train_labels = labels[:train_end_index]
    test_labels = labels[train_end_index:]

    fav_idx = onp.where(train_labels == 1)[0]
    unfav_idx = onp.where(train_labels == 0)[0]

    # train a simple classifier on the data
    min_strat = min(len(fav_idx), len(unfav_idx))
    ## pick a balanced sample from the training set
    fav_bal = onp.random.choice(fav_idx, min_strat, replace=False)
    unfav_bal = onp.random.choice(unfav_idx, min_strat, replace=False)
    bal_data = jnp.vstack((train_data[fav_bal], train_data[unfav_bal]))
    bal_labels = jnp.concatenate((jnp.ones(min_strat), jnp.zeros(min_strat)), dtype=jnp.int16).flatten()

    return train_data, train_labels, bal_data, bal_labels, test_data, test_labels


def get_iter_label_dist_randomization(key, data, labels, batch_size):
    fav_mask = (labels == 1).astype(float).flatten()
    unfav_mask = (labels == 0).astype(float).flatten()
    num_fav, num_unfav = fav_mask.sum(), unfav_mask.sum()

    while True:
        splits = ksplit(key, 3)
        key = splits[2]
        perc_fav = jax.random.uniform(splits[0])
        m_fav_mask = fav_mask * perc_fav
        m_unfav_mask = unfav_mask * (1 - perc_fav)
        selection_probs = (m_fav_mask/num_fav) + (m_unfav_mask/num_unfav)
        batch_idx = jax.random.choice(splits[1], jnp.arange(data.shape[0]), (batch_size, ), replace=False, p=selection_probs)
        yield data[batch_idx], labels[batch_idx]


def auc_bnn(bnn: BNN, x, y, n=100):
    preds_mean = bnn.get_mean_predictions(x, n)
    # preds_mean = ensemble_preds.mean(axis=-1)
    return roc_auc_score(onp.array(y), onp.array(preds_mean)[:, 1])

def accuracy_bnn(bnn: BNN, x, y, n=100):
    # ensemble_preds = bnn.get_ensemble_predictions(x, n)
    preds_mean = bnn.get_mean_predictions(x, n) #ensemble_preds.mean(axis=-1)
    preds_mean = jnp.argmax(preds_mean, axis=1)
    return (preds_mean == y).astype(jnp.float32).mean()


def get_confusion_instances(bnn: BNN, x, y, n=100):
    # preds = bnn.get_ensemble_predictions(x, n)
    mean_preds = bnn.get_mean_predictions(x, n)
    mean_preds = jnp.argmax(mean_preds, axis=1)
    tp = jax.tree_util.tree_map(lambda x, y: x and y, (mean_preds == y).tolist(), (mean_preds == 1).tolist())
    tn = jax.tree_util.tree_map(lambda x, y: x and y, (mean_preds == y).tolist(), (mean_preds == 0).tolist())
    fp = jax.tree_util.tree_map(lambda x, y: x and y, (mean_preds != y).tolist(), (mean_preds == 1).tolist())
    fn = jax.tree_util.tree_map(lambda x, y: x and y, (mean_preds != y).tolist(), (mean_preds == 0).tolist())
    return tp, tn, fp, fn

def get_tpr(bnn: BNN, x, y, n=100):
    tp, _, _, fn = get_confusion_instances(bnn, x, y, n)
    tp, fn = onp.array(tp).astype(float).sum(), onp.array(fn).astype(float).sum()
    return tp/(tp + fn)

def get_tpr_fpr_tnr_fnr(bnn: BNN, x, y, n=100):
    tp, tn, fp, fn = get_confusion_instances(bnn, x, y, n)
    tp, tn, fp, fn = [onp.array(x).astype(float).sum() for x in [tp, tn, fp, fn]]
    tpr = tp/(tp + fn)
    tnr = tn/(tn + fp)
    fpr = 1 - tnr
    fnr = 1 - tpr
    return tpr, fpr, tnr, fnr

class SimpleMLPClassifier():
    class MLP(hk.Module):
        def __init__(self, num_layers: int, hidden_dim: int, dropout: float, verbose=False):
            super(SimpleMLPClassifier.MLP, self).__init__('mlp')
            self.num_layers = num_layers
            self.hidden_dim = hidden_dim
            self.dropout = dropout
            self.verbose=verbose

        def __call__(self, x: chex.Array, is_training: bool = False):
            x_ = x
            for i in range(self.num_layers - 1):
                x_ = hk.Linear(self.hidden_dim)(x_)
                # x_ = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=.999)(x_, is_training)
                x_ = jax.nn.leaky_relu(x_)
                x_ = hk.cond(is_training, lambda a: hk.dropout(hk.next_rng_key(), self.dropout, a), lambda a: a, x_)

            x_ = hk.Linear(2)(x_)
            return x_

    def __init__(self, key:jax.random.PRNGKey, input_shape: Tuple, num_layers: int, hidden_dim: int, dropout: float, lr=0.005, verbose=False):
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.key, split = jax.random.split(key)
        self.init, self.apply = hk.transform(lambda x, t: self.MLP(num_layers, hidden_dim, dropout)(x, t))
        self.key, split = jax.random.split(self.key)
        self.params = self.init(split, jnp.zeros(input_shape), True)
        self.lr = lr
        self.optim = optax.adamw(lr)
        self.opt_state = self.optim.init(self.params)
        self.verbose = verbose

    def get_next_key(self):
        self.key, split = jax.random.split(self.key)
        return split

    def fit(self, train_data, train_labels, num_iterations=10000, batch_size=1000,
            train_data_eval=None, test_data=None, train_labels_eval=None, test_labels=None, extra={}):
        def loss_fn(params, key, x, y, is_training=True):
            logits = self.apply(params, key, x, is_training)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
            return loss.mean()
        loss_value_grad = jax.jit(jax.value_and_grad(loss_fn))

        def train_step_fn(carry, i):
            # TODO Change the carry to include the maximum accuracy and the best parameters.
            key, data, labels, eval_data, eval_labels, m_test_data, m_test_labels, params, opt_state = carry
            key, split = jax.random.split(key)
            # jax.debug.breakpoint()
            batch_idx = jax.random.choice(split, jnp.arange(data.shape[0]), (batch_size, ), replace=False)
            key, split = jax.random.split(key)
            loss, grad = loss_value_grad(params, split, data[batch_idx], labels[batch_idx])
            updates, opt_state = self.optim.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)

            def print_eval(key_loss):
                key, loss = key_loss
                key, split = jax.random.split(key)
                train_syn_preds = jnp.argmax(self.apply(params, split, data, False), axis=1)
                train_syn_acc   = (train_syn_preds == labels).astype(float).mean()

                key, split = jax.random.split(key)
                train_predictions = jnp.argmax(self.apply(params, split, eval_data, False), axis=1)
                train_acc = (train_predictions == eval_labels).astype(float).mean()

                key, split = jax.random.split(key)
                test_predictions = jnp.argmax(self.apply(params, split, m_test_data, False), axis=1)
                test_acc = (test_predictions == m_test_labels).astype(float).mean()

                jax.debug.print('loss: {a}, synth acc: {b}, train acc: {c}, test acc: {d}', a=loss,
                                b=train_syn_acc, c=train_acc, d=test_acc)



            key, split = jax.random.split(key)
            train_predictions = jnp.argmax(self.apply(params, split, eval_data, False), axis=1)
            train_acc = (train_predictions == eval_labels).astype(float).mean()

            key, split = jax.random.split(key)
            jax.lax.cond(jnp.mod(i, 3000) == 0, lambda k: print_eval(k), lambda k: None, (split, loss))
            jax.lax.cond(jnp.mod(i, 3000) == 0, lambda x: jax.debug.print('iteration {x}', x=x), lambda x: None, i)

            return (key, data, labels, eval_data, eval_labels, m_test_data, m_test_labels, params, opt_state), train_acc

        split = self.get_next_key()
        (_, _, _, _, _, _, _, self.params, self.opt_state), acc_hist = jax.lax.scan(train_step_fn,
                                                                                    (split, train_data, train_labels, train_data_eval, train_labels_eval,
                                                                                     test_data, test_labels, self.params, self.opt_state), jnp.arange(num_iterations))
        return acc_hist

    def predict(self, x):
        split = self.get_next_key()
        return jnp.argmax(self.apply(self.params, split, x, False), axis=1)

    def get_auc_roc(self, x: chex.Array, labels: chex.Array) -> float:
        split = self.get_next_key()
        auc_score = auc_roc_softmax_from_classifier(split, self.params, x, labels, self.apply)
        return auc_score

def get_accuracy(key, data, labels, params, pred_fn):
    key, split = jax.random.split(key)
    preds: jnp.ndarray = jnp.argmax(pred_fn(params, split, data, False), axis=1)
    acc = (preds == labels.flatten()).astype(int).mean()
    return acc


def mean_where(x: jnp.ndarray, mask: jnp.ndarray):
    total = jnp.sum(mask)
    sum = (x * mask).sum()
    # jax.debug.print('shape a: {a}', a=x.shape)
    # jax.debug.print('{a}/{b}', a=sum, b=total)
    return sum/total


def get_tpr_pure(key, x, gt_labels, params, pred_fn):
    mask = (gt_labels == 1).astype(jnp.int16).flatten()
    # from the capable, find the ones predicted capable
    key, split = jax.random.split(key)
    predictions = jnp.argmax(pred_fn(params, split, x, False), axis=1) # classifier.predict(x[indices_capable])
    # get number of predictions which are favored
    return mean_where(predictions, mask)
    # num_predicted_favored = (predictions == 1).sum()
    # return num_predicted_favored / len(indices_capable)


def get_tnr_pure(key, x, gt_labels, params, pred_fn):
    # indices_capable = jnp.where(gt_labels == 0)[0]
    mask = (gt_labels == 0).astype(jnp.int16).flatten()
    # from the capable, find the ones predicted capable
    key, split = jax.random.split(key)
    predictions = jnp.argmax(pred_fn(params, split, x, False), axis=1) # classifier.predict(x[indices_capable])
    # jax.debug.print('min: {a}, max: {b}, pred x mask: {c}', a=jnp.min(predictions), b=jnp.max(predictions), c=(predictions*mask).shape)
    # get number of predictions which are favored
    return mean_where((predictions == 0).astype(jnp.int16), mask)
    # num_predicted_favored = (predictions == 0).sum()
    # return num_predicted_favored / len(indices_capable)

def get_tpr(x, gt_labels, classifier):
    indices_capable = jnp.where(gt_labels == 1)[0]
    # from the capable, find the ones predicted capable
    predictions = classifier.predict(x[indices_capable])
    # get number of predictions which are favored
    num_predicted_favored = (predictions == 1).sum()
    return num_predicted_favored / len(indices_capable)


def get_tnr(x, gt_labels, classifier):
    indices_incapable = jnp.where(gt_labels == 0)[0]
    # from the capable, find the ones predicted capable
    predictions = classifier.predict(x[indices_incapable])
    # get number of predictions which are favored
    num_predicted_unfavored = (predictions == 0).sum()
    return num_predicted_unfavored / len(indices_incapable)

class HaikuAdapter():
    key: jax.random.PRNGKey
    def get_next_key(self):
        self.key, split = jax.random.split(self.key)
        return split

    def fit(self, train_data, train_labels, num_iterations: int, batch_size: int):
        return NotImplementedError('fit function not implemented')

class SimpleMLPClassifierBestAcc(SimpleMLPClassifier):
    def fit(self, train_data, train_labels, num_iterations=10000, batch_size=1000,
            train_data_eval=None, test_data=None, train_labels_eval=None, test_labels=None):
        def loss_fn(params, key, x, y, is_training=True):
            logits = self.apply(params, key, x, is_training)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
            return loss.mean()
        loss_value_grad = jax.jit(jax.value_and_grad(loss_fn))

        def train_step_fn(carry, i):
            # TODO Change the carry to include the maximum accuracy and the best parameters.
            key, data, labels, eval_data, eval_labels, m_test_data, m_test_labels, \
                params, opt_state, best_acc, best_params = carry
            key, split = jax.random.split(key)
            batch_idx = jax.random.choice(split, jnp.arange(data.shape[0]), (batch_size, ), replace=False)
            key, split = jax.random.split(key)
            loss, grad = loss_value_grad(params, split, data[batch_idx], labels[batch_idx])
            updates, opt_state = self.optim.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)

            def print_eval(key_loss):
                key, loss = key_loss
                key, split = jax.random.split(key)
                train_syn_preds = jnp.argmax(self.apply(params, split, data, False), axis=1)
                train_syn_acc   = (train_syn_preds == labels).astype(float).mean()

                key, split = jax.random.split(key)
                train_predictions = jnp.argmax(self.apply(params, split, eval_data, False), axis=1)
                train_acc = (train_predictions == eval_labels).astype(float).mean()

                key, split = jax.random.split(key)
                test_predictions = jnp.argmax(self.apply(params, split, m_test_data, False), axis=1)
                test_acc = (test_predictions == m_test_labels).astype(float).mean()

                jax.debug.print('loss: {a}, synth acc: {b}, train acc: {c}, test acc: {d}', a=loss,
                                b=train_syn_acc, c=train_acc, d=test_acc)



            key, split = jax.random.split(key)
            train_predictions = jnp.argmax(self.apply(params, split, eval_data, False), axis=1)
            train_acc = (train_predictions == eval_labels).astype(float).mean()

            key, split = jax.random.split(key)
            jax.lax.cond(jnp.mod(i, 3000) == 0, lambda k: print_eval(k), lambda k: None, (split, loss))
            jax.lax.cond(jnp.mod(i, 3000) == 0, lambda x: jax.debug.print('iteration {x}', x=x), lambda x: None, i)

            m_best_acc = jax.lax.cond(train_acc > best_acc, lambda x: x[1], lambda x: x[0], (best_acc, train_acc))
            m_best_params = jax.lax.cond(train_acc > best_acc, lambda x: x[1], lambda x: x[0], (best_params, params))

            return (key, data, labels, eval_data, eval_labels, m_test_data, m_test_labels, params, opt_state,
                    m_best_acc, m_best_params), train_acc

        split = self.get_next_key()
        (_, _, _, _, _, _, _, self.params, self.opt_state, best_acc, best_params), acc_hist = jax.lax.scan(train_step_fn,
                                                                                                           (split, train_data, train_labels, train_data_eval, train_labels_eval,
                                                                                                            test_data, test_labels, self.params, self.opt_state, 0., self.params), jnp.arange(num_iterations))
        return acc_hist, best_acc, best_params

class SimpleMLPClassifierWithClassDistributionRandomization(SimpleMLPClassifier):
    def fit(self, train_data, train_labels, num_iterations=10000, batch_size=1000,
            train_data_eval=None, test_data=None, train_labels_eval=None, test_labels=None, verbose=False):
        self._verbose = verbose
        fav_mask = (train_labels == 1).astype(float).flatten()
        unfav_mask = (train_labels == 0).astype(float).flatten()
        num_fav, num_unfav = fav_mask.sum(), unfav_mask.sum()
        fav_idxs = jnp.where(train_labels == 1)[0]
        unfav_idxs = jnp.where(train_labels == 0)[0]
        def loss_fn(params, key, x, y, is_training=True):
            logits = self.apply(params, key, x, is_training)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
            return loss.mean()
        loss_value_grad = jax.jit(jax.value_and_grad(loss_fn))

        def train_step_fn(carry, i):
            # TODO Change the carry to include the maximum accuracy and the best parameters.
            key, data, labels, eval_data, eval_labels, m_test_data, m_test_labels, params, opt_state = carry
            key, split = jax.random.split(key)
            # Calculate the percentage of favorable labels
            perc_fav = jax.random.uniform(split)
            # perc_unfav = 1 - perc_fav

            m_fav_mask = fav_mask * perc_fav
            m_unfav_mask = unfav_mask * (1 - perc_fav)
            selection_probs = (m_fav_mask/num_fav) + (m_unfav_mask/num_unfav)

            key, split = jax.random.split(key)
            batch_idx = jax.random.choice(split, jnp.arange(data.shape[0]), (batch_size, ), replace=False, p=selection_probs)
            key, split = jax.random.split(key)
            loss, grad = loss_value_grad(params, split, data[batch_idx], labels[batch_idx])
            updates, opt_state = self.optim.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)

            def print_eval(key_loss):
                if not self._verbose:
                    return
                key, loss = key_loss
                key, split = jax.random.split(key)
                train_syn_preds = jnp.argmax(self.apply(params, split, data, False), axis=1)
                train_syn_acc   = (train_syn_preds == labels).astype(float).mean()

                key, split = jax.random.split(key)
                train_predictions = jnp.argmax(self.apply(params, split, eval_data, False), axis=1)
                train_acc = (train_predictions == eval_labels).astype(float).mean()

                key, split = jax.random.split(key)
                test_predictions = jnp.argmax(self.apply(params, split, m_test_data, False), axis=1)
                test_acc = (test_predictions == m_test_labels).astype(float).mean()

                jax.debug.print('loss: {a}, synth acc: {b}, train acc: {c}, test acc: {d}', a=loss,
                                b=train_syn_acc, c=train_acc, d=test_acc)



            key, split = jax.random.split(key)
            train_predictions = jnp.argmax(self.apply(params, split, eval_data, False), axis=1)
            train_acc = (train_predictions == eval_labels).astype(float).mean()

            key, split = jax.random.split(key)
            jax.lax.cond(jnp.mod(i, 3000) == 0, lambda k: print_eval(k), lambda k: None, (split, loss))
            # jax.lax.cond(jnp.mod(i, 3000) == 0, lambda x: jax.debug.print('iteration {x}', x=x), lambda x: None, i)

            return (key, data, labels, eval_data, eval_labels, m_test_data, m_test_labels, params, opt_state), train_acc

        split = self.get_next_key()
        (_, _, _, _, _, _, _, self.params, self.opt_state), acc_hist = jax.lax.scan(train_step_fn,
                                                                                    (split, train_data, train_labels, train_data_eval, train_labels_eval,
                                                                                     test_data, test_labels, self.params, self.opt_state), jnp.arange(num_iterations))
        return acc_hist

