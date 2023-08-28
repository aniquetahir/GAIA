import random

import jax
from jax.experimental.host_callback import call as hcb
import numpy as np
import numpy as onp
import functools
from jax import jit, vmap
from anique.jax.common import restructure, destructure, get_destructure_ranges
from anique.jax.models.bnn import BNN, get_weight_sampler, get_nd_scale_mixture
import jax.numpy as jnp
from oryx.core.ppl import random_variable, log_prob
from tensorflow_probability.substrates import jax as tfp
import optax
import chex
from typing import Dict, List
import haiku as hk
from copy import deepcopy
from tqdm import tqdm
import timeit
from anique.jax.models.image import infinite_dataloader

tfd = tfp.distributions
ksplit = jax.random.split



class ImageBNN(BNN):
    def __init__(self, key, input_shape, num_classes,
                 mu_init_range=(-.2, .2), rho_init_range=(-5, -4),
                 pi=0.5, sigma1=.5, sigma2=0.0005, learning_rate=0.0001, name=None, verbose=False):
        self._key = key
        self._verbose = verbose

        self._num_classes = num_classes
        self._lr = learning_rate
        self._is_training = True

        # Define a resnet who's weights are going to be drawn from a distribution
        resnet = hk.without_apply_rng(hk.transform_with_state(
            lambda x: hk.nets.ResNet18(num_classes, resnet_v2=True)(x, is_training=True)
        ))
        resnet_params, resnet_state = resnet.init(self.next_key(), jnp.zeros(input_shape))
        self._resnet = resnet
        self._resnet_params = resnet_params
        self._resnet_state = resnet_state

        # Initialize $\mu$ and $\rho$ for the weights
        dest_params = destructure(resnet_params, jax.tree_util.tree_structure(resnet_params))
        mus = jax.random.uniform(self.next_key(), dest_params.shape, minval=mu_init_range[0], maxval=mu_init_range[1])
        rhos = jax.random.uniform(self.next_key(), dest_params.shape, minval=rho_init_range[0], maxval=rho_init_range[1])

        baysian_params = {'mus': mus, 'rhos': rhos}
        self._baysian_params = baysian_params

        # Define the opimizer
        optim = optax.adam(learning_rate)
        opt_state = optim.init(baysian_params)

        self._optim = optim
        self._opt_state = opt_state
        ref_resnet_params = deepcopy(resnet_params)

        self._destructure_ranges = get_destructure_ranges(resnet_params)

        def sample_weights(key: chex.PRNGKey, mu_tensor: chex.Array, rho_tensor: chex.Array) -> chex.Array:
            mu = mu_tensor.flatten()
            rho = rho_tensor.flatten()
            a = random_variable(tfd.MultivariateNormalDiag(loc=mu, scale_diag=jnp.log1p(jnp.exp(rho))))(key)
            return a.reshape(mu_tensor.shape)

        def sampler(key, baysian_params):
            sampled_weights = sample_weights(key, baysian_params['mus'], baysian_params['rhos'])
            restructured_sample = restructure(ref_resnet_params, sampled_weights, self._destructure_ranges,
                                              jax.tree_util.tree_structure(ref_resnet_params))
            return restructured_sample

        self._weight_sampler = sampler
        self._prior = get_nd_scale_mixture(pi, sigma1, sigma2, dest_params.shape[0])

        self.set_training(self._is_training)



    def set_training(self, is_training=True):
        self._is_training = is_training
        resnet = hk.without_apply_rng(hk.transform_with_state(
            lambda x: hk.nets.ResNet18(self._num_classes, resnet_v2=True)(x, is_training=is_training)
        ))
        self._resnet = resnet

        @jit
        def get_logits(key, state, d_x, bays_params):
            params = self._weight_sampler(key, bays_params)
            logits, new_state = self._resnet.apply(params, state, d_x)
            return logits, new_state

        @jit
        def predict_w_sample(key, bay_params, state,  x):
            params = self._weight_sampler(key, bay_params)
            logits, state = self._resnet.apply(params, state, x)
            return logits, state, params

        def get_stacked_predictions(key, bays_params, resnet_state,  data, n):
            splits = jax.random.split(key, n)
            logits = vmap(lambda k, s, x, b: get_logits(k, s, x, b)[0],
                          in_axes=(0, None, None, None))(splits, resnet_state, data, bays_params)
            logits = jnp.transpose(jax.nn.softmax(logits), (1, 2, 0)) # TODO Verify if this still holds
            return logits

        def get_mean_predictions(key, bays_params, resnet_state, data, n):
            def total_predictor(carry, i):
                params, key, x, total = carry
                key, split = ksplit(key)
                logits, _ = get_logits(split, resnet_state, x, params)
                logit_probs = jax.nn.softmax(logits)
                total = total + logit_probs
                return (params, key, x, total), 0.

            (_, key, _, prob_sum), _ = jax.lax.scan(total_predictor,
                                                    (bays_params, key, data, jnp.zeros((data.shape[0], self._num_classes))),
                                                    jnp.arange(n))
            return prob_sum/n

        self._mean_pred_fn = jit(get_mean_predictions, static_argnames=('n', ))
        self.get_stacked_predictions = jit(get_stacked_predictions, static_argnames=('n', ))
        self._predict_w_sample = predict_w_sample


    def get_ensemble_predictions(self, x: chex.Array, n=30):
        return self.get_stacked_predictions(self.next_key(), self._baysian_params, self._resnet_state, x, n)

    def get_mean_predictions(self, x: chex.Array, n=30):
        return self._mean_pred_fn(self.next_key(), self._baysian_params, self._resnet_state, x, n)

    def get_loss_fn(self, NUM_MINIBATCHES):
        # NUM_MINIBATCHES = NUM_MINIBATCHES
        try:
            log_pi_denom = onp.log((2. ** NUM_MINIBATCHES) - 1)
        except OverflowError:
            log_pi_denom = NUM_MINIBATCHES

        @functools.partial(jit, static_argnames=['apply_fn', 'prior'])
        def loss_fn(b_params, resnet_state, key, data, labels, apply_fn, prior, i):
            mu, rho = b_params['mus'], b_params['rhos']
            logits, new_resnet_state, params = apply_fn(key, b_params, resnet_state, data)

            # Regularization
            ## Make the means closer to zero
            l2_reg = jnp.sqrt((mu ** 2).mean())

            params = destructure(params, jax.tree_util.tree_structure(params))

            log_prior = log_prob(prior)(params).mean()
            log_variational_post = log_prob(get_weight_sampler(mu, rho))(params)
            nll = optax.softmax_cross_entropy_with_integer_labels(logits, labels.astype(int)).mean()
            # nll_mus = optax.softmax_cross_entropy_with_integer_labels(logits_mus, labels).mean()

            # Some things are preprocessed outside jit because of large number calculations.
            pi = jnp.exp(((NUM_MINIBATCHES - i) * jnp.log(2)) - log_pi_denom)


            loss = ((log_variational_post - log_prior) * pi) + nll + 0.001 # * l2_reg

            return loss, new_resnet_state
        return loss_fn


    def sample_random_batch(self, key, data_iter, batch_size):
        data_all = ([], [], [])
        for i in range(10):
            x, y, a = next(data_iter)
            data_all[0].append(x.numpy())
            data_all[1].append(y.numpy())
            data_all[2].append(a.numpy())

        data_all = (onp.vstack(data_all[0]), onp.concatenate(data_all[1]), onp.concatenate(data_all[2]),)

        data, labels, prot_attr_mask = data_all

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

    def train_loop(self, num_iterations, data_iter, valid_iter, update_step, loss_fn, extra):
        valid_inf_iter = infinite_dataloader(valid_iter)
        data_inf_iter = infinite_dataloader(data_iter)

        print_freq = num_iterations // 100
        batch_size = extra['batch_size']
        NUM_MINIBATCHES = extra['num_minibatches']
        NUM_CAN_HANDLE = 10


        @jax.jit
        def update_step(i, key, batch_data, batch_labels, params, resnet_state, opt_state):
            (elbo, new_resnet_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, resnet_state, key, batch_data, batch_labels, self._predict_w_sample, self._prior,
                                                                                        (i % NUM_MINIBATCHES) + 1)
            updates, opt_state = self._optim.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            return params, new_resnet_state, opt_state, elbo

        def print_validation(print_args):
            i, elbo, acc = print_args

            if i % print_freq == 0 and self._verbose:
                print(f'iter: {i}, elbo: {elbo}, valid_acc: {acc}')


        def calc_valid_acc(key_params):
            key, params, resnet_state = key_params
            params_resnet = restructure(self._resnet_params, params['mus'], self._destructure_ranges, jax.tree_util.tree_structure(self._resnet_params))

            x_valid, y_valid, a_valid = next(valid_inf_iter)
            x_valid, y_valid, a_valid = x_valid.numpy(), y_valid.numpy().astype(int), a_valid.numpy()

            x_train, y_train, a_train = next(data_inf_iter)
            x_train, y_train, a_train = x_train.numpy(), y_train.numpy().astype(int), a_train.numpy()

            logits, _ = self._resnet.apply(params_resnet, resnet_state, x_valid)
            train_logits, _ = self._resnet.apply(params_resnet, resnet_state, x_train)

            # logits, _ = self.predict_w_sample(key, params, valid_iter[0][:valid_limit][:, 1:])
            valid_acc = (jnp.argmax(logits, axis=1) == y_valid).astype(jnp.float32).mean()
            train_acc = (jnp.argmax(train_logits, axis=1) == y_train).astype(jnp.float32).mean()

            if self._verbose:
                jax.debug.print('valid acc:{a}, train acc: {b}', a=valid_acc, b=train_acc)
            return valid_acc, train_acc


        def calc_train_acc(key_params):
            key, params, resnet_state = key_params
            # jax.debug.print('{a}', a=key)
            valid_limit = 10000
            params_resnet = restructure(self._resnet_params, params['mus'], self._destructure_ranges, jax.tree_util.tree_structure(self._resnet_params))

            train_x, train_y, train_a = next(data_inf_iter)
            train_x, train_y, train_a = train_x.numpy(), train_y.numpy().astype(int), train_a.numpy().astype(int)
            train_x, train_y, train_a = jnp.array(train_x), jnp.array(train_y), jnp.array(train_a)

            train_logits, _ = self._resnet.apply(params_resnet, resnet_state, train_x)
            # logits, _ = self.predict_w_sample(key, params, valid_iter[0][:valid_limit][:, 1:])
            # valid_acc = (jnp.argmax(logits, axis=1) == valid_iter[1][:valid_limit]).astype(jnp.float32).mean()
            train_acc = (jnp.argmax(train_logits, axis=1) == train_y).astype(jnp.float32).mean()

            return train_acc


        @functools.partial(jax.jit, static_argnames=('num_confirmations'))
        def get_train_probs(key, params, resnet_state, num_confirmations = 50):
            train_x, train_y, train_a = next(data_inf_iter)
            train_x, train_y, train_a = [jnp.array(x.numpy()) for x in (train_x, train_y, train_a, )]

            average_probs = jnp.zeros((train_x.shape[0], self._num_classes))
            for i in range(num_confirmations):
                key, split = ksplit(key)
                logits, _, _ = self._predict_w_sample(split, params, resnet_state, train_x)
                logit_probs = jax.nn.softmax(logits)
                average_probs += logit_probs
            return average_probs/num_confirmations

        epoch_train_probs = []
        def add_train_probs(probs):
            epoch_train_probs.append(onp.array(probs))

        sample_x, sample_y = self.sample_random_batch(self.next_key(), data_iter, batch_size)
        shape_data_x, shape_data_y = sample_x.shape, sample_y.shape
        def host_get_batch(a) -> Dict[str, np.ndarray]:
            data = self.sample_random_batch(self.next_key(), data_iter, batch_size)
            return {
                'x': data[0],
                'y': data[1]
            }


        @functools.partial(jax.jit, backend='gpu')
        def loop_fn_hcb(carry, i):
            key, params, resnet_state, opt_state, best_model, last_model_stats = carry
            # jax.debug.breakpoint()
            # get data using host callback
            data = hcb(host_get_batch, None, result_shape={'x': np.ones(shape_data_x), 'y': np.ones(shape_data_y)})

            best_stats, best_params, best_resnet_state = best_model
            key, i_key = ksplit(key)
            key, split = ksplit(key)
            x, y = data['x'], data['y']
            # update the params and optimizer state
            params, resnet_state, opt_state, elbo = update_step(i, i_key, x, y, params, resnet_state, opt_state)

            key, split = ksplit(key)
            valid_acc, train_acc = jax.lax.cond(i % print_freq == 0, calc_valid_acc, lambda k: (0., 0.), (split, params, resnet_state))
            key, split = ksplit(key)
            train_acc = calc_train_acc((split, params, resnet_state))
            key, split = ksplit(key)
            # jax.debug.print('loop {a}', a = i)
            jax.debug.callback(print_validation, (i, elbo, valid_acc))
            key, split = ksplit(key)

            train_probs = jax.lax.cond(i % len(y) == 0, # WARNING. This i already has mod applied
                                       lambda k: get_train_probs(k[0], k[1], k[2]),
                                       lambda k: jnp.zeros((len(x), self._num_classes)),
                                       (split, params, resnet_state))


            curr_stats = train_acc

            # Average out the current stats using the previous stats
            curr_stats = (curr_stats + last_model_stats) / 2

            # if current stats > best stats
            # operands = curr_stats, curr_params, best_stats, best_params
            best_stats, best_params, best_resnet_state = jax.lax.cond(
                curr_stats > best_stats,
                # if true
                lambda x: (x[0], x[1], x[2]),
                lambda x: (x[3], x[4], x[5]),
                (curr_stats, params, resnet_state, best_stats, best_params, best_resnet_state)
            )

            ## Update the best stats
            return (key, params, resnet_state, opt_state,
                    (best_stats, best_params, best_resnet_state), curr_stats) , curr_stats
        @functools.partial(jax.jit, backend='gpu')
        def loop_fn(carry, i):
            key, params, resnet_state, opt_state, data, best_model, last_model_stats = carry
            best_stats, best_params, best_resnet_state = best_model
            key, i_key = ksplit(key)
            key, split = ksplit(key)

            x, y = data[0][i], data[1][i]

            # update the params and optimizer state
            params, resnet_state, opt_state, elbo = update_step(i, i_key, x, y, params, resnet_state, opt_state)

            key, split = ksplit(key)
            valid_acc, train_acc = jax.lax.cond(i % print_freq == 0, calc_valid_acc, lambda k: (0., 0.), (split, params, resnet_state))
            key, split = ksplit(key)
            train_acc = calc_train_acc((split, params, resnet_state))
            key, split = ksplit(key)
            # jax.debug.print('loop {a}', a = i)
            jax.debug.callback(print_validation, (i, elbo, valid_acc))
            key, split = ksplit(key)
            train_probs = jax.lax.cond(i % len(y) == 0, # WARNING. This i already has mod applied
                                       lambda k: get_train_probs(k[0], k[1], k[2]),
                                       lambda k: jnp.zeros((len(x), self._num_classes)),
                                       (split, params, resnet_state))
            jax.lax.cond(i % len(y) == 0,
                         lambda k: jax.debug.callback(add_train_probs, k),
                         lambda k: [], train_probs)

            curr_stats = train_acc
            # Average out the current stats using the previous stats
            curr_stats = (curr_stats + last_model_stats) / 2

            # if current stats > best stats
            # operands = curr_stats, curr_params, best_stats, best_params
            best_stats, best_params, best_resnet_state = jax.lax.cond(
                curr_stats > best_stats,
                # if true
                lambda x: (x[0], x[1], x[2]),
                lambda x: (x[3], x[4], x[5]),
                (curr_stats, params, resnet_state, best_stats, best_params, best_resnet_state)
            )
            ## Update the best stats
            return (key, params, resnet_state, opt_state, data,
                    (best_stats, best_params, best_resnet_state), curr_stats) , i

        best_stats = 0.
        best_params = self._baysian_params
        best_resnet_state = self._resnet_state
        curr_stats = 0.

        key = self.next_key()
        params = self._baysian_params
        resnet_state = self._resnet_state
        opt_state = self._opt_state


        (_, params, resnet_state, opt_state, (best_stats, best_params, best_resnet_state), curr_stats), _ = \
            jax.lax.scan(
                loop_fn_hcb,
                (self.next_key(), params, resnet_state, opt_state, (best_stats, best_params, best_resnet_state), curr_stats),
                jnp.arange(num_iterations)
            )

        self._baysian_params = best_params
        self._resnet_state = best_resnet_state

        return {}
