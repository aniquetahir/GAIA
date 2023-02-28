import jax
import jax.random as random
import jax.numpy as jnp
import numpy as onp
import haiku as hk
import optax
# import tensorflow_datasets as tfds
import chex
from copy import deepcopy
from jax import jit, vmap
import oryx
from oryx.core.ppl import random_variable
from oryx.core.ppl import log_prob
from oryx.core.ppl import joint_sample
from oryx.core.ppl import joint_log_prob
from oryx.core.ppl import block
from oryx.core.ppl import intervene
from oryx.core.ppl import conditional
from oryx.core.ppl import graph_replace
from oryx.core.ppl import nest
import functools
from tensorflow_probability.substrates import jax as tfp
from tqdm import tqdm

from utils.jax.common import destructure, get_destructure_ranges, restructure

tfd = tfp.distributions


def get_gaussian_scale_mixture(pi, sigma1, sigma2):
    assert (sigma1 > sigma2)

    def additive_gaussian(key):
        mix = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=[pi, 1 - pi]),
            components_distribution=tfd.Normal(
                loc=[0., 0.],
                scale=[sigma1, sigma2]
            )
        )
        a = random_variable(mix)(key)
        return a

    return additive_gaussian


def get_nd_scale_mixture_dist(pi, sigma1, sigma2, n):
    return tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=[pi, 1 - pi]),
        components_distribution=tfd.MultivariateNormalDiag(
            loc=[jnp.zeros((n,)), jnp.zeros((n,))],
            scale_diag=[jnp.ones((n,)) * sigma1, jnp.ones((n,)) * sigma2]
        )
    )


def get_nd_scale_mixture(pi, sigma1, sigma2, n):
    assert (sigma1 > sigma2)

    def additive_gaussian(key):
        mix = get_nd_scale_mixture_dist(pi, sigma1, sigma2, n)
        x = random_variable(mix)(key)
        return x

    return additive_gaussian


def sample_weights(key: chex.PRNGKey, mu_tensor: chex.Array, rho_tensor: chex.Array) -> chex.Array:
    mu = mu_tensor.flatten()
    rho = rho_tensor.flatten()
    a = random_variable(tfd.MultivariateNormalDiag(loc=mu, scale_diag=jnp.log1p(jnp.exp(rho))))(key)
    return a.reshape(mu_tensor.shape)


def get_weight_sampler(mu_tensor: chex.Array, rho_tensor: chex.Array):
    mu = mu_tensor.flatten()
    rho = rho_tensor.flatten()

    def sampler(key: chex.PRNGKey):
        a = random_variable(tfd.MultivariateNormalDiag(loc=mu, scale_diag=jnp.log1p(jnp.exp(rho))))(key)
        return a.reshape(mu_tensor.shape)

    return sampler


class MyMLP(hk.Module):
    def __init__(self, num_layers, hidden_dim, num_classes, name=None):
        super(MyMLP, self).__init__(name=name)
        self._num_layers = num_layers
        self._hidden_dim = hidden_dim
        self._num_classes = num_classes

    def __call__(self, batch):
        x = hk.Flatten()(batch)
        for i in range(self._num_layers - 1):
            x = hk.Linear(self._hidden_dim)(x)
            x = jax.nn.relu(x)
        x = hk.Linear(self._num_classes)(x)
        return x



class BNN:
    def __init__(self, key, input_shape, num_layers, hidden_dim, num_classes,
                 mu_init_range=(-0.2, 0.2), rho_init_range=(-5, -4),
                 pi=0.5, sigma1=1., sigma2=0.002, learning_rate=0.001,
                 name=None, verbose=False):
        self._key = key
        self._verbose = verbose

        self._num_layers = num_layers
        self._hidden_dim = hidden_dim
        self._num_classes = num_classes

        self._lr = learning_rate

        # Define an MLP who's weights are going to be drawn from a distribution
        layers = jnp.repeat(jnp.array([hidden_dim]), num_layers - 1)
        layers = jnp.hstack((layers, num_classes))

        mlp = hk.without_apply_rng(hk.transform(lambda x: MyMLP(num_layers, hidden_dim, num_classes)(x)))
        mlp_params = mlp.init(self.next_key(), jnp.zeros(input_shape))
        self._mlp = mlp
        self._mlp_params = mlp_params

        # Initialize \mu and \rho for the weights
        dest_params = destructure(mlp_params, jax.tree_util.tree_structure(mlp_params))
        mus = random.uniform(self.next_key(), dest_params.shape, minval=mu_init_range[0], maxval=mu_init_range[1])
        rhos = random.uniform(self.next_key(), dest_params.shape, minval=rho_init_range[0], maxval=rho_init_range[1])

        baysian_params = {'mus': mus, 'rhos': rhos}
        self._baysian_params = baysian_params

        # Define the optimizer
        optim = optax.adam(learning_rate)
        opt_state = optim.init(baysian_params)

        self._optim = optim
        self._opt_state = opt_state
        ref_mlp_params = deepcopy(mlp_params)

        def sample_weights(key: chex.PRNGKey, mu_tensor: chex.Array, rho_tensor: chex.Array) -> chex.Array:
            mu = mu_tensor.flatten()
            rho = rho_tensor.flatten()
            a = random_variable(tfd.MultivariateNormalDiag(loc=mu, scale_diag=jnp.log1p(jnp.exp(rho))))(key)
            return a.reshape(mu_tensor.shape)

        def sampler(key, baysian_params):
            sampled_weights = sample_weights(key, baysian_params['mus'], baysian_params['rhos'])
            restructured_sample = restructure(ref_mlp_params, sampled_weights, self._destructure_ranges,
                                              jax.tree_util.tree_structure(ref_mlp_params))
            return restructured_sample

        @jit
        def get_logits(key, d_x, bays_params):
            params = self.weight_sampler(key, bays_params)
            logits = self._mlp.apply(params, d_x)
            return logits


        self.weight_sampler = sampler
        self._destructure_ranges = get_destructure_ranges(mlp_params)

        # Define the prior distributions
        self.prior = get_nd_scale_mixture(pi, sigma1, sigma2, dest_params.shape[0])

        # @jit
        def predict_w_sample(key, bay_params, x):
            params = self.weight_sampler(key, bay_params)
            logits = self._mlp.apply(params, x)
            return logits, params

        self.predict_w_sample = predict_w_sample
        def get_stacked_predictions(key, bays_params, data, n):
            splits = jax.random.split(key, n)
            logits = vmap(get_logits, in_axes=(0, None, None))(splits, data, bays_params)
            logits = jnp.transpose(jax.nn.softmax(logits), (1, 2, 0))
            return logits

        # self.get_stacked_predictions = get_stacked_predictions

        def get_mean_predictions(key, bays_params, data, n):
            def total_predictor(carry, i):
                params, key, x, total = carry
                key, split = jax.random.split(key)
                logits = get_logits(split, x, params)
                logit_probs = jax.nn.softmax(logits)
                total = total + logit_probs
                return (params, key, x, total), 0.

            (_, key, _, prob_sum), _ = jax.lax.scan(total_predictor,
                                                 (bays_params, key, data, jnp.zeros((data.shape[0], num_classes))),
                                                 jnp.arange(n))
            return prob_sum/n
        self.mean_pred_fn = jit(get_mean_predictions, static_argnames=('n',))
        self.get_stacked_predictions = jit(get_stacked_predictions, static_argnames=('n',))

    def next_key(self):
        self._key, split = random.split(self._key)
        return split

    def train_loop(self, num_iterations, data_iter, valid_iter, update_step, loss_fn, extra):
        for i in tqdm(range(num_iterations)):
            i_key = self.next_key()
            data, labels = next(data_iter)
            self._baysian_params, self._opt_state = update_step(i, i_key, data, labels, self._baysian_params,
                                                                self._opt_state)
            if i % (num_iterations//10) == 0:
                # print the value of the loss
                elbo = loss_fn(self._baysian_params, i_key, data, labels, self.predict_w_sample, self.prior, i)
                if self._verbose:
                    print(f'ELBO: {elbo}')

                # print the value of the validation accuracy
                valid_batch, valid_labels = next(valid_iter)
                logits, _ = self.predict_w_sample(self.next_key(), self._baysian_params, valid_batch)
                acc = (jnp.argmax(logits, axis=1) == valid_labels).astype(jnp.float32).mean()
                if self._verbose:
                    print(f'Valid Acc: {acc}')
        return {}

    def get_loss_fn(self, NUM_MINIBATCHES):
        log_pi_denom = onp.log((2. ** NUM_MINIBATCHES) - 1)
        @functools.partial(jit, static_argnames=['apply_fn', 'prior'])
        def loss_fn(b_params, key, data, labels, apply_fn, prior, i):
            mu, rho = b_params['mus'], b_params['rhos']
            logits, params = apply_fn(key, b_params, data)
            params = destructure(params, jax.tree_util.tree_structure(params))

            log_prior = log_prob(prior)(params).mean()
            log_variational_post = log_prob(get_weight_sampler(mu, rho))(params)
            nll = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

            # Some things are preprocessed outside jit because of large number calculations.
            pi = jnp.exp(((NUM_MINIBATCHES - i) * jnp.log(2)) - log_pi_denom)
            # pi = (2. ** (pi_param - i))/((2. ** pi_param) -1 )
            # jax.debug.print('{a}', a=pi)
            loss = ((log_variational_post - log_prior) * pi) + nll
            return loss
        return loss_fn

    def train(self, data_iterator, valid_iterator, NUM_MINIBATCHES, num_iterations=10000, extra={}):
        loss_fn = self.get_loss_fn(NUM_MINIBATCHES)


        @jit
        def update_step(i, key, batch_data, batch_labels, params, opt_state):
            grads = jax.grad(loss_fn)(params, key, batch_data, batch_labels, self.predict_w_sample, self.prior,
                                      (i % NUM_MINIBATCHES) + 1)
            updates, opt_state = self._optim.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            return params, opt_state

        return self.train_loop(num_iterations, data_iterator, valid_iterator, update_step, loss_fn, extra)


    def get_ensemble_predictions(self, x: chex.Array, n=30):
        return self.get_stacked_predictions(self.next_key(), self._baysian_params, x, n)

    def get_mean_predictions(self, x: chex.Array, n=30):
        return self.mean_pred_fn(self.next_key(), self._baysian_params, x, n)

    def get_epistemic_uncertainty(self, x, n=30):
        """
        The epistemic uncertainty is the Variance over the distribution of the predictions for the randomly
        sampled models.
        :param x:
        :param n:
        :return:
        """
        # Get stacked predictions
        ensemble_preds = self.get_ensemble_predictions(x, n)

        # Find the variance of the predictions over each dimension of the logits
        mean_variances = jnp.var(ensemble_preds, axis=-1).mean(axis=-1)
        return mean_variances

    def get_total_uncertainty(self, x, n=30):
        ensemble_preds = self.get_ensemble_predictions(x, n)
        mean_preds = jnp.mean(ensemble_preds, axis=-1)
        total_uncertainty = jax.scipy.special.entr(mean_preds).mean(axis=-1)
        return total_uncertainty

    def get_aleatoric_uncertainty_v2(self, x, n=30):
        ensemble_preds = self.get_ensemble_predictions(x, n)
        entropies = jax.scipy.special.entr(ensemble_preds)
        per_sample_mean_entropies = entropies.mean(axis=(1,2,))
        return per_sample_mean_entropies
    def get_aleatoric_uncertainty(self, x, n=30):
        # Get the total uncertainty (Entropy of the mean prediction)
        total_uncertainty = self.get_total_uncertainty(x, n)

        # Get the Epistemic uncertainty
        epistemic_uncertainty = self.get_epistemic_uncertainty(x, n)

        # Aleatoric = Total - Epistemic
        aleatoric_uncertainty = total_uncertainty - epistemic_uncertainty
        return aleatoric_uncertainty


    def __call__(self, x):
        params = self.weight_sampler(self.next_key(), self._baysian_params)
        logits = self._mlp.apply(params, x)
        return logits

