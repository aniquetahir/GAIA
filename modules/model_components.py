from bnn_utils import SimpleMLPClassifier
import jax, jax.numpy as jnp
import numpy as onp
from jax import jit
from functools import partial
import optax
from jaxlib.xla_extension import DeviceArray
import haiku as hk
from utils.jax.common import fake_scan
from sklearn.preprocessing import StandardScaler
from aif360.datasets import BinaryLabelDataset

ksplit = jax.random.split





def incompetent_get(fn):
    def hyphen_default(*args, **kwargs):
        try:
            result = fn(*args, **kwargs)
            if type(result) is DeviceArray:
                result = result.item()
            return result
        except Exception as e:
            print(e)
            return '-'
    return hyphen_default
class WeightedMLP(SimpleMLPClassifier):
    def __init__(self, key, input_shape, num_layers, hidden_dim, dropout, lr=0.005, verbose=False):
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.key, split = jax.random.split(key)
        self.init, self.apply = hk.transform(lambda x, t: self.MLP(num_layers, hidden_dim, dropout)(x, t))
        self.key, split = jax.random.split(self.key)
        self.params = self.init(split, jnp.zeros(input_shape), True)
        self.lr = lr

        self.scaler = StandardScaler()

        # scheduler = optax.exponential_decay(lr, 10000, 0.999)
        # self.optim = optax.chain(
        #     optax.clip_by_global_norm(1.),
        #     optax.scale_by_adam(),
        #     optax.scale_by_schedule(scheduler),
        #     optax.scale(-1.)
        # )
        self.optim = optax.adamw(lr)

        self.opt_state = self.optim.init(self.params)
        meta_scheduler = optax.exponential_decay(0.002, 10000, 0.999)
        self.meta_optim = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.scale_by_rms(),
            optax.scale_by_schedule(meta_scheduler),
            optax.scale(-1.0)
        )
        self.meta_params = {'min': 0., 'max': 1.}
        self.meta_opt_state = self.meta_optim.init(self.meta_params)
        self.verbose = verbose
        self.prot_attr_idx = None


    def get_beta(self, params, unc_val):
        unc_min = params['min']
        unc_max = params['max']
        return ((unc_val - unc_min) / (unc_max - unc_min)) ** 2.25

    def loss_ce(self, params, key, x, y, weights, is_training=True):
        logits = self.apply(params, key, x, is_training)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y) * weights
        return loss.mean()

    def loss_fair(self, params, key, x, y, weights, prot_attrs, is_training=True):
        logits = self.apply(params, key, x, is_training)
        loss_ce_all = optax.softmax_cross_entropy_with_integer_labels(logits, y) * weights
        loss_a_1 = (loss_ce_all * prot_attrs).sum()/prot_attrs.sum()
        loss_a_2 = (loss_ce_all * (1 - prot_attrs)).sum()/(1-prot_attrs).sum()
        return jnp.abs(loss_a_2 - loss_a_1)

    def total_loss(self, params, key, x, y, prot_attrs, beta, is_training=True):
        # NOTE: beta is higher for the samples which have high uncertainty
        return self.loss_fair(params, key, jax.lax.stop_gradient(x), y, beta, prot_attrs, is_training) + \
               self.loss_ce(params, key, x, y, (1-beta), is_training)

    def total_loss_meta(self, meta_params, params, key, x, y, prot_attrs, uncert, is_training=True):
        betas = self.get_beta(meta_params, uncert)

    def outer_loss(self, eta, key, theta, state, samples, labels, prot_mask, batch_size, uncerts):
        # Get the betas from eta
        betas = self.get_beta(eta, uncerts)
        # TODO enable/disable for experimentation
        # betas = betas * eta['rw']

        key, split = ksplit(key)
        batch_idx = jax.random.choice(split, jnp.arange(samples.shape[0]), (batch_size, ), replace=False)

        # Do the inner step to get the new theta and state
        loss_value_grad = jax.value_and_grad(self.total_loss)
        loss, grad = loss_value_grad(theta, split, samples[batch_idx], labels[batch_idx],
                                     prot_mask[batch_idx], betas[batch_idx])
        updates, state = self.optim.update(grad, state, theta)
        theta = optax.apply_updates(theta, updates)

        # get the loss = fairness loss + cross entropy loss
        key, split = ksplit(key)
        loss_outer = self.total_loss(theta, split, samples, labels, prot_mask, 0.5) + jnp.sqrt((eta['min'] - 0) ** 2 + (eta['max'] - 1) ** 2)
        # jax.debug.print('outer loss: {a}', a=loss_outer)

        # return loss, (theta, state)
        return loss_outer, (theta, state, loss)

    def outer_step(self, carry, i, batch_size):
        key, eta, meta_state, uncerts, samples, labels, prot_mask, _, _, _, _, theta, _, _, state = carry
        key, split = ksplit(key)
        # get the grad of the outer loss
        outer_loss, (theta, state, inner_loss) = self.outer_loss(eta, split, theta, state, samples, labels, prot_mask, batch_size, uncerts)
        ## grad, (theta, state, inner_loss) = jax.grad(self.outer_loss,has_aux=True)(eta, split, theta, state, samples, labels, prot_mask, batch_size, uncerts)
        # jax.lax.cond(i % 100 == 0, lambda x: jax.debug.print('grad: {a}', a=x), lambda x: None, grad)
        # update the meta optim
        ## meta_updates, meta_state = self.meta_optim.update(grad, meta_state)
        # get the new eta
        ## eta = optax.apply_updates(eta, meta_updates)
        # jax.lax.cond(i % 1000 == 0, lambda x: jax.debug.print('eta: {a}', a=x), lambda x: None, eta)
        # return eta, theta, meta state, state
        return eta, theta, meta_state, state, inner_loss


    def train_step_fn(self, carry, i, batch_size):
        # TODO Change the carry to include the maximum accuracy and the best parameters.
        key, meta_params, meta_state, uncert, data, labels, prot_mask, eval_data, eval_labels, m_test_data, m_test_labels, params, \
            best_model, last_loss, opt_state = carry

        best_loss, best_params = best_model
        key, split = jax.random.split(key)


        meta_params, params, meta_state, opt_state, inner_loss = self.outer_step(carry, i, batch_size)
        # jax.debug.breakpoint()
        # batch_idx = jax.random.choice(split, jnp.arange(data.shape[0]), (batch_size, ), replace=False)
        # key, split = jax.random.split(key)
        # loss_value_grad = jax.value_and_grad(self.total_loss)
        # loss, grad = loss_value_grad(params, split, data[batch_idx], labels[batch_idx],
        #                              prot_mask[batch_idx], beta[batch_idx])
        # updates, opt_state = self.optim.update(grad, opt_state, params)
        # params = optax.apply_updates(params, updates)

        # get the bce and fairness loss individually
        # update the meta optimizer params to reduce the total loss

        def print_eval(key_loss):
            if self.verbose == False:
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
        jax.lax.cond(jnp.mod(i, 3000) == 0, lambda k: print_eval(k), lambda k: None, (split, inner_loss))
        # jax.lax.cond(jnp.mod(i, 3000) == 0, lambda x: jax.debug.print('iteration {x}', x=x), lambda x: None, i)

        # compare the current loss with the best loss
        curr_loss_smoothed = (last_loss + inner_loss)/2

        # if curr loss < best loss, replace
        best_loss, best_params = jax.lax.cond(curr_loss_smoothed < best_loss,
                lambda x: (x[0], x[1],),
                lambda x: (x[2], x[3],),
                (curr_loss_smoothed, params, best_loss, best_params)
            )

        return (key, meta_params, meta_state, uncert, data, labels, prot_mask, eval_data, eval_labels, m_test_data, m_test_labels, params,
                (best_loss, best_params), curr_loss_smoothed, opt_state), train_acc


    def transform_data(self, x):
        non_prot_attrs = [a for a in onp.arange(x.shape[1]) if a!=self.prot_attr_idx]
        return self.scaler.transform(x[:, non_prot_attrs])

    def fit(self, train_data, train_labels, num_iterations=10000,
            batch_size=1000, train_data_eval=None, test_data=None, train_labels_eval=None, test_labels=None,
            extra={'uncertainties': None, 'prot_attr_idx': None, 'rw': None}):
        uncertainties = jnp.array(extra['uncertainties'])
        prot_attr_idx = extra['prot_attr_idx']
        rw = extra['rw']

        self.prot_attr_idx = prot_attr_idx

        # get the data excluding the protected attributes
        non_prot_attr = [x for x in onp.arange(train_data.shape[1]) if x!=prot_attr_idx]
        train_data_training = train_data[:, non_prot_attr]

        train_data_training = self.scaler.fit_transform(train_data_training)

        # get the prot attr mask
        prot_mask = train_data[:, prot_attr_idx]

        self.meta_params = {
            'min': max(min(uncertainties), 0.1),
            'max': min(max(uncertainties), 0.35),
            'rw': rw
        }
        uncertainties = jnp.clip(uncertainties, 0.1, 0.35)

        self.meta_opt_state = self.meta_optim.init({
            'min': jnp.min(uncertainties),
            'max': jnp.max(uncertainties)
        })



        # create a scale for the uncertainties to be used for weights
        # get_beta = jit(partial(self.get_beta, unc_min=jnp.min(uncertainties), unc_max=jnp.max(uncertainties)))

        # get the betas
        # betas = get_beta(uncertainties)

        m_train_step_fn = partial(self.train_step_fn, batch_size=batch_size)

        (_, self.meta_params, self.meta_opt_state, _, _, _, _, _, _, _, _, self.params, (best_loss, best_params), _, self.opt_state), acc_hist = \
        jax.lax.scan(
            m_train_step_fn,
            (self.get_next_key(), self.meta_params, self.meta_opt_state, uncertainties, train_data_training, train_labels, prot_mask,
            self.transform_data(train_data_eval), train_labels_eval, self.transform_data(test_data), test_labels,
             self.params, (1.e10, self.params), 1.e10, self.opt_state),
            jnp.arange(num_iterations)
        )
        self.params = best_params

        return acc_hist
    def predict(self, x):
        split = self.get_next_key()
        x = self.transform_data(x)
        return jnp.argmax(self.apply(self.params, split, x, False), axis=1)
