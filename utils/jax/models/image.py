import functools

import haiku as hk
import jax.numpy as jnp
import os
import pickle
import jax.random
import numpy as np
import numpy as onp
import chex
import optax
from haiku import PRNGSequence
from tqdm import tqdm

from jax.experimental.host_callback import call as hcb


CELEBA_PATH = '/home/anique/projects/fair-mixup/celeba'

from utils.datasets.celeba import get_loader
ksplit = jax.random.split


class ResnetClassifier():
    def __init__(self, data_shape, is_training=True, lr=0.001, key=0, verbose=False):
        self.verbose = verbose
        self.rseq = PRNGSequence(key)
        self._is_training = is_training
        init, apply = hk.without_apply_rng(hk.transform_with_state(
            lambda x: hk.nets.ResNet18(2, resnet_v2=True)(x, is_training=is_training)
        ))
        params, resnet_state = init(self.next_key(), jnp.zeros(data_shape))
        self._params = params
        self._resnet_state = resnet_state
        self._optim = optax.adam(lr)
        self._optim_state = self._optim.init(params)
        self._is_trained = False


    def next_key(self):
        return next(self.rseq)

    def set_training(self, is_training=True):
        self._is_training = is_training
        _, self.apply = hk.without_apply_rng(hk.transform_with_state(
            lambda x: hk.nets.ResNet18(2, resnet_v2=True)(x, is_training=is_training)
        ))


    def fit(self, train_data_iter, test_data_iter, num_iterations, batch_size):
        # raise NotImplementedError('fit not implemented')

        def loss_func(m_params, m_state, x, y):
            logits, new_state = self.apply(m_params, m_state, x)
            nll = optax.softmax_cross_entropy_with_integer_labels(logits, y.astype(int)).mean()
            return nll, new_state


        @functools.partial(jax.jit, static_argnames=('loss_fn', ))
        def update_step(key, m_params, m_state, data, labels, opt_state, loss_fn):
            (loss_val, new_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(m_params, m_state, data, labels)
            updates, opt_state = self._optim.update(grads, opt_state, m_params)
            m_params = optax.apply_updates(m_params, updates)

            return m_params, new_state, opt_state, loss_val

        # training loop
        for i in tqdm(range(num_iterations)):
            x, y, a = next(train_data_iter)
            x, y = jnp.array(x.numpy()), jnp.array(y.numpy().astype(int))
            self._params, self._resnet_state, self._optim_state, loss = \
                update_step(self.next_key(), self._params, self._resnet_state, x, y, self._optim_state, loss_func)

            if i % 100 == 0:
                print(f'loss: {loss}')
                vx, vy, _ = next(test_data_iter)
                vx, vy = jnp.array(vx.numpy()), jnp.array(vy.numpy().astype(int))
                logits, _ = self.apply(self._params, self._resnet_state, vx)
                valid_acc = (jnp.argmax(logits, axis=1) == vy).astype(jnp.float32).mean()

                logits, _ = self.apply(self._params, self._resnet_state, x)
                train_acc = (jnp.argmax(logits, axis=1) == y).astype(jnp.float32).mean()
                print(f'valid acc: {valid_acc}, train_acc: {train_acc}')
        self._is_trained = True

    def transform(self, x):
        logits, _ = self.apply(self._params, self._resnet_state, x)
        return onp.array(jnp.argmax(logits, axis=1))

    def predict(self, x):
        return self.transform(x)

class WeightedResnetClassifier(ResnetClassifier):
    def __init__(self, data_shape, is_training=True, lr=0.001, key=0, verbose=False):
        super(WeightedResnetClassifier, self).__init__(data_shape, is_training, lr, key, verbose=verbose)
        self._is_training = is_training
        _, self.apply = hk.without_apply_rng(hk.transform_with_state(
            lambda x: hk.nets.ResNet18(2, resnet_v2=True)(x, is_training=is_training)
        ))
        _, self.apply_with_training_option =  hk.without_apply_rng(hk.transform_with_state(
            lambda x, it: hk.nets.ResNet18(2, resnet_v2=True)(x, is_training=it)
        ))

    def get_beta(self, params, unc_val):
        unc_min = params['min']
        unc_max = params['max']
        return ((unc_val - unc_min) / (unc_max - unc_min)) ** 1

    def loss_ce(self, params, resnet_state, x, y, weights, is_training=True):
        logits, new_state = self.apply_with_training_option(params, resnet_state, x, is_training)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y) * weights
        return loss.mean(), new_state


    def loss_fair(self, params, resnet_state, x, y, weights, prot_attrs, is_training=True):
        logits, new_state = self.apply_with_training_option(params, resnet_state, x, is_training)
        loss_ce_all = optax.softmax_cross_entropy_with_integer_labels(logits, y) * weights
        loss_a_1 = (loss_ce_all * prot_attrs).sum()/prot_attrs.sum()
        loss_a_2 = (loss_ce_all * (1-prot_attrs)).sum()/(1-prot_attrs).sum()
        return jnp.abs(loss_a_2 - loss_a_1), new_state


    def total_loss(self, params, resnet_state, x, y, prot_attrs, beta, is_training=True):
        lam = 4.
        l_fair, new_state = self.loss_fair(params, resnet_state, x, y, beta, prot_attrs, is_training)
        l_util, new_state = self.loss_ce(params, new_state, x, y, (1-beta), is_training)
        return lam * l_fair + l_util, new_state


    def inner_step(self, theta, resnet_state, opt_state, x, y, a, beta_batch):
        _, _ = self.total_loss(theta, resnet_state, x, y, a, beta_batch)
        loss_value_grad = jax.value_and_grad(self.total_loss, has_aux=True)
        (loss, new_state), grad = loss_value_grad(theta, resnet_state, x, y, a, beta_batch)
        updates, opt_state = self._optim.update(grad, opt_state, theta)
        theta = optax.apply_updates(theta, updates)

        return loss, (theta, new_state, opt_state)


    def train_step_fn(self, carry, i):
        beta_batch, x, y, a, x_test, y_test, params, resnet_state, best_model, last_loss, opt_state = carry

        best_loss, best_params, best_model_state = best_model
        loss, (params, resnet_state, opt_state) = self.inner_step(params, resnet_state, opt_state, x, y, a, beta_batch)

        def print_eval(loss):
            if self.verbose == False:
                return
            train_preds = jnp.argmax(self.apply_with_training_option(params, resnet_state, x, False)[0], axis=1)
            train_acc   = (train_preds == y).astype(float).mean()

            test_predictions = jnp.argmax(self.apply_with_training_option(params, resnet_state, x_test, False)[0], axis=1)
            test_acc = (test_predictions == y_test).astype(float).mean()

            jax.debug.print('loss: {a}, train acc: {c}, test acc: {d}', a=loss,
                            c=train_acc, d=test_acc)


        train_predictions = jnp.argmax(self.apply(params, resnet_state, x)[0], axis=1)
        train_acc = (train_predictions == y).astype(float).mean()

        jax.lax.cond(jnp.mod(i, 300) == 0, lambda k: print_eval(k), lambda k: None, loss)
        # jax.lax.cond(jnp.mod(i, 3000) == 0, lambda x: jax.debug.print('iteration {x}', x=x), lambda x: None, i)

        # compare the current loss with the best loss
        curr_loss_smoothed = (last_loss + loss)/2

        # if curr loss < best loss, replace
        best_loss, best_params, best_model_state = jax.lax.cond(curr_loss_smoothed < best_loss,
                                                                lambda x: (x[0], x[1], x[4]),
                                                                lambda x: (x[2], x[3], x[5]),
                                                                (curr_loss_smoothed, params, best_loss, best_params, resnet_state, best_model_state)
                                                                )

        return (beta_batch, x, y, a, x_test, y_test, params, resnet_state,
                (best_loss, best_params, best_model_state), curr_loss_smoothed, opt_state), train_acc



    def fit(self, train_data_iter, test_data_iter, num_iterations, batch_size,
            extra={
                'uncertainties': None,
                'rw': None
            }):

        uncertainties = jnp.array(extra['uncertainties'])

        # get the data excluding the protected attributes
        train_iter = iter(infinite_dataloader(train_data_iter))
        test_iter = iter(infinite_dataloader(test_data_iter))

        uplim = 0.0
        downlim = 0.17
        # self.meta_params = {
        #     'min': 0.,
        #     'max': 0.44
        # }
        self.meta_params = {
            'min': max(jnp.min(uncertainties), uplim),
            'max': min(jnp.max(uncertainties), downlim),
        }
        uncertainties = jnp.clip(uncertainties, uplim, downlim)




        # create a scale for the uncertainties to be used for weights
        # get_beta = jit(partial(self.get_beta, unc_min=jnp.min(uncertainties), unc_max=jnp.max(uncertainties)))

        # get the betas
        # betas = get_beta(uncertainties)

        params = self._params
        resnet_state = self._resnet_state
        best_loss = 1.e10
        best_params = params
        best_model_state = resnet_state

        train_step_fn = jax.jit(self.train_step_fn)
        # train_step_fn = self.train_step_fn
        get_beta = jax.jit(self.get_beta)
        curr_loss = 1.e10

        def next_sample(tmp=None):
            b_i, x, y, a = next(train_iter)
            if len(y) != batch_size:
                b_i, x, y, a = next(train_iter)

            b_i, x, y, a = b_i.numpy().astype(int), x.numpy(), y.numpy().astype(int), a.numpy().astype(int)


            return {
                'b': get_beta(self.meta_params, uncertainties[b_i]),
                'x': x,
                'y': y,
                'a': a
            }

        def next_test_sample(tmp=None):
            b_i, x, y, a = next(test_iter)
            if len(y) != batch_size:
                b_i, x, y, a = next(test_iter)

            return {
                'x': x.numpy(),
                'y': y.numpy().astype(int),
                'a': a.numpy().astype(int)
            }
        tmp_train_sample = next_sample()
        train_b_shape, train_x_shape, train_y_shape, train_a_shape = \
            tmp_train_sample['b'].shape, tmp_train_sample['x'].shape, \
                tmp_train_sample['y'].shape, tmp_train_sample['a'].shape

        tmp_test_sample = next_test_sample()
        test_x_shape, test_y_shape, test_a_shape = \
            tmp_test_sample['x'].shape, tmp_test_sample['y'].shape, tmp_test_sample['a'].shape


        def train_step_fn_hcb(carry, i):
            params, resnet_state, best_model, last_loss, opt_state = carry

            train_batch = hcb(next_sample, None, result_shape={
                'b': jnp.ones(train_b_shape),
                'x': np.ones(train_x_shape),
                'y': np.ones(train_y_shape, dtype=int),
                'a': np.ones(train_a_shape, dtype=int)
            })

            test_batch = hcb(next_test_sample, None, result_shape={
                'x': np.ones(test_x_shape),
                'y': np.ones(test_y_shape, dtype=int),
                'a': np.ones(test_a_shape, dtype=int)
            })

            beta_batch, x, y, a  = train_batch['b'], train_batch['x'], train_batch['y'], train_batch['a']
            x_test, y_test = test_batch['x'], test_batch['y']

            best_loss, best_params, best_model_state = best_model
            # jax.debug.breakpoint()
            loss, (params, resnet_state, opt_state) = self.inner_step(params, resnet_state, opt_state, x, y, a, beta_batch)

            def print_eval(loss):
                if self.verbose == False:
                    return
                train_preds = jnp.argmax(self.apply_with_training_option(params, resnet_state, x, False)[0], axis=1)
                train_acc   = (train_preds == y).astype(float).mean()

                test_predictions = jnp.argmax(self.apply_with_training_option(params, resnet_state, x_test, False)[0], axis=1)
                test_acc = (test_predictions == y_test).astype(float).mean()

                jax.debug.print('loss: {a}, train acc: {c}, test acc: {d}', a=loss,
                                c=train_acc, d=test_acc)


            train_predictions = jnp.argmax(self.apply(params, resnet_state, x)[0], axis=1)
            train_acc = (train_predictions == y).astype(float).mean()

            jax.lax.cond(jnp.mod(i, 300) == 0, lambda k: print_eval(k), lambda k: None, loss)
            jax.lax.cond(jnp.mod(i, 100) == 0, lambda x: jax.debug.print('iteration {x}', x=x), lambda x: None, i)

            # compare the current loss with the best loss
            curr_loss_smoothed = (last_loss + loss)/2

            # if curr loss < best loss, replace
            best_loss, best_params, best_model_state = jax.lax.cond(curr_loss_smoothed < best_loss,
                                                                    lambda x: (x[0], x[1], x[4]),
                                                                    lambda x: (x[2], x[3], x[5]),
                                                                    (curr_loss_smoothed, params, best_loss, best_params, resnet_state, best_model_state)
                                                                    )

            return (params, resnet_state,
                    (best_loss, best_params, best_model_state), curr_loss_smoothed, opt_state), train_acc


        (_, _, (best_loss, best_params, best_model_state), curr_loss, self._optim_state), _ = \
            jax.lax.scan(train_step_fn_hcb,  # looping function
                         (
                             params, resnet_state,
                             (best_loss, best_params, best_model_state),
                             curr_loss, self._optim_state),  # carry
                         jnp.arange(num_iterations)  # iterative array
                         )

        self._params = best_params
        self._resnet_state = best_model_state
        # for i in tqdm(range(num_iterations)):
        #     # get the batch
        #     b_i, x, y, a = next(train_iter)
        #     b_i, x, y, a = b_i.numpy().astype(int), x.numpy(), y.numpy().astype(int), a.numpy().astype(int)
        #
        #     _, x_test, y_test, _ = next(test_iter)
        #     x_test, y_test = x_test.numpy(), y_test.numpy().astype(int)
        #
        #     # map the uncertainties
        #     b_uncs = uncertainties[b_i]
        #
        #     # get beta from the uncertainties
        #     b_betas = get_beta(self.meta_params, b_uncs)
        #
        #     # train on the batch
        #     (_, _, _, _, _, _,
        #         params, resnet_state, (best_loss, best_params, best_model_state),
        #         curr_loss, self._optim_state), acc = train_step_fn(
        #             (b_betas, x, y, a, x_test, y_test, params, resnet_state,
        #              (best_loss, best_params, best_model_state),
        #              curr_loss, self._optim_state
        #              ), i)
        #
        # self._params = best_params
        # self._resnet_state = best_model_state

        # return acc_hist
        # raise NotImplementedError('fit not implemented')

class Resnet18Encoder(hk.Module):
    def __init__(self, num_classes, name=None):
        super(Resnet18Encoder, self).__init__(name=name)
        self.num_classes = num_classes
        self.resnet = hk.nets.ResNet18(num_classes, resnet_v2=True)


    def __call__(self, x):
        outputs = self.resnet(x)
        return outputs.reshape(-1, 512, 8, 8)


def infinite_dataloader(dl):
    iterator = iter(dl)
    while True:
        try:
            next_ret = next(iterator)
        except StopIteration:
            iterator = iter(dl)
            next_ret = next(iterator)

        yield next_ret


def get_dl_with_idx(target_id = 2, batch_size = 64, split='train'):
    with open(os.path.join(CELEBA_PATH, 'data_frame.pickle'), 'rb') as handle:
        df = pickle.load(handle)
    train_df = df[split]
    return get_loader(train_df, os.path.join(CELEBA_PATH, f'split/{split}'), target_id, batch_size, ret_idxs=True)

def get_celeba_dl(target_id = 2, batch_size = 64, split='train'):
    with open(os.path.join(CELEBA_PATH, 'data_frame.pickle'), 'rb') as handle:
        df = pickle.load(handle)
    train_df = df[split]
    return get_loader(train_df, os.path.join(CELEBA_PATH, f'split/{split}'), target_id, batch_size)

def train_simple_resnet(target_id, batch_size=64, num_iterations=10000):
    with open(os.path.join(CELEBA_PATH, 'data_frame.pickle'), 'rb') as handle:
        df = pickle.load(handle)

    train_df = df['train']
    test_df = df['test']

    train_dataloader = get_loader(train_df, os.path.join(CELEBA_PATH, 'split/train'), target_id, batch_size)
    num_training = len(train_dataloader)
    train_dataloader = iter(infinite_dataloader(train_dataloader))
    num_minibatches = num_training//batch_size

    test_dataloader = get_loader(test_df, os.path.join(CELEBA_PATH, 'split/test'), target_id, batch_size)
    test_dataloader = iter(infinite_dataloader(test_dataloader))

    train_batch_x, _, _ = next(train_dataloader)
    train_batch_x = train_batch_x.numpy()

    key = jax.random.PRNGKey(0)
    kseq = PRNGSequence(key)

    # get the model parameters
    init, apply = hk.without_apply_rng(hk.transform_with_state(
        lambda x: hk.nets.ResNet18(2, resnet_v2=True)(x, is_training=True)
    ))

    params, resnet_state = init(next(kseq), jnp.zeros_like(train_batch_x[:2]))
    optim = optax.adam(0.001)
    optim_state = optim.init(params)

    def loss_func(m_params, m_state, x, y):
        logits, new_state = apply(m_params, m_state, x)
        nll = optax.softmax_cross_entropy_with_integer_labels(logits, y.astype(int)).mean()
        return nll, new_state


    @functools.partial(jax.jit, static_argnames=('loss_fn', ))
    def update_step(key, m_params, m_state, data, labels, opt_state, loss_fn):
        (loss_val, new_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(m_params, m_state, data, labels)
        updates, opt_state = optim.update(grads, opt_state, m_params)
        m_params = optax.apply_updates(m_params, updates)

        return m_params, new_state, opt_state, loss_val

    num_iterations = 10000
    for i in tqdm(range(num_iterations)):
        x, y, a = next(train_dataloader)
        x, y = jnp.array(x.numpy()), jnp.array(y.numpy().astype(int))
        params, resnet_state, optim_state, loss = update_step(next(kseq), params, resnet_state, x, y, optim_state, loss_func)
        if i % 100 == 0:
            print(f'loss: {loss}')
            # Calculate the performance on the validation set
            vx, vy, a = next(test_dataloader)
            vx, vy = jnp.array(vx.numpy()), jnp.array(vy.numpy().astype(int))
            logits, _ = apply(params, resnet_state, vx)
            valid_acc = (jnp.argmax(logits, axis=1) == vy).astype(jnp.float32).mean()
            # get the train accuracy
            logits, _ = apply(params, resnet_state, x)
            train_acc = (jnp.argmax(logits, axis=1) == y).astype(jnp.float32).mean()
            print(f'valid acc: {valid_acc}, train_acc: {train_acc}')

