import jax
import chex
import optax
from jax import numpy as jnp
from metrics import mean_where


def get_tpr_pure(key, x, gt_labels, params, pred_fn):
    mask = (gt_labels == 1).astype(jnp.int_).flatten()
    # from the capable, find the ones predicted capable
    key, split = jax.random.split(key)
    predictions = jnp.argmax(pred_fn(params, split, x, False), axis=1) # classifier.predict(x[indices_capable])
    # get number of predictions which are favored
    return mean_where(predictions, mask)
    # num_predicted_favored = (predictions == 1).sum()
    # return num_predicted_favored / len(indices_capable)


def get_tnr_pure(key, x, gt_labels, params, pred_fn):
    # indices_capable = jnp.where(gt_labels == 0)[0]
    mask = (gt_labels == 0).astype(jnp.int_).flatten()
    # from the capable, find the ones predicted capable
    key, split = jax.random.split(key)
    predictions = jnp.argmax(pred_fn(params, split, x, False), axis=1) # classifier.predict(x[indices_capable])
    # jax.debug.print('min: {a}, max: {b}, pred x mask: {c}', a=jnp.min(predictions), b=jnp.max(predictions), c=(predictions*mask).shape)
    # get number of predictions which are favored
    return mean_where((predictions == 0).astype(jnp.int_), mask)
    # num_predicted_favored = (predictions == 0).sum()
    # return num_predicted_favored / len(indices_capable)

