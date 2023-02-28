import jax
import chex
from jax import numpy as jnp
from sklearn.metrics import roc_auc_score
import numpy as onp

def get_accuracy(key: chex.PRNGKey, data: chex.Array, labels: chex.Array, params: chex.PyTreeDef, pred_fn):
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


def auc_roc_softmax_from_classifier(key: chex.PRNGKey, params, x: chex.Array, labels: chex.Array, apply_fn) -> float:
    key, split = jax.random.split(key)
    logits = apply_fn(params, split, x, False)
    return binary_auc_roc_softmax(logits, labels)


def binary_auc_roc_softmax(logits: chex.Array, labels: chex.Array) -> float:
    chex.assert_rank(labels, 1)
    chex.assert_shape(logits, (..., 2))
    # get the probabilities from the logits

    probs = jax.nn.softmax(logits, axis=1)
    return roc_auc_score(onp.array(labels), onp.array(probs)[:, 1])




