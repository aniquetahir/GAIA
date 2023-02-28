import jax.numpy as jnp
import functools
from jax import jit
import chex
import jax
from typing import List
from jaxlib.xla_extension import DeviceArray

@functools.partial(jit, static_argnums=(1,))
def destructure(params, treedef: chex.PyTreeDef):
    flat_params, _ = jax.tree_util.tree_flatten(params)
    return jnp.concatenate(jax.tree_map(lambda x: x.flatten(), flat_params))


def destructure_divide(orig_params, destructured_params, divide_keys: List[List[str]]):
    """
    Divide the destructured parameters into lists of parameter sets based on the dictionary keys in `divide_keys`

    :param orig_params:
    :param destructured_params:
    :param divide_keys:
    :return:
    """
    for key_list in divide_keys:
        # restructure the params
        # get the param values for only the desired keys
        # destructure the new subdict
        # add to the list of
        pass
    pass
    raise NotImplementedError('destructure_divide not implemented')


def get_destructure_ranges(orig_params):
    def get_limits(carry, x):
        return carry + x, jnp.array([carry, carry + x])

    # Get the flattened shapes of the original params
    flat_orig, structure = jax.tree_util.tree_flatten(orig_params)
    shape_list = jax.tree_util.tree_map(lambda x: jnp.prod(jnp.array(x.shape)), flat_orig)
    _, destructure_ranges = jax.lax.scan(get_limits, 0, jnp.stack(shape_list))
    return tuple(tuple(x.tolist()) for x in destructure_ranges)


@functools.partial(jit, static_argnums=(2, 3))
def restructure(orig_params, destructured_params, destructure_ranges, orig_stucture: chex.PyTreeDef):
    flat_orig, structure = jax.tree_util.tree_flatten(orig_params)
    # print(destructure_ranges)

    # print(jax.tree_util.tree_map(lambda x: (x[0], x[1]), destructure_ranges))
    destructured_params_split = [destructured_params[x[0]: x[1]] for x in destructure_ranges]

    reshaped_params = jax.tree_util.tree_map(
        lambda x, y: x.reshape(y.shape),
        destructured_params_split, flat_orig
    )

    reshaped_params = jax.tree_util.tree_unflatten(orig_stucture, reshaped_params)
    return reshaped_params


def fake_scan(f, init, xs, length=None):
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, jnp.stack(ys)


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
