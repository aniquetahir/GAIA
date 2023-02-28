from utils.common import load_pickle, save_pickle
import numpy as onp
import jax.numpy as jnp

def get_data_from_pd(pd_file_path):
    df_adult = load_pickle(pd_file_path)
    data_numpy = df_adult.to_numpy()
    # data_numpy[:, [0, 1]] = data_numpy[:, [1, 0]]
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

def get_adult_data():
    return get_data_from_pd('adult_df.pd')

def get_compas_data():
    return get_data_from_pd('compas_df.pd')

def get_german_data():
    tr_d, tr_l, b_d, b_l, t_d, t_l =  get_data_from_pd('german_df.pd')
    tr_l, b_l, t_l = 2 - tr_l, 2 - b_l, 2 - t_l
    return tr_d, tr_l, b_d, b_l, t_d, t_l

