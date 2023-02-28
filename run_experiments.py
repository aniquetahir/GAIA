import functools

from typing import List, Dict, Tuple, Callable
import haiku as hk
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from modules.post_processing import finetuning, weighted_training
from modules.model_components import incompetent_get
from utils.jax.models.bnn import BNN
import jax, jax.numpy as jnp
from jax import jit
from bnn_utils import MyBNN, get_tpr_fpr_tnr_fnr as t_f_t_f_pr_nr_bnn, get_tpr as get_tpr_bnn, accuracy_bnn, ksplit
from bnn_utils import SimpleMLPClassifier, SimpleMLPClassifierWithClassDistributionRandomization, \
    get_tpr_pure, get_accuracy, get_tnr_pure
import chex
from itertools import product
from typing import List, Tuple
from tqdm import tqdm
from fairness_datasets import get_adult_data, get_compas_data, get_german_data
from utils.common import load_pickle, save_pickle
from aif360.sklearn.metrics import average_odds_difference, equal_opportunity_difference, \
    generalized_entropy_error, consistency_score, statistical_parity_difference
from modules.fairness_algorithms import get_reweight_stats, get_lfr_stats, get_opt_stats, get_adb_stats, \
    get_mf_stats, get_calibeo_stats, get_roc_stats
from jaxlib.xla_extension import DeviceArray
from utils.jax.models.bnn import log_prob, get_weight_sampler, restructure, destructure
import optax


onp = np


dataset_hparams = {
    'Adult': {
        'lr': 0.001,
        'num_layers': 0,
        'hidden_dim': 128,
        'dropout': 0.,
        'num_iterations': 100000
    },
    'German': {
        'lr': 0.001,
        'num_layers': 0,
        'hidden_dim': 128,
        'dropout': 0.,
        'num_iterations': 100000
    }
}



class BNNwProtDistributionShift(MyBNN):
    def sample_random_batch(self, key, data_all, prot_attr_mask: chex.Array, batch_size):
        data, labels = data_all

        fav_mask = (labels == 1).astype(float).flatten()
        unfav_mask = (labels == 0).astype(float).flatten()
        num_fav, num_unfav = fav_mask.sum(), unfav_mask.sum()

        prot_mask = (prot_attr_mask == 1).astype(float).flatten()
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

        batch_idx = jax.random.choice(split, jnp.arange(data.shape[0]), (batch_size, ), replace=False, p=selection_probs * selection_prot_probs)
        return data[batch_idx], labels[batch_idx]



class BNNwDistributionShift(MyBNN):
    def sample_random_batch(self, key, data_all, prot_attr_mask: chex.Array, batch_size):
        data, labels = data_all

        fav_mask = (labels == 1).astype(float).flatten()
        unfav_mask = (labels == 0).astype(float).flatten()
        num_fav, num_unfav = fav_mask.sum(), unfav_mask.sum()

        key, split = ksplit(key)
        perc_fav = jax.random.uniform(split)
        m_fav_mask = fav_mask * perc_fav
        m_unfav_mask = unfav_mask * (1 - perc_fav)
        selection_probs = (m_fav_mask/num_fav) + (m_unfav_mask/num_unfav)

        key, split = ksplit(key)

        batch_idx = jax.random.choice(split, jnp.arange(data.shape[0]), (batch_size, ), replace=False, p=selection_probs)
        return data[batch_idx], labels[batch_idx]


class StatsGenerator:
    PTYPE_EPIS = 0
    PTYPE_ALEA = 1

    def __init__(self, model, train_data, train_labels, test_data, test_labels, protected_attr_idx, batch_size=500, key=7):
        self.pre_init()
        self._model = model
        self._train_data = train_data
        self._train_labels = train_labels
        self._test_data = test_data
        self._test_labels = test_labels
        self._rng_seq = hk.PRNGSequence(key)
        self._prot_attr_idx = protected_attr_idx
        self.BATCH_SIZE=batch_size
        self.post_init()
        self._non_finetuned_params = None
        self._is_trained = False

    def pre_init(self):
        pass

    def post_init(self):
        pass

    def train(self):
        raise NotImplementedError('`train` not implemented')

    def key(self):
        return next(self._rng_seq)

    def transform_data(self, data):
        return data

    def get_tpr(self, data, labels):
        raise NotImplementedError('get_tpr not implemented')

    def get_equalized_odds(self, data, labels, protected_attr_idx):
        cls0_idxs = np.where(data[:, protected_attr_idx] == 0)[0]
        cls1_idxs = np.where(data[:, protected_attr_idx] == 1)[0]
        cls0_tpr = self.get_tpr(data[cls0_idxs], labels[cls0_idxs])
        cls1_tpr = self.get_tpr(data[cls1_idxs], labels[cls1_idxs])
        return np.abs(cls1_tpr - cls0_tpr)

    def get_accuracy(self, data, labels):
        raise NotImplementedError('get_accuracy not implemented')

    def get_balanced_accuracy(self, data, labels):
        raise NotImplementedError('get_balanced_accuracy not implemented')

    def get_predictions(self, data):
        raise NotImplementedError('`get_predictions` not implemented')

    def filter_data(self, x, filter_perc, filter_type):
        raise NotImplementedError('`filter_data` not implemented')

    def save_params(self):
        raise NotImplementedError('`save_params` not implemented')

    def restore_params(self):
        raise NotImplementedError

    def get_all_stats(self, data, labels, protected_attr_idx, filter_perc, filter_type):
        if not self._is_trained:
            self.train()
            self._is_trained = True
            self.save_params()
        else:
            self.restore_params()

        # Filter the data which is best for predictions
        filtered_idxs = self.filter_data(data, filter_perc, filter_type)
        non_prot_attrs = [x for x in range(data.shape[1]) if x != protected_attr_idx]

        data_fil, labels_fil = data[filtered_idxs], labels[filtered_idxs]

        y_true = pd.DataFrame(labels_fil)
        prot = np.array(data_fil[:, protected_attr_idx]).flatten()
        y_preds = self.get_predictions(data_fil)
        y_preds = np.array(y_preds)

        ig = incompetent_get
        stats = {
            'statistical_parity': ig(statistical_parity_difference)(y_true, y_preds, prot_attr=prot),
            'avg_odds_diff':                ig(average_odds_difference)(y_true, y_preds, prot_attr=prot),
            'equal_opportunity_diff':       ig(equal_opportunity_difference)(y_true, y_preds, prot_attr=prot),
            'generalized_entropy_error':    ig(generalized_entropy_error)(y_true.to_numpy().flatten(), y_preds),
            'consistency_score':            ig(consistency_score)(np.array(data_fil[:, non_prot_attrs]), y_preds),
            'accuracy':                     ig(self.get_accuracy)(data_fil, labels_fil),
            'bal_accuracy':                 ig(balanced_accuracy_score)(y_true, y_preds),
            # 'EO':                           ig(self.get_equalized_odds)(data_fil, labels_fil, protected_attr_idx)
        }

        # num_minibatches = len(self._train_data) // self.BATCH_SIZE
        # # Fine tune the data
        # finetuning(self._model, self._train_data, self._train_labels,
        #            self._test_data, self._test_labels,
        #            functools.partial(self.filter_data, filter_perc=filter_perc, filter_type=self.PTYPE_ALEA),
        #            num_minibatches, self.TRAIN_ITERS, extra={
        #                         'batch_size': self.BATCH_SIZE,
        #                         'prot_attrs': [self._prot_attr_idx]},
        #            n=self._num_verifications, verbose=False)
#
#
        # # data, labels = data[filtered_idxs], labels[filtered_idxs]
        # y_true = pd.DataFrame(labels)
        # prot = np.array(data[:, protected_attr_idx]).flatten()
        # y_preds = self.get_predictions(data)
        # y_preds = np.array(y_preds)
#
        # stats2 = {
        #     'finetune_statistical_parity': ig(statistical_parity_difference)(y_true, y_preds, prot_attr=prot),
        #     'finetune_avg_odds_diff':                ig(average_odds_difference)(y_true, y_preds, prot_attr=prot),
        #     'finetune_equal_opportunity_diff':       ig(equal_opportunity_difference)(y_true, y_preds, prot_attr=prot),
        #     'finetune_generalized_entropy_error':    ig(generalized_entropy_error)(y_true.to_numpy().flatten(), y_preds),
        #     'finetune_consistency_score':            ig(consistency_score)(np.array(data[:, non_prot_attrs]), y_preds),
        #     'finetune_accuracy':                     ig(self.get_accuracy)(data, labels),
        #     'finetune_bal_accuracy':                 ig(self.get_balanced_accuracy)(data, labels),
        #     'finetune_EO':                           ig(self.get_equalized_odds)(data, labels, protected_attr_idx)
        # }
#
        # for k, v in stats2.items():
        #     stats[k] = v

        return stats


class BNNStatsGenerator(StatsGenerator):
    _model: BNN
    _num_verifications: int = 150
    TRAIN_ITERS: int = 50000


    def post_init(self):
        non_prot_attrs = [x for x in range(self._train_data.shape[1]) if x != self._prot_attr_idx]
        self._non_prot_attrs = non_prot_attrs

    def train(self):
        # td = self.transform_data
        self._model.train(
            (self._train_data, self._train_labels, ),
            (self._test_data, self._test_labels, ),
            len(self._train_data) // self.BATCH_SIZE,
            num_iterations=self.TRAIN_ITERS, extra={
                'batch_size': self.BATCH_SIZE,
                'prot_attrs': [self._prot_attr_idx]
            }
        )


    def transform_data(self, data):
        return data[:, self._non_prot_attrs]

    def get_tpr(self, data, labels):
        return get_tpr_bnn(self._model, self.transform_data(data), labels, self._num_verifications)

    def get_accuracy(self, data, labels):
        return accuracy_bnn(self._model, self.transform_data(data), labels, self._num_verifications).item()

    def get_predictions(self, data):
        return onp.array(jnp.argmax(self._model.get_mean_predictions(self.transform_data(data), self._num_verifications), axis=1))

    def get_balanced_accuracy(self, data, labels):
        tpr, fpr, tnr, fnr = t_f_t_f_pr_nr_bnn(self._model, self.transform_data(data), labels, self._num_verifications)
        return ((tpr + tnr) / 2).item()

    def save_params(self):
        self._non_finetuned_params = self._model._baysian_params

    def restore_params(self):
        self._model._baysian_params = self._non_finetuned_params

    def filter_data(self, x, filter_perc, filter_type: int):
        # get the number of samples in the final results
        num_samples = int(len(x) * filter_perc)

        # get the uncertainty
        uncertainties = self._model.get_aleatoric_uncertainty_v2(self.transform_data(x), self._num_verifications) \
                            if filter_type == self.PTYPE_ALEA \
                            else self._model.get_epistemic_uncertainty(self.transform_data(x), self._num_verifications)

        # filter the data based on the uncertainty
        unc_sorted = sorted(enumerate(uncertainties.tolist()), key=lambda x: x[1])
        filtered_idxs = list(zip(*unc_sorted))[0][:num_samples]

        # return the indices
        return list(filtered_idxs)


class SimpleMLPStatsGenerator(StatsGenerator):
    _model: SimpleMLPClassifier
    TRAIN_ITERS: int = 50000

    def post_init(self):
        non_prot_attrs = [x for x in onp.arange(self._train_data.shape[1]) if x != self._prot_attr_idx]
        self._non_prot_attrs = non_prot_attrs

    def train(self):
        # td = self.transform_data
        self._model.fit(self._train_data[:, self._non_prot_attrs], self._train_labels,
                        num_iterations=self.TRAIN_ITERS, batch_size=self.BATCH_SIZE,
                        train_data_eval=self._train_data[:, self._non_prot_attrs], test_data=self._test_data[:, self._non_prot_attrs],
                        train_labels_eval=self._train_labels, test_labels=self._test_labels)

    def transform_data(self, data):
        return data[:, self._non_prot_attrs]

    def get_tpr(self, data, labels):
        return get_tpr_pure(self.key(), self.transform_data(data), labels, self._model.params, self._model.apply).item()

    def get_accuracy(self, data, labels):
        return get_accuracy(self.key(), self.transform_data(data), labels, self._model.params, self._model.apply).item()

    def get_predictions(self, data):
        return self._model.predict(self.transform_data(data))

    def get_balanced_accuracy(self, data, labels):
        tpr = self.get_tpr(data, labels)
        tnr = get_tnr_pure(self.key(), self.transform_data(data), labels, self._model.params, self._model.apply)

        return ((tpr + tnr) / 2).item()


def bnn_pruning_results_mlp_student(bnn: BNN, train_data, train_labels, test_data, test_labels,
                                    prot_attr_idx, dataset_name, bnn_type, BATCH_SIZE=500,
                                    TRAIN_ITERS=50000, NUM_VERIF=150):
    bnn_stats_gen = BNNStatsGenerator(bnn, train_data, train_labels, test_data, test_labels,
                                      prot_attr_idx, batch_size=BATCH_SIZE, key=bnn.next_key())

    all_results = []
    bnn_stats = bnn_stats_gen.get_all_stats(test_data, test_labels, prot_attr_idx, 1., 1)

    final_model, weight_mlp_stats = weighted_training(bnn, train_data, train_labels, test_data, test_labels,
                                    prot_attr_idx, TRAIN_ITERS, BATCH_SIZE, NUM_VERIF, dataset_hparams[dataset_name])

    all_stats = {}
    for k, v in bnn_stats.items():
        all_stats[f'bnn_{k}'] = v

    for k, v in weight_mlp_stats.items():
        all_stats[f'weighted_{k}'] = v


    all_stats['dataset'] = dataset_name
    all_stats['bnn_version'] = bnn_type

    all_results.append(all_stats)

    return all_results

def dict_add_pre(pre: str, dict: Dict) -> Dict:
    new_dict = {}
    for k, v in dict.items():
        new_dict[f'{pre}_{k}'] = v
    return new_dict

def merge_dicts(*many_dicts):
    merged_dict = {}
    for d in many_dicts:
        for k, v in d.items():
            merged_dict[k] = v
    return merged_dict

if __name__ == "__main__":
    rng_seq = hk.PRNGSequence(7)
    LR = 0.001

    datasets = [
        ('Adult', get_adult_data, 1, 500),
        ('German', get_german_data, 1, 64)
    ]

    all_results = []
    all_baseline_results = []

    for ds_name, ds_getr_fn, prot_attr_idx, batch_size in tqdm(datasets, desc=' datasets', position=0):
        # print('=' * 10)
        # print(ds_name)
        # print('=' * 10)


        for i in tqdm(range(10), desc='repetition', position=1, leave=False):
            train_data, train_labels, _, _, test_data, test_labels = ds_getr_fn()
            try:
                rw_stats = get_reweight_stats(train_data, train_labels, test_data, test_labels, ds_name, prot_attr_idx)
                lfr_stats = get_lfr_stats(train_data, train_labels, test_data, test_labels, ds_name, prot_attr_idx)
                opt_stats = get_opt_stats(train_data, train_labels, test_data, test_labels, ds_name, prot_attr_idx)
                adb_stats = get_adb_stats(train_data, train_labels, test_data, test_labels, ds_name, prot_attr_idx)
                mf_stats = get_mf_stats(train_data, train_labels, test_data, test_labels, ds_name, prot_attr_idx)
                calibeo_stats = get_calibeo_stats(train_data, train_labels, test_data, test_labels, ds_name, prot_attr_idx)
                roc_stats = get_roc_stats(train_data, train_labels, test_data, test_labels, ds_name, prot_attr_idx)
            except Exception as e:
               print(e)
               continue

            baseline_results = merge_dicts(
                dict_add_pre('rw', rw_stats),
                dict_add_pre('lfr', lfr_stats),
                dict_add_pre('opt', opt_stats),
                dict_add_pre('adb', adb_stats),
                dict_add_pre('mf', mf_stats),
                dict_add_pre('calibeo', calibeo_stats),
                dict_add_pre('roc', roc_stats),
            )
            baseline_results['dataset'] = ds_name
            all_baseline_results.append([baseline_results])

            myBNN = BNNwDistributionShift(next(rng_seq), train_data[:10][:, 1:].shape, 3, 256, 2, learning_rate=LR)
            results = bnn_pruning_results_mlp_student(myBNN, train_data, train_labels,
                                                      test_data, test_labels, prot_attr_idx=prot_attr_idx,
                                                      dataset_name=ds_name, bnn_type='LabelShift', BATCH_SIZE=batch_size)

            all_results.append(results)

            myBNN = BNNwProtDistributionShift(next(rng_seq), train_data[:10][:, 1:].shape, 3, 256, 2, learning_rate=LR)
            results = bnn_pruning_results_mlp_student(myBNN, train_data, train_labels,
                                                      test_data, test_labels, prot_attr_idx=prot_attr_idx,
                                                      dataset_name=ds_name, bnn_type='AttrLabelShift', BATCH_SIZE=batch_size)
            all_results.append(results)

        # save_pickle(
        #     [y for x in all_results for y in x],
        #     f'compiled_results/filter_results_{ds_name}.pkl')

    save_pickle(
        [y for x in all_results for y in x],
        f'compiled_results/filter_results.pkl')
    save_pickle(
        [y for x in all_baseline_results for y in x],
        f'compiled_results/filter_baseline_results.pkl'
    )
