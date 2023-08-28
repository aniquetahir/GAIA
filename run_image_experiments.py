import os.path
import pickle

from typing import List, Dict, Tuple, Callable
import haiku as hk
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from utils.datasets.celeba import CelebA, get_dataset as get_celeba_dataset

from modules.post_processing import weighted_training_image
from modules.model_components import incompetent_get
import jax, jax.numpy as jnp

from utils.jax.models.image_bnn import ImageBNN
from utils.jax.models.image import ResnetClassifier, infinite_dataloader, CELEBA_PATH

from bnn_utils import get_tpr_fpr_tnr_fnr as t_f_t_f_pr_nr_bnn, get_tpr as get_tpr_bnn, accuracy_bnn, ksplit
from bnn_utils import get_tpr_pure, get_accuracy, get_tnr_pure

from tqdm import tqdm

from utils.common import save_pickle
from aif360.sklearn.metrics import average_odds_difference, equal_opportunity_difference, \
    generalized_entropy_error, consistency_score, statistical_parity_difference
from utils.jax.models.image import get_dl_with_idx, get_celeba_dl
from modules.fairness_algorithms import get_fmu_stats, get_fairbatch_stats

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
    },
    'celeba': {
        'lr': 0.001,
        'num_iterations': 5000
    }
}

class ImageBNNwProtDistributionShift(ImageBNN):
    def sample_random_batch(self, key, data_iter, batch_size):
        ds: CelebA = data_iter.dataset
        prot_attr_idx = ds.gender_id
        target_idx = ds.target_id

        labels, prot_attr_mask = ds.labels[:, target_idx], ds.labels[:, prot_attr_idx]

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


        # batch_idx = jax.random.choice(split, jnp.arange(data.shape[0]), (batch_size, ), replace=False, p=selection_probs)
        batch_idx = jax.random.choice(split, jnp.arange(len(labels)), (batch_size, ), replace=False, p=selection_probs * selection_prot_probs)

        x, y, _ = ds.__getitem__(np.array(batch_idx))

        return np.stack([n.numpy() for n in x]), np.array(y)




class ImageBNNwDistributionShift(ImageBNN):
    def sample_random_batch(self, key, data_iter, batch_size):
        ds: CelebA = data_iter.dataset
        prot_attr_idx = ds.gender_id
        target_idx = ds.target_id

        labels, prot_attr_mask = ds.labels[:, target_idx], ds.labels[:, prot_attr_idx]

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


        batch_idx = jax.random.choice(split, jnp.arange(len(labels)), (batch_size, ), replace=False, p=selection_probs)
        # batch_idx = jax.random.choice(split, jnp.arange(len(labels)), (batch_size, ), replace=False, p=selection_probs * selection_prot_probs)

        x, y, _ = ds.__getitem__(np.array(batch_idx))

        return np.stack([n.numpy() for n in x]), np.array(y)

class ImageStatsGenerator:
    PTYPE_EPIS = 0
    PTYPE_ALEA = 1

    def __init__(self, model,
                 train_dataloader, test_dataloader, batch_size=128, key=7):
        self.pre_init()
        self._model: ResnetClassifier = model
        self._train_dataloader = train_dataloader
        self._test_dataloader = test_dataloader
        self._rng_seq = hk.PRNGSequence(key)
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

    def get_equalized_odds(self, data_iter):
        # TODO Implement
        raise NotImplementedError('get_equalized_odds not implemented')
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

    def get_all_stats(self, dataloader):
        if not self._is_trained:
            # if os.path.exists('bnn_image.state'):
            #     with open('bnn_image.state', 'rb') as bnn_state_file:
            #         bnn_state = pickle.load(bnn_state_file)
            #         self._model._baysian_params = bnn_state['b_params']
            #         self._model._resnet_state = bnn_state['r_state']
            # else:
            self.train()
            self._is_trained = True
            self.save_params()

            # save the bnn
            # with open('bnn_image.state', 'wb') as bnn_state_file:
            #     bnn_state = {
            #         'b_params': self._model._baysian_params,
            #         'r_state': self._model._resnet_state
            #     }
            #     pickle.dump(bnn_state, bnn_state_file)
        else:
            self.restore_params()


        # Filter the data which is best for predictions
        # filtered_idxs = self.filter_data(data, filter_perc, filter_type)
        # non_prot_attrs = [x for x in range(data.shape[1]) if x != protected_attr_idx]

        # data_fil, labels_fil = data[filtered_idxs], labels[filtered_idxs]

        # Create an iterator
        # Get the predictions and the labels
        data_iter = iter(dataloader)
        preds = []
        labels = []
        a_s = []
        self._model.set_training(False)

        if os.path.exists('bnn_image.state'):
            return {}

        while True:
            try:
                x, y, a = next(data_iter)
            except StopIteration:
                break
            x, y, a = x.numpy(), y.numpy().astype(int), a.numpy().astype(int)
            labels.append(y)
            a_s.append(a)
            pred = self.get_predictions(x)
            preds.append(pred)

        # stack everything
        y_preds = np.array(jnp.concatenate(preds))
        y_true = pd.DataFrame(np.concatenate(labels))
        prot = np.concatenate(a_s)

        ig = incompetent_get
        stats = {
            'statistical_parity': ig(statistical_parity_difference)(y_true, y_preds, prot_attr=prot),
            'avg_odds_diff':                ig(average_odds_difference)(y_true, y_preds, prot_attr=prot),
            'equal_opportunity_diff':       ig(equal_opportunity_difference)(y_true, y_preds, prot_attr=prot),
            'generalized_entropy_error':    ig(generalized_entropy_error)(y_true.to_numpy().flatten(), y_preds),
            'bal_accuracy':                 ig(balanced_accuracy_score)(y_true, y_preds),
        }

        return stats


class ImageBNNStatsGenerator(ImageStatsGenerator):
    _model: ImageBNN
    _num_verifications: int = 150
    TRAIN_ITERS: int = 10000


    def post_init(self):
        pass

    def train(self):
        # td = self.transform_data
        # create an infinite looping loader
        # train_loader = infinite_dataloader(self._train_dataloader)
        # test_loader = infinite_dataloader(self._test_dataloader)

        train_loader = self._train_dataloader
        test_loader = self._test_dataloader

        # num_training = len(self._train_dataloader)
        num_minibatches = len(self._train_dataloader)
        # num_training//self.BATCH_SIZE

        self._model.train(
            train_loader, test_loader, num_minibatches, num_iterations=self.TRAIN_ITERS,
            extra={
                'batch_size': self.BATCH_SIZE,
                'prot_attrs': [20],
                'num_minibatches': num_minibatches
            }
        )


    def transform_data(self, data):
        return data

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
        raise NotImplementedError('filter_data not implemented')
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


class SimpleImageStatsGenerator(ImageStatsGenerator):
    _model: ResnetClassifier
    TRAIN_ITERS: int = 50000

    def post_init(self):
        # No need to get the non protected attributes since image does not contain protected stuff
        # non_prot_attrs = [x for x in onp.arange(self._train_data.shape[1]) if x != self._prot_attr_idx]
        # self._non_prot_attrs = non_prot_attrs
        pass

    def train(self):
        train_loader = infinite_dataloader(self._train_dataloader)
        test_loader = infinite_dataloader(self._test_dataloader)

        # Train the model
        self._model.fit(train_loader, test_loader, num_iterations=self.TRAIN_ITERS, batch_size=self.BATCH_SIZE)


    def transform_data(self, data):
        # null transform
        return data

    def get_tpr(self, data, labels):
        m_apply = lambda params, x: self._model.apply(params, self._model._resnet_state, x)
        return get_tpr_pure(self.key(), self.transform_data(data), labels, self._model.params, m_apply).item()

    def get_accuracy(self, data, labels):
        m_apply = lambda params, x: self._model.apply(params, self._model._resnet_state, x)
        return get_accuracy(self.key(), self.transform_data(data), labels, self._model.params, m_apply).item()

    def get_predictions(self, data):
        return self._model.predict(self.transform_data(data))

    def get_balanced_accuracy(self, data, labels):
        m_apply = lambda params, x: self._model.apply(params, self._model._resnet_state, x)
        tpr = self.get_tpr(data, labels)
        tnr = get_tnr_pure(self.key(), self.transform_data(data), labels, self._model.params, m_apply)

        return ((tpr + tnr) / 2).item()

def bnn_pruning_results_mlp_student(bnn: ImageBNN, train_loader, test_loader,
                                    dataset_name, bnn_type, BATCH_SIZE=500,
                                    TRAIN_ITERS=50000, NUM_VERIF=150):
    bnn_stats_gen = ImageBNNStatsGenerator(bnn, train_loader, test_loader, BATCH_SIZE, key=bnn.next_key())

    all_results = []
    bnn_stats = bnn_stats_gen.get_all_stats(test_loader)

    train_loader_idx = get_dl_with_idx(batch_size=BATCH_SIZE)
    test_loader_idx = get_dl_with_idx(batch_size=BATCH_SIZE, split='test')

    final_model, weight_mlp_stats = weighted_training_image(bnn, train_loader_idx, test_loader_idx,
                                                            TRAIN_ITERS, BATCH_SIZE, NUM_VERIF, dataset_hparams[dataset_name])

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

def get_celeba_data(target_id, batch_size):
    return get_celeba_dl(target_id, batch_size, 'train'), get_celeba_dl(target_id, batch_size, 'test')

if __name__ == "__main__":
    rng_seq = hk.PRNGSequence(7)
    LR = 0.001

    datasets = [
        ('celeba', get_celeba_data, 2, 64)
    ]

    all_results = []
    all_baseline_results = []

    with open(os.path.join(CELEBA_PATH, 'data_frame.pickle'), 'rb') as handle:
        celeba_df = pickle.load(handle)

    celeba_train_df = celeba_df['train']
    celeba_train_folder = os.path.join(CELEBA_PATH, 'split/train')

    celeba_test_df = celeba_df['test']
    celeba_test_folder = os.path.join(CELEBA_PATH, 'split/test')

    for ds_name, ds_getr_fn, prot_attr_idx, batch_size in tqdm(datasets, desc=' datasets', position=0):
        # print('=' * 10)
        # print(ds_name)
        # print('=' * 10)
        if ds_name == 'celeba':
            # celeba_dataset_test = get_celeba_dataset(celeba_test_df, celeba_test_folder, 2)
            celeba_dataset_train = get_celeba_dataset(celeba_train_df, celeba_train_folder, prot_attr_idx)

        for i in tqdm(range(5), desc='repetition', position=1, leave=False):
            # train_data, train_labels, _, _, test_data, test_labels = ds_getr_fn()
            train_loader, test_loader = ds_getr_fn(prot_attr_idx, batch_size)
            try:
                fmu_stats = get_fmu_stats(celeba_train_df, celeba_train_folder, celeba_test_df, celeba_test_folder,
                              ds_name, batch_size=batch_size)

                fairbatch_stats = get_fairbatch_stats(celeba_dataset_train, celeba_train_df, celeba_train_folder,
                                                     celeba_test_df, celeba_test_folder, ds_name, batch_size=batch_size * 5)

            except Exception as e:
               print(e)
               continue

            baseline_results = merge_dicts(
            dict_add_pre('fmu', fmu_stats),
            dict_add_pre('fairbatch', fairbatch_stats)
            )
            # baseline_results['dataset'] = ds_name
            all_baseline_results.append([baseline_results])

            # imBNN = ImageBNNwDistributionShift(next(rng_seq), next(iter(train_loader))[0].shape, 2, learning_rate=LR, verbose=False)
            # results = bnn_pruning_results_mlp_student(imBNN, train_loader, test_loader, 'celeba', 'LabelShift', batch_size, NUM_VERIF=100)
            # all_results.append(results)

            imBNN = ImageBNNwProtDistributionShift(next(rng_seq), next(iter(train_loader))[0].shape, 2, learning_rate=LR, verbose=True)
            results = bnn_pruning_results_mlp_student(imBNN, train_loader, test_loader, 'celeba', 'AttrLabelShift', batch_size, NUM_VERIF=100)
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

