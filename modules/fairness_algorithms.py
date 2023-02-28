import pandas as pd

from aif360.algorithms import Transformer
from aif360.algorithms.preprocessing import Reweighing, LFR, DisparateImpactRemover, OptimPreproc
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas
from aif360.sklearn.metrics import statistical_parity_difference, average_odds_difference, equal_opportunity_difference,\
    generalized_entropy_error, consistency_score
from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions import get_distortion_adult, get_distortion_german
from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import  OptTools
from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing
from aif360.algorithms.inprocessing.meta_fair_classifier import MetaFairClassifier

# Postprocessing
from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import CalibratedEqOddsPostprocessing
from aif360.algorithms.postprocessing.reject_option_classification import RejectOptionClassification


from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils.jax.common import incompetent_get as ig

from typing import List, Tuple, Dict, Callable
import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

GERMAN_STR = 'German'
ADULT_STR = 'Adult'

opt_preproc_options = {
    ADULT_STR: {
        'distortion_fun': get_distortion_adult,
        'epsilon': .05,
        'clist': [0.99, 1.99, 2.99],
        'dlist': [.1, .05, 0]
    },
    GERMAN_STR: {
        'distortion_fun': get_distortion_german,
        'epsilon': .05,
        'clist': [0.99, 1.99, 2.99],
        'dlist': [.1, .05, 0]
    }
}


data_load_fns = {
    GERMAN_STR: load_preproc_data_german,
    ADULT_STR: load_preproc_data_adult
}
def dataset_from_np(data, labels, load_fn: Callable, prot_attr_name: str='sex', prot_attr_idx: int=None):
    dataset = load_fn([prot_attr_name])
    mms_label = MinMaxScaler(feature_range=(min(dataset.labels), max(dataset.labels)))
    mms = MinMaxScaler()
    dataset.features = data
    dataset.labels = mms_label.fit_transform(labels.reshape(-1, 1))
    dataset.scores = mms.fit_transform(dataset.labels)
    dataset.protected_attributes = data[:, prot_attr_idx].reshape(-1, 1)
    dataset.instance_weights = np.ones((len(labels), 1))
    dataset.instance_names = [str(x) for x in np.arange(len(labels)).tolist()]
    return dataset

def german_from_np(data, labels, prot_attr_name='sex', prot_attr_idx=1):
    german_dataset = dataset_from_np(data, labels, load_preproc_data_german, prot_attr_name, prot_attr_idx)
    return german_dataset

def adult_from_np(data, labels, prot_attr_name='sex', prot_attr_idx=1):
    adult_dataset = dataset_from_np(data, labels, load_preproc_data_adult, prot_attr_name, prot_attr_idx)
    return adult_dataset


def get_priv_groups(dataset: str, prot_attr_idx: int):
    if dataset == GERMAN_STR:
        if prot_attr_idx == 1:
            privileged_groups = [{'sex': 1}]
            unprivileged_groups = [{'sex': 0}]
            prot_attr_name = 'sex'
        elif prot_attr_idx == 0:
            privileged_groups = [{'age': 1}]
            unprivileged_groups = [{'age': 0}]
            prot_attr_name = 'age'

    if dataset == ADULT_STR:
        if prot_attr_idx == 1:
            privileged_groups = [{'sex': 1}]
            unprivileged_groups = [{'sex': 0}]
            prot_attr_name = 'sex'
        else:
            privileged_groups = [{'race': 1}]
            unprivileged_groups = [{'race': 0}]
            prot_attr_name = 'race'

    return  privileged_groups, unprivileged_groups, prot_attr_name


def get_reweight_model(train_data, train_labels, prot_attr_idx, dataset_name: str):
    privileged_groups, unprivileged_groups, prot_attr_name = get_priv_groups(dataset_name, prot_attr_idx)
    dataset = dataset_from_np(train_data, train_labels, data_load_fns[dataset_name], prot_attr_name, prot_attr_idx)

    scaler = StandardScaler()

    RW = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    RW.fit(dataset)
    dataset_transf_train = RW.transform(dataset)
    iw = dataset_transf_train.instance_weights.ravel()
    iw = (iw * train_data.shape[0]) / iw.sum()
    scaler.fit(train_data)
    lmod = LogisticRegression()
    lmod.fit(scaler.transform(train_data), train_labels.astype(float), sample_weight=iw)
    return scaler, lmod, iw


def get_lfr_model(train_data, train_labels, prot_attr_idx, dataset_name):
    privileged_groups, unprivileged_groups, prot_attr_name = get_priv_groups(dataset_name, prot_attr_idx)
    dataset = dataset_from_np(train_data, train_labels, data_load_fns[dataset_name], prot_attr_name, prot_attr_idx)
    scaler = StandardScaler()
    dataset.features = scaler.fit_transform(dataset.features)

    TR = LFR(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups,
             k=10, Ax=0.1, Ay=1.0, Az=2.)
    TR = TR.fit(dataset, maxiter=5000, maxfun=5000)
    return scaler, TR

def get_optim_model(train_data, train_labels, prot_attr_idx, dataset_name):
    # create dataset from the train data
    privileged_groups, unprivileged_groups, prot_attr_name = get_priv_groups(dataset_name, prot_attr_idx)
    dataset = dataset_from_np(train_data, train_labels, data_load_fns[dataset_name], prot_attr_name, prot_attr_idx)

    OP = OptimPreproc(OptTools, opt_preproc_options[dataset_name])
    OP = OP.fit(dataset)
    dataset_trans_train = OP.transform(dataset)
    dataset_trans_train = dataset.align_datasets(dataset_trans_train)
    return dataset_trans_train, OP


def get_calibeo_stats(train_data, train_labels, test_data, test_labels, dataset_name, prot_attr_idx):
    privileged_groups, unprivileged_groups, prot_attr_name = get_priv_groups(dataset_name, prot_attr_idx)
    dataset_orig_train = dataset_from_np(train_data, train_labels, data_load_fns[dataset_name],
                                         prot_attr_name, prot_attr_idx)
    dataset_orig_test = dataset_from_np(test_data ,test_labels * 0., data_load_fns[dataset_name],
                                        prot_attr_name, prot_attr_idx)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(dataset_orig_train.features)
    y_train = dataset_orig_train.labels.ravel()

    dataset_orig_train.features = X_train
    dataset_orig_test.features = scaler.transform(dataset_orig_test.features)

    lmod = LogisticRegression()
    lmod.fit(X_train, y_train)

    train_preds = lmod.predict(X_train).flatten()

    dataset_orig_train_preds = dataset_orig_train.copy(deepcopy=True)
    dataset_orig_train_preds.labels = train_preds.reshape(-1, 1)

    dataset_orig_test_preds = dataset_from_np(dataset_orig_test.features, lmod.predict(dataset_orig_test.features).flatten(),
                                              data_load_fns[dataset_name], prot_attr_name, prot_attr_idx)

    cpp = CalibratedEqOddsPostprocessing(privileged_groups = privileged_groups,
                                         unprivileged_groups=unprivileged_groups)

    cpp = cpp.fit(dataset_orig_train, dataset_orig_train_preds)

    mms = MinMaxScaler()
    dataset_transf_test_pred = cpp.predict(dataset_orig_test_preds)
    y_preds = mms.fit_transform(dataset_transf_test_pred.labels).flatten()

    return get_stats_general(prot_attr_idx, test_data, test_labels, y_preds)

def get_mf_model(train_data, train_labels, prot_attr_idx, dataset_name):
    _, _, prot_attr_name = get_priv_groups(dataset_name, prot_attr_idx)
    dataset = dataset_from_np(train_data, train_labels, data_load_fns[dataset_name], prot_attr_name, prot_attr_idx)
    scaler = StandardScaler()
    dataset.features = scaler.fit_transform(dataset.features)

    debiased_model = MetaFairClassifier(tau=.7, sensitive_attr=prot_attr_name, type='fdr').fit(dataset)
    return scaler, debiased_model

def get_mf_stats(train_data, train_labels, test_data, test_labels, dataset, prot_attr_idx):
    _, _, prot_attr_name = get_priv_groups(dataset, prot_attr_idx)
    scaler, model = get_mf_model(train_data, train_labels, prot_attr_idx, dataset)
    dataset_test = dataset_from_np(test_data, test_labels, data_load_fns[dataset], prot_attr_name, prot_attr_idx)
    dataset_test.features = scaler.transform(dataset_test.features)

    dataset_debiasing_test = model.predict(dataset_test)
    mms = MinMaxScaler()
    y_preds = mms.fit_transform(dataset_debiasing_test.labels).flatten()
    return get_stats_general(prot_attr_idx, test_data, test_labels, y_preds)


def get_adb_model(train_data, train_labels, prot_attr_idx, dataset_name):
    privileged_groups, unprivileged_groups, prot_attr_name = get_priv_groups(dataset_name, prot_attr_idx)
    dataset = dataset_from_np(train_data, train_labels, data_load_fns[dataset_name], prot_attr_name, prot_attr_idx)
    scaler = StandardScaler()
    dataset.features = scaler.fit_transform(dataset.features)

    sess = tf.Session()
    debiased_model = AdversarialDebiasing(privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups,
                                          scope_name='debiased_classifier', debias=True, sess=sess)

    debiased_model.fit(dataset)

    return sess, scaler, debiased_model


def get_adb_stats(train_data, train_labels, test_data, test_labels, dataset, prot_attr_idx):
    _, _, prot_attr_name = get_priv_groups(dataset, prot_attr_idx)
    sess, scaler, ADB = get_adb_model(train_data, train_labels, prot_attr_idx, dataset)
    dataset_test = dataset_from_np(test_data, test_labels * 0., data_load_fns[dataset], prot_attr_name, prot_attr_idx)
    dataset_test.features = scaler.transform(dataset_test.features)

    dataset_debiasing_test = ADB.predict(dataset_test)
    mms = MinMaxScaler()
    y_preds = mms.fit_transform(dataset_debiasing_test.labels).flatten()

    sess.close()
    tf.reset_default_graph()

    return get_stats_general(prot_attr_idx, test_data, test_labels, y_preds)


def get_roc_model(train_data, train_labels, prot_attr_idx, dataset_name):
    privileged_groups, unprivileged_groups, prot_attr_name = get_priv_groups(dataset_name, prot_attr_idx)
    dataset = dataset_from_np(train_data, train_labels, data_load_fns[dataset_name], prot_attr_name, prot_attr_idx)
    scaler = StandardScaler()
    dataset.features = scaler.fit_transform(dataset.features)
    lmod = LogisticRegression()
    lmod.fit(dataset.features, dataset.labels.ravel())
    y_train_pred = lmod.predict(dataset.features)

    dataset_orig_train_pred = dataset.copy(deepcopy=True)
    dataset_orig_train_pred.labels = y_train_pred

    ROC = RejectOptionClassification(
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups,
        low_class_thresh=0.01, high_class_thresh=0.99,
        num_class_thresh=100, num_ROC_margin=50,
        metric_name='Statistical parity difference',
        metric_ub=0.05, metric_lb=-0.05
    )
    ROC = ROC.fit(dataset, dataset_orig_train_pred)

    return scaler, ROC

def get_roc_stats(train_data, train_labels, test_data, test_labels, dataset_name, prot_attr_idx):
    _, _, prot_attr_name = get_priv_groups(dataset_name, prot_attr_idx)
    scaler, ROC = get_roc_model(train_data, train_labels, prot_attr_idx, dataset_name)
    dataset_test = dataset_from_np(test_data, test_labels * 0., data_load_fns[dataset_name], prot_attr_name, prot_attr_idx)
    dataset_test.features = scaler.transform(dataset_test.features)
    dataset_test_pred = ROC.predict(dataset_test)

    mms = MinMaxScaler()
    y_preds = mms.fit_transform(dataset_test_pred.labels).flatten()
    return get_stats_general(prot_attr_idx, test_data, test_labels, y_preds)

def get_stats_general(prot_attr_idx, test_data, test_labels, y_preds):
    non_prot_attrs = [x for x in np.arange(test_data.shape[1]) if x != prot_attr_idx]
    y_true = pd.DataFrame(test_labels)
    prot = np.array(test_data[:, prot_attr_idx]).flatten()
    stats = {
        'statistical_parity': ig(statistical_parity_difference)(y_true, y_preds, prot_attr=prot),
        'avg_odds_diff': ig(average_odds_difference)(y_true, y_preds, prot_attr=prot),
        'equal_opportunity_diff':       ig(equal_opportunity_difference)(y_true, y_preds, prot_attr=prot),
        'generalized_entropy_error':    ig(generalized_entropy_error)(y_true.to_numpy().flatten(), y_preds),
        'consistency_score':            ig(consistency_score)(np.array(test_data[:, non_prot_attrs]), y_preds),
        'accuracy':                     (y_preds == test_labels).astype(float).mean(),
        'bal_accuracy':                 ig(balanced_accuracy_score)(y_true, y_preds)
    }

    return stats

def get_lfr_stats(train_data, train_labels, test_data, test_labels, dataset_name, prot_attr_idx):
    _, _, prot_attr_name = get_priv_groups(dataset_name, prot_attr_idx)
    scaler, lfr = get_lfr_model(train_data, train_labels, prot_attr_idx, dataset_name)
    test_data_scaled = scaler.transform(test_data)
    dataset = dataset_from_np(test_data, test_labels * 0., data_load_fns[dataset_name], prot_attr_name, prot_attr_idx)
    dataset.features = test_data_scaled
    dataset = lfr.transform(dataset)

    mms = MinMaxScaler()
    y_preds = mms.fit_transform(dataset.labels).flatten()

    return get_stats_general(prot_attr_idx, test_data, test_labels, y_preds)

def get_reweight_stats(train_data, train_labels, test_data, test_labels, dataset: str, prot_attr_idx: int) -> Dict:
    scaler, trained_model, _ = get_reweight_model(train_data, train_labels, prot_attr_idx, dataset)
    y_preds = trained_model.predict(scaler.transform(test_data))
    mms = MinMaxScaler()
    y_preds = np.array(y_preds)
    y_preds = mms.fit_transform(y_preds.reshape(-1, 1)).flatten()
    return get_stats_general(prot_attr_idx, test_data, test_labels, y_preds)


def get_opt_stats(train_data, train_labels, test_data, test_labels, dataset, prot_attr_idx):
    _, _, prot_attr_name = get_priv_groups(dataset, prot_attr_idx)
    dataset_trans_train, OP = get_optim_model(train_data, train_labels, prot_attr_idx, dataset)
    dataset_test = dataset_from_np(test_data, test_labels * 0., data_load_fns[dataset], prot_attr_name, prot_attr_idx)
    dataset_transf_test = OP.transform(dataset_test, transform_Y=True)
    dataset_transf_test = dataset_test.align_datasets(dataset_transf_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(dataset_trans_train.features)
    y_train = dataset_trans_train.labels.ravel()

    lmod = LogisticRegression()
    lmod.fit(X_train, y_train)

    mms = MinMaxScaler()
    X_test = scaler.transform(dataset_transf_test.features)
    y_preds = lmod.predict(X_test)
    y_preds = mms.fit_transform(y_preds.reshape(-1, 1)).flatten()

    return get_stats_general(prot_attr_idx, test_data, test_labels, y_preds)






