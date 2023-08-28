import numpy as np
import pandas as pd

from utils.jax.models.bnn import BNN
from utils.jax.models.image_bnn import ImageBNN
import numpy as onp
from modules.model_components import WeightedMLP
from modules.model_components import incompetent_get
from aif360.sklearn.metrics import statistical_parity_difference, average_odds_difference, \
    equal_opportunity_difference, generalized_entropy_error, consistency_score
from sklearn.metrics import balanced_accuracy_score
import plotly.express as px
from utils.common import save_pickle
import os
import pickle
from tqdm import tqdm


def finetuning(model: BNN, train_data, train_labels, test_data, test_labels,
               filter_fn,
               num_minibatches, num_iterations, extra, n,
               verbose=False):
    """
    Create a head which fine-tunes the BNN output to perform well on ambiguous samples
    :param model: The model for uncertainty prediction
    :param train_data: The data for training
    :param train_labels: Training labels
    :return:
    """
    if verbose:
        print("Finetuning the BNN")

    # get the model parameters
    frozen_keys = list(model._mlp_params.keys())[:-1]

    extra['frozen_layers'] = frozen_keys

    # Get the train set with high uncertainty
    certain_idxs = filter_fn(train_data)
    uncertain_idxs = onp.setxor1d(np.arange(train_data.shape[0]), certain_idxs)

    if len(uncertain_idxs) > 100:
        # Fine tune the model
        model.train((train_data[uncertain_idxs], train_labels[uncertain_idxs]), (test_data, test_labels),
                    num_minibatches, num_iterations, extra)

    return model, len(uncertain_idxs)



def weighted_training(model: BNN, train_data, train_labels, test_data, test_labels, prot_attr_idx, num_iterations,
                      batch_size, n, hparams):
    """
    Train a new model such that the high uncertainty samples are weighed more than the low uncertainty samples.
    :param model:
    :param train_data:
    :param train_labels:
    :param test_data:
    :param test_labels:
    :return:
    """
    lr = hparams['lr']
    num_layers = hparams['num_layers']
    hidden_dim = hparams['hidden_dim']
    dropout = hparams['dropout']
    num_iterations = hparams['num_iterations']


    non_prot_attrs = [x for x in onp.arange(train_data.shape[1]) if x!=prot_attr_idx]
    uncertainties = model.get_aleatoric_uncertainty_v2(train_data[:, non_prot_attrs], n)
    # rw_scaler, reweight_model, rw_weights = get_reweight_model_german(train_data, train_labels)
    post_model = WeightedMLP(model.next_key(), train_data[:, 1:].shape, num_layers, hidden_dim, dropout, lr=lr, verbose=False)
    post_model.fit(train_data, train_labels, num_iterations, # num_iterations,
                   batch_size,
                   train_data, test_data, train_labels, test_labels, {
                        'uncertainties': uncertainties,
                        'prot_attr_idx': prot_attr_idx,
                        'rw': None
                   })

    # TODO Remove German Dataset reweighting
    ## =====Reweight Method=====
    # rw_scaler, reweight_model, rw_weights = get_reweight_model_german(train_data, train_labels)
    ## =========================
    y_true = pd.DataFrame(test_labels)
    prot = np.array(test_data[:, prot_attr_idx]).flatten()
    y_preds = post_model.predict(test_data)
    y_preds = np.array(y_preds)

    tus = model.get_aleatoric_uncertainty_v2(test_data[:, non_prot_attrs], n)
    tus = enumerate(tus.tolist())
    tus = sorted(tus, key=lambda x: x[1])
    tus = [x[0] for x in tus]

    prun = lambda x: (test_data[tus[:int(x*len(test_labels))]], test_labels[tus[:int(x*len(test_labels))]], y_preds[tus[:int(x*len(test_labels))]], prot[tus[:int(x*len(test_labels))]])


    def get_accuracy(model, data, labels):
        preds = model.predict(data)
        return (preds == labels).astype(float).mean()

    def get_bal_accuracy(data, labels):
        pass

    # Calculate the metrics using reweighting scheme


    ig = incompetent_get
    # eod = ig(equal_opportunity_difference)
    # bacc = ig(balanced_accuracy_score)
    # acc = ig(get_accuracy)
    # xs = np.linspace(0, 1, 11)[1:]
    # y_eods = []
    # y_baccs = []
    # df_plot = []
    # for x in xs:
    #     i_data, i_lbl, i_pred, i_prot = prun(x)
    #     eod_x = eod(pd.DataFrame(i_lbl), i_pred, prot_attr=i_prot)
    #     bacc_x = bacc(pd.DataFrame(i_lbl), i_pred)
    #     y_eods.append(eod_x)
    #     y_baccs.append(bacc_x)
    #     df_plot.append({
    #         'Pruning': 1 - x,
    #         'Equality Odds Difference': abs(eod_x),
    #         'bacc': bacc_x,
    #         'Accuracy': acc(post_model, i_data, i_lbl)
    #     })
    #
    # save_pickle(df_plot, 'result_analysis/german_prune_stats.pkl')
    # df_plot = pd.DataFrame(df_plot)
    # fig = px.line(df_plot, x='Pruning', y='Accuracy', template='ggplot2')
    # fig.update_layout(font=dict(size=15))
    # fig.show()



    stats = {
        'statistical_parity': ig(statistical_parity_difference)(y_true, y_preds, prot_attr=prot),
        'avg_odds_diff':                ig(average_odds_difference)(y_true, y_preds, prot_attr=prot),
        'equal_opportunity_diff':       ig(equal_opportunity_difference)(y_true, y_preds, prot_attr=prot),
        'generalized_entropy_error':    ig(generalized_entropy_error)(y_true.to_numpy().flatten(), y_preds),
        'consistency_score':            ig(consistency_score)(np.array(test_data[:, non_prot_attrs]), y_preds),
        'accuracy':                     ig(get_accuracy)(post_model, test_data, test_labels),
        'bal_accuracy':                 ig(balanced_accuracy_score)(y_true, y_preds)
        # 'EO':                           ig(self.get_equalized_odds)(data_fil, labels_fil, protected_attr_idx)
    }

    return post_model, stats


def weighted_training_image(model: ImageBNN, train_loader, test_loader, num_iterations,
                            batch_size, n, hparams):
    """
    Train a new model such that the high uncertainty samples are weighed more than the low uncertainty samples.
    :param model:
    :param train_data:
    :param train_labels:
    :param test_data:
    :param test_labels:
    :return:
    """
    lr = hparams['lr']
    num_iterations = hparams['num_iterations']
    model.set_training(False)
    uncertainties = onp.zeros(len(train_loader.dataset))
    test_uncertainties = onp.zeros(len(test_loader.dataset))

    if os.path.exists('bnn_unc.pkl'):
        with open('bnn_unc.pkl', 'rb') as unc_file:
            uncertainties = pickle.load(unc_file)
    else:
        for i, x, y, a in tqdm(iter(train_loader)):
            i, x, y = i.numpy(), x.numpy(), y.numpy().astype(int)
            m_unc = model.get_aleatoric_uncertainty_v2(x, n)
            uncertainties[i] = np.array(m_unc)
        with open('bnn_unc.pkl', 'wb') as unc_file:
            pickle.dump(uncertainties, unc_file)


    # Get uncertainties for the test data
    if os.path.exists('bnn_unc_test.pkl'):
        with open('bnn_unc_test.pkl', 'rb') as unc_file:
            test_uncertainties = pickle.load(unc_file)
    else:
        for i, x, y, a in tqdm(iter(test_loader)):
            i, x, y = i.numpy(), x.numpy(), y.numpy().astype(int)
            m_unc = model.get_aleatoric_uncertainty_v2(x, n)
            test_uncertainties[i] = np.array(m_unc)
        with open('bnn_unc_test.pkl', 'rb') as unc_file:
            pickle.dump(test_uncertainties, unc_file)

    # uncertainties = jnp.concatenate(uncertainties)

    # uncertainties = model.get_aleatoric_uncertainty_v2(train_data[:, non_prot_attrs], n)
    # rw_scaler, reweight_model, rw_weights = get_reweight_model_german(train_data, train_labels)
    train_iter = infinite_dataloader(train_loader)
    test_iter = infinite_dataloader(test_loader)

    _, tmp_x, _, _ = next(train_iter)

    post_model = WeightedResnetClassifierV3(tmp_x.shape, lr=lr, key=model.next_key(), verbose=True)
    model_utils = post_model.fit(train_iter, test_iter, num_iterations, batch_size, {
        'uncertainties': uncertainties,
        'uncertainties_test': test_uncertainties
    })

    # TODO Remove German Dataset reweighting
    ## =====Reweight Method=====
    # rw_scaler, reweight_model, rw_weights = get_reweight_model_german(train_data, train_labels)
    ## =========================
    post_model.set_training(False)

    # Get all the test labels
    test_labels = []
    prot = []
    y_preds = []
    get_beta = model_utils['get_beta']
    for i, x, y, a in tqdm(iter(test_loader), total=len(test_loader)):
        i, x, y, a = i.numpy(), x.numpy(), y.numpy().astype(int), a.numpy().astype(int)
        x_unc = {
            'img': x,
            'unc': get_beta(test_uncertainties[i])
        }
        test_labels.append(y)
        prot.append(a)
        y_preds.append(onp.array(post_model.predict(x_unc)))

    test_labels = onp.concatenate(test_labels)
    prot = onp.concatenate(prot)
    y_preds = onp.concatenate(y_preds)

    y_true = pd.DataFrame(test_labels)


    def get_accuracy(model, data, labels):
        preds = model.predict(data)
        return (preds == labels).astype(float).mean()

    ig = incompetent_get
    # eod = ig(equal_opportunity_difference)
    # bacc = ig(balanced_accuracy_score)
    # acc = ig(get_accuracy)
    # xs = np.linspace(0, 1, 11)[1:]
    # y_eods = []
    # y_baccs = []
    # df_plot = []
    # for x in xs:
    #     i_data, i_lbl, i_pred, i_prot = prun(x)
    #     eod_x = eod(pd.DataFrame(i_lbl), i_pred, prot_attr=i_prot)
    #     bacc_x = bacc(pd.DataFrame(i_lbl), i_pred)
    #     y_eods.append(eod_x)
    #     y_baccs.append(bacc_x)
    #     df_plot.append({
    #         'Pruning': 1 - x,
    #         'Equality Odds Difference': abs(eod_x),
    #         'bacc': bacc_x,
    #         'Accuracy': acc(post_model, i_data, i_lbl)
    #     })
    #
    # save_pickle(df_plot, 'result_analysis/german_prune_stats.pkl')
    # df_plot = pd.DataFrame(df_plot)
    # fig = px.line(df_plot, x='Pruning', y='Accuracy', template='ggplot2')
    # fig.update_layout(font=dict(size=15))
    # fig.show()

    stats = {
        'statistical_parity': ig(statistical_parity_difference)(y_true, y_preds, prot_attr=prot),
        'avg_odds_diff':                ig(average_odds_difference)(y_true, y_preds, prot_attr=prot),
        'equal_opportunity_diff':       ig(equal_opportunity_difference)(y_true, y_preds, prot_attr=prot),
        'generalized_entropy_error':    ig(generalized_entropy_error)(y_true.to_numpy().flatten(), y_preds),
        # 'consistency_score':            ig(consistency_score)(np.array(test_data[:, non_prot_attrs]), y_preds),
        # 'accuracy':                     ig(get_accuracy)(post_model, test_data, test_labels),
        'bal_accuracy':                 ig(balanced_accuracy_score)(y_true, y_preds)
        # 'EO':                           ig(self.get_equalized_odds)(data_fil, labels_fil, protected_attr_idx)
    }
    print(stats)

    return post_model, stats
