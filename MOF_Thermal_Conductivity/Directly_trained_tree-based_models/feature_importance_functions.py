# Functions adapted from https://inria.github.io/scikit-learn-mooc/python_scripts/dev_features_importance.html

import matplotlib.pyplot as plt
import numpy as np

def get_score_after_permutation(model, X, y, curr_feat):
    """return the score of model when curr_feat is permuted"""

    X_permuted = X.copy()
    col_idx = list(data_notop.keys()).index(curr_feat)
    # permute one column
    X_permuted[:, col_idx] = np.random.permutation(
        X_permuted[:, col_idx]
    )

    permuted_score = model.score(X_permuted, y)
    return permuted_score

def get_feature_importance(model, X, y, curr_feat):
    """compare the score when curr_feat is permuted"""

    baseline_score_train = model.score(X, y)
    permuted_score_train = get_score_after_permutation(model, X, y, curr_feat)

    # feature importance is the difference between the two scores
    feature_importance = baseline_score_train - permuted_score_train
    return feature_importance

def permutation_importance(model, X, y, n_repeats=10):
    """Calculate importance score for each feature."""

    importances = []
    for curr_feat in data_notop.keys():
        list_feature_importance = []
        for n_round in range(n_repeats):
            list_feature_importance.append(
                get_feature_importance(model, X, y, curr_feat)
            )

        importances.append(list_feature_importance)

    return {
        "importances_mean": np.mean(importances, axis=1),
        "importances_std": np.std(importances, axis=1),
        "importances": importances,
    }


def plot_feature_importances(perm_importance_result, feat_name, model_name):
    """bar plot the feature importance"""

    fig, ax = plt.subplots(figsize=(7,5))

    indices = perm_importance_result["importances_mean"].argsort()
    newindices = indices[-10:]
    plt.barh(
        range(len(newindices)),
        perm_importance_result["importances_mean"][newindices],
        xerr=perm_importance_result["importances_std"][newindices],
        #ec='gray',
        height=0.6
    )

    ax.set_yticks(range(len(newindices)))
    _ = ax.set_yticklabels(feat_name[newindices])
    
    plt.title(f'Top 10 most important features in {model_name} model')
    plt.xlabel('Mean Decrease in Accuracy',fontsize=12)
