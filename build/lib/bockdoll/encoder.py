import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold

from .folder import stratified_group_k_fold


def fold_split(x, y, fold, group=None):
    # fold e.g) StratifiedKFold(n_splits = 5, random_state = 2020, shuffle = True)
    # fold e.g) GroupKFold(n_splits = 5, random_state = 2020, shuffle = True)

    if fold.__class__ in (GroupKFold, stratified_group_k_fold):
        if group is None:
            raise ValueError("Specify groups for group k-fold")
        else:
            fold_split = fold.split(x, y, groups=group)
    else:
        fold_split = fold.split(x, y)

    return fold_split


class BockdollEncoding:
    """
    Encoder for cateogrical variables (features), and this is only for binary task. 

    :::Encoding types:::
    
    Fitting
    -------------------
    'label by prior' 
        label encoder by sorted target mean values

    'label by count'
        label encoder by sorted class count

    'prob0'
        target mean encoder for target == 0 is True

    'prob1'
        target mean encoder for target == 1 is True

    'beta bias'
        refer to : https://en.wikipedia.org/wiki/Geometric_distribution#Parameter_estimation
        'beta bias' must be used in fitting and tranforming as same

    'Beta Encoding'
        refer to : https://mattmotoki.github.io/beta-target-encoding.html
    
    'Beta Encoding_c'
        Biased estimation as the same with 'beta bias' for the probability from 'Beta Encoding'


    Transforming (mapping)
    -------------------
    Fitting values from train, then mapping them into test (or valid)

    'label'
        used for 'label by prior' and 'label by count'

    'prob'
        target probability mean values by class (or group)
        used for 'prob0' or 'prob1'

    'logit'
        logit, or so-called log odds, estimation
        used for 'prob0' or 'prob1'
    
    'gini imp'
        gini impurity, or decision score, used in decision tree
    
    'entropy'
        binary cross-entropy
    
    'focal loss'
        refer to : https://www.kaggle.com/ragnar123/efficientnet-x-384,
                   https://arxiv.org/abs/1708.02002

        modified cross-entropy for imbalanced binary dataset, 
        which is stated as in the paper, "reduces the relative loss for well-classified exmaples, putting more focus on hard, misclassified examples."
    
    'entropy geom'
        simply,
            binary cross-entropy / probability (target mean by class)

    'beta bias'
        used for 'beta bias' only

    'Beta Encoding'
        used for 'Beta Encoding' and 'Beta Encoding_c' only
        
    """

    def __init__(
        self,
        smoothing=0.3,
        min_sample=10,
        gamma=2.0,
        alpha=0.25,
        numleaves=5,
        N_min=10,
        base=2,
    ):
        self.smoothing = smoothing
        self.min_sample = min_sample
        self.numleaves = numleaves
        self.gamma = gamma
        self.alpha = alpha
        self.N_min = N_min
        self.mapping = None

        self.y = None
        self.prior = None

        self.type_fit = set(
            [
                "label by prior",
                "label by count",
                "prob0",
                "prob1",
                "beta bias",
                "beta unbias",
                "Beta Encoding_c",
                "Beta Encoding",
            ]
        )

        self.type_trans = set(
            [
                "label",
                "prob",
                "logit",
                "gini imp",
                "entropy",
                "focal loss",
                "entropy geom",
                "beta bias",
                "Beta Encoding",
            ]
        )

        if base == 2:
            self.log = np.log2
        else:
            self.log = np.log

    @staticmethod
    def _clip(x):
        return np.clip(x, 1e-4, 0.99)

    def _smoothingProb(self, n, p, y):
        smoove = 1 / (1 + np.exp(-(n - self.min_sample) / self.smoothing))
        smooth = y.mean() * (1 - smoove) + p * smoove
        return smooth

    def _logodds(self, p):
        return self.log(p / (1 - p))

    def _gini_imp(self, p):
        return 1 - p ** 2 - (1 - p) ** 2

    def _entropy(self, p):
        return -p * self.log(p) - (1 - p) * self.log(1 - p)

    def _focal_loss(self, p):
        return -(
            self.alpha * self.log(p) * np.power((1.0 - p), self.gamma)
            + (1.0 - self.alpha) * self.log(1.0 - p) * np.power(p, self.gamma)
        )

    def _entropy_geom(self, p):
        return (-p * self.log(p) - (1 - p) * self.log(1 - p)) / p

    def _beta_bias(self, p, n):  # https://en.wikipedia.org/wiki/Geometric_distribution
        bias = p * (1 - p) / n
        return p - bias

    def fit(self, X, y, type_enc="prob0"):
        """
        Fitting

        X : must be pd.DataFrame or pd.Series object
        y : 1-D array or Series
        type_enc : type for encoding
        
        """

        if type(X) in (type(pd.Series(0)), type(pd.DataFrame([]))):
            if type(X) == type(pd.Series(0)):
                X = pd.DataFrame(X)
        else:
            raise ValueError(
                "X input must be pd.Series or pd.DataFrame with proper column name and index"
            )

        if type_enc not in self.type_fit:
            raise ValueError(
                type_enc
                + ", this encoding type is not available, please use one of the encoding in the list, "
                + str(self.type_fit)
            )

        mapping = {}

        # label by count
        if type_enc == "label by count":
            self.y = y
            for cols in X.columns:
                label = (
                    X.assign(target=self.y)
                    .groupby(cols)["target"]
                    .agg(["count", "mean"])
                    .sort_values("count", ascending=False)
                )
                label["label"] = np.arange(len(label)).astype(int)
                mapping[cols] = label["label"]

        # label by prior
        if type_enc == "label by prior":
            self.y = y
            for cols in X.columns:
                label = (
                    X.assign(target=self.y)
                    .groupby(cols)["target"]
                    .agg(["count", "mean"])
                    .sort_values("mean", ascending=False)
                )
                label["label"] = np.arange(len(label)).astype(int)
                mapping[cols] = label["label"]

        # prob 0
        if type_enc == "prob0":
            self.y = y
            for cols in X.columns:
                smooth_mean = (
                    X.assign(target=self.y)
                    .groupby(cols)["target"]
                    .agg(lambda x: self._smoothingProb(x.count(), x.mean(), self.y))
                )
                mapping[cols] = smooth_mean
            self.prior = self.y.mean()

        # prob1
        if type_enc == "prob1":
            self.y = 1 - y
            for cols in X.columns:
                smooth_mean = (
                    X.assign(target=self.y)
                    .groupby(cols)["target"]
                    .agg(lambda x: self._smoothingProb(x.count(), x.mean(), self.y))
                )
                mapping[cols] = smooth_mean
            self.prior = self.y.mean()

        # geometric beta bias
        if type_enc == "beta bias":
            self.y = y
            mapping_new = {}
            for cols in X.columns:
                mapping_new[cols] = (
                    X.assign(target=self.y)
                    .groupby(cols)["target"]
                    .agg(
                        smooth_mean=lambda x: self._smoothingProb(
                            x.count(), x.mean(), self.y
                        ),
                        count=lambda x: x.count(),
                    )
                )
                mapping[cols] = (
                    mapping_new[cols]["smooth_mean"]
                    - mapping_new[cols]["smooth_mean"]
                    * (1 - mapping_new[cols]["smooth_mean"])
                    / mapping_new[cols]["count"]
                )

        # geometric beta unbias
        if type_enc == "beta unbias":
            self.y = y
            mapping_new = {}
            for cols in X.columns:
                mapping_new[cols] = (
                    X.assign(target=self.y)
                    .groupby(cols)["target"]
                    .agg(
                        smooth_mean=lambda x: self._smoothingProb(
                            x.count(), x.mean(), self.y
                        ),
                        count=lambda x: x.count(),
                    )
                )
                mapping[cols] = mapping_new[cols]["smooth_mean"] - mapping_new[cols][
                    "smooth_mean"
                ] * (1 - mapping_new[cols]["smooth_mean"]) / (
                    mapping_new[cols]["count"] - 1
                )

        if type_enc == "Beta Encoding":
            self.y = y
            mapping_new = {}
            X_alpha = {}
            X_beta = {}
            X_BE = {}
            eps = 1e-7
            for cols in X.columns:
                X_Nn = (
                    X.assign(target=self.y)
                    .groupby(cols)["target"]
                    .agg(n=lambda x: x.sum(), N=lambda x: x.count())
                )
                mapping_new[cols] = X_Nn

                # alpha posterior
                X_alpha[cols] = (
                    self.y.mean() * np.maximum(self.N_min - mapping_new[cols]["N"], 0)
                    + mapping_new[cols]["n"]
                )

                # beta posterior
                X_beta[cols] = (
                    (1 - self.y.mean())
                    * np.maximum(self.N_min - mapping_new[cols]["N"], 0)
                    + mapping_new[cols]["N"]
                    - mapping_new[cols]["n"]
                )

                X_BE[cols] = X_alpha[cols] / (X_alpha[cols] + X_beta[cols] + eps)
                mapping[cols] = X_BE[cols]

        if type_enc == "Beta Encoding_c":
            self.y = y
            mapping_new = {}
            X_alpha = {}
            X_beta = {}
            X_BE = {}
            for cols in X.columns:
                X_Nn = (
                    X.assign(target=self.y)
                    .groupby(cols)["target"]
                    .agg(n=lambda x: x.sum(), N=lambda x: x.count())
                )
                mapping_new[cols] = X_Nn

                # alpha posterior
                X_alpha[cols] = (
                    self.y.mean() * np.maximum(self.N_min - mapping_new[cols]["N"], 0)
                    + mapping_new[cols]["n"]
                )

                # beta posterior
                X_beta[cols] = (
                    (1 - self.y.mean())
                    * np.maximum(self.N_min - mapping_new[cols]["N"], 0)
                    + mapping_new[cols]["N"]
                    - mapping_new[cols]["n"]
                )

                X_BE[cols] = X_alpha[cols] / (X_alpha[cols] + X_beta[cols])
                mapping[cols] = (
                    X_BE[cols]
                    - (X_BE[cols] * (1 - X_BE[cols])) / mapping_new[cols]["N"]
                )

        self.mapping = mapping

    def transform(self, X_in, type_enc="prob"):
        """
        Mapping after fitting
        """
        if type(X_in) in (type(pd.Series(0)), type(pd.DataFrame([]))):
            if type(X_in) == type(pd.Series(0)):
                X_in = pd.DataFrame(X_in)
        else:
            raise ValueError(
                "X input must be pd.Series or pd.DataFrame with proper column name and index"
            )

        if type_enc not in self.type_trans:
            raise ValueError(
                type_enc
                + ", this encoding type is not available, please use one of the encoding in the list, "
                + str(self.type_trans)
            )

        X = X_in.copy(deep=True)

        if type_enc == "label":
            for cols in X.columns:
                X[cols] = X[cols].map(self.mapping[cols]).fillna(-1).astype(int)
            return X

        if type_enc == "prob":
            for cols in X.columns:
                X[cols] = X[cols].map(self.mapping[cols])
            return X.fillna(X.mean())

        if type_enc == "logit":
            for cols in X.columns:
                X[cols] = X[cols].map(
                    self.mapping[cols].apply(lambda x: self._logodds(self._clip(x)))
                )
            return X.fillna(X.mean())

        if type_enc == "gini imp":
            for cols in X.columns:
                X[cols] = X[cols].map(
                    self.mapping[cols].apply(lambda x: self._gini_imp(self._clip(x)))
                )
            return X.fillna(X.mean())

        if type_enc == "entropy":
            for cols in X.columns:
                X[cols] = X[cols].map(
                    self.mapping[cols].apply(lambda x: self._entropy(self._clip(x)))
                )
            return X.fillna(X.mean())

        if type_enc == "focal loss":
            for cols in X.columns:
                X[cols] = X[cols].map(
                    self.mapping[cols].apply(lambda x: self._focal_loss(self._clip(x)))
                )
            return X.fillna(X.mean())

        if type_enc == "entropy geom":
            for cols in X.columns:
                X[cols] = X[cols].map(
                    self.mapping[cols].apply(
                        lambda x: self._entropy_geom(self._clip(x))
                    )
                )
            return X.fillna(X.mean())

        if type_enc == "beta bias":
            # must fit as beta bias
            for cols in X.columns:
                X[cols] = X[cols].map(self.mapping[cols])

            return X.fillna(X.mean())

        if type_enc == "Beta Encoding":
            for cols in X.columns:
                X[cols] = X[cols].map(self.mapping[cols])

            return X.fillna(X.mean())

    def fold_encode(
        self, X_in, y, fold, group=None, type_enc_fit="prob0", type_enc_trans="prob"
    ):
        """
        K-fold encoding for preventing overfitting

        Parameters
        ------------
        x : X_train, features
        y : y_train, target (binary)
        
        fold : initiated K-fold class
        groups : groups for group K-fold

        Returns
        ------------
        K-fold target encoded X_train, y_train

            e.g) enc = BockdollEncoding()

                 fold = sklearn.model_selection.KFold(n_splits = 5)
                 fold = sklearn.model_selection.StratifiedKFold(n_splits = 5)        
                 fold = sklearn.model_selection.GroupKFold(n_splits)

                 enc.fold_encode(x, y, fold)
        """

        if type(X_in) in (type(pd.Series(0)), type(pd.DataFrame([]))):
            if type(X_in) == type(pd.Series(0)):
                X_in = pd.DataFrame(X_in)
        else:
            raise ValueError(
                "X input must be pd.Series or pd.DataFrame with proper column name and index"
            )

        if type_enc_fit not in self.type_fit:
            raise ValueError(
                type_enc
                + ", this encoding type is not available, please use one of the encoding in the list, "
                + str(self.type_fit)
            )

        if type_enc_trans not in self.type_trans:
            raise ValueError(
                type_enc
                + ", this encoding type is not available, please use one of the encoding in the list, "
                + str(self.type_trans)
            )

        X_train = X_in.copy(deep=True)
        y_train = y.copy(deep=True)

        if group is not None:
            group = group.copy(deep=True)

        splitted_fold = fold_split(X_train, y_train, fold, group=group)

        oof = pd.DataFrame([])

        for tr_ind, oof_ind in splitted_fold:
            fold_df, oof_df = X_train.iloc[tr_ind], X_train.iloc[oof_ind]
            fold_y, oof_y = y_train.iloc[tr_ind], y_train.iloc[oof_ind]

            self.fit(fold_df, fold_y, type_enc=type_enc_fit)
            oof = oof.append(
                self.transform(oof_df, type_enc=type_enc_trans), ignore_index=False
            )
        X_train = oof.sort_index()
        y_train = y_train.sort_index()

        return X_train, y_train
