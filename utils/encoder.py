import numpy as np


class BockdollEncoding:
    """
    Encoder for cateogrical variables (features), except for one hot encoding

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
    def clip(x):
        return np.clip(x, 1e-4, 0.99)

    def smoothingProb(self, n, p, y):
        smoove = 1 / (1 + np.exp(-(n - self.min_sample) / self.smoothing))
        smooth = y.mean() * (1 - smoove) + p * smoove
        return smooth

    def logodds(self, p):
        return self.log(p / (1 - p))

    def gini_imp(self, p):
        return 1 - p ** 2 - (1 - p) ** 2

    def entropy(self, p):
        return -p * self.log(p) - (1 - p) * self.log(1 - p)

    def focal_loss(self, p):
        return -(
            self.alpha * self.log(p) * np.power((1.0 - p), self.gamma)
            + (1.0 - self.alpha) * self.log(1.0 - p) * np.power(p, self.gamma)
        )

    def entropy_geom(self, p):
        return (-p * self.log(p) - (1 - p) * self.log(1 - p)) / p

    def beta_bias(self, p, n):  # https://en.wikipedia.org/wiki/Geometric_distribution
        bias = p * (1 - p) / n
        return p - bias

    def fit(self, X, y, type_enc="prob0"):
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
                    .agg(lambda x: self.smoothingProb(x.count(), x.mean(), self.y))
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
                    .agg(lambda x: self.smoothingProb(x.count(), x.mean(), self.y))
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
                        smooth_mean=lambda x: self.smoothingProb(
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
                        smooth_mean=lambda x: self.smoothingProb(
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
                    self.mapping[cols].apply(lambda x: self.logodds(self.clip(x)))
                )
            return X.fillna(X.mean())

        if type_enc == "gini imp":
            for cols in X.columns:
                X[cols] = X[cols].map(
                    self.mapping[cols].apply(lambda x: self.gini_imp(x))
                )
            return X.fillna(X.mean())

        if type_enc == "entropy":
            for cols in X.columns:
                X[cols] = X[cols].map(
                    self.mapping[cols].apply(lambda x: self.entropy(self.clip(x)))
                )
            return X.fillna(X.mean())

        if type_enc == "focal loss":
            for cols in X.columns:
                X[cols] = X[cols].map(
                    self.mapping[cols].apply(lambda x: self.focal_loss(self.clip(x)))
                )
            return X.fillna(X.mean())

        if type_enc == "entropy geom":
            for cols in X.columns:
                X[cols] = X[cols].map(
                    self.mapping[cols].apply(lambda x: self.entropy_geom(self.clip(x)))
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
