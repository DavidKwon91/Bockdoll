from collections import Counter, defaultdict
import random
import numpy as np

#%%


class stratified_group_k_fold:
    def __init__(self, n_splits=5, num_class=2):
        self.n_splits = n_splits
        self.num_class = num_class

    def __repr__(self):
        return self.__class__.__name__ + "(n_splits = %s, num_class = %s)" % (
            self.n_splits,
            self.num_class,
        )

    def split(self, X, y, groups, seed=None):
        y_counts_per_group = defaultdict(lambda: np.zeros(self.num_class))
        y_distr = Counter()

        for label, g in zip(y, groups):
            y_counts_per_group[g][label] += 1
            y_distr[label] += 1

        y_counts_per_fold = defaultdict(lambda: np.zeros(self.num_class))
        groups_per_fold = defaultdict(set)

        def std_y_prob_per_fold(
            y_counts, which_fold, num_class=self.num_class, total_fold=self.n_splits
        ):
            y_counts_per_fold[which_fold] += y_counts
            std_per_label = []
            for label in range(num_class):
                label_std = np.std(
                    [
                        y_counts_per_fold[i][label] / y_distr[label]
                        for i in range(total_fold)
                    ]
                )
                std_per_label.append(label_std)
            y_counts_per_fold[which_fold] -= y_counts
            return np.mean(std_per_label)

        groups_and_y_counts = list(y_counts_per_group.items())
        random.Random(seed).shuffle(groups_and_y_counts)

        for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
            best_fold = None
            min_eval = None
            for i in range(self.n_splits):
                fold_eval = std_y_prob_per_fold(
                    y_counts,
                    which_fold=i,
                    num_class=self.num_class,
                    total_fold=self.n_splits,
                )

                if min_eval is None or fold_eval < min_eval:
                    # finding the minimum std of y_prob in folds
                    min_eval = fold_eval
                    best_fold = i
                # if the fold that has the minimum st do fy_prob in each fold found, counts and gorups in it
            y_counts_per_fold[best_fold] += y_counts
            groups_per_fold[best_fold].add(g)

        all_groups = set(groups)

        for i in range(self.n_splits):
            fold_groups = all_groups - groups_per_fold[i]
            oof_groups = groups_per_fold[i]

            fold_ind = [i for i, g in enumerate(groups) if g in fold_groups]
            oof_ind = [i for i, g in enumerate(groups) if g in oof_groups]

            yield fold_ind, oof_ind

    def get_n_splits(self):
        return self.n_splits


# %%
