import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from six.moves import xrange


class DataSet:
    def __init__(self,
                 X,
                 seed=None):
        """Construct a DataSet.
        Seed arg provides for convenient deterministic testing.
        """
        seed1, seed2 = random_seed.get_seed(seed)
        # If op level seed is not set, use whatever graph level seed is returned
        np.random.seed(seed1 if seed is None else seed2)
        # dtype = dtypes.as_dtype(dtype).base_dtype
        # if dtype not in (dtypes.uint8, dtypes.float32):
        #     raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
        #                     dtype)
        self.seed = seed1 if seed is None else seed2
        self._num_examples = X.shape[0]

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        # if reshape:
        #     assert X.shape[3] == 1
        #     X = X.reshape(X.shape[0],
        #                   X.shape[1] * X.shape[2])
        self._X = X
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def X(self):
        return self._X

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""

        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._X = self.X[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            X_rest_part = self._X[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._X = self.X[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            X_new_part = self._X[start:end]
            return np.concatenate((X_rest_part, X_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._X[start:end]
