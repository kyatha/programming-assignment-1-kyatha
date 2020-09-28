import numpy as np
from sklearn.metrics import r2_score

class RidgeRegression:
    """
    Parameters
    ----------
    alpha: regularization strength
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        pass

    def train(self, x_train, y_train):
        """Receive the input training data, then learn the model.

        Parameters
        ----------
        x_train: np.array, shape (num_samples, num_features)
        y_train: np.array, shape (num_samples, )

        Returns
        -------
        None
        """
        self.theta = np.zeros(x_train.shape[1])
        # put your training code here
        return
    def fit(self, x_train, y_train):
        # alias for train
        self.train(x_train, y_train)

    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)

    def predict(self, x_test):
        """Do prediction via the learned model.

        Parameters
        ----------
        x_test: np.array, shape (num_samples, num_features)

        Returns
        -------
        pred: np.array, shape (num_samples, )
        """

        pred = x_test.dot(self.theta)

        return pred