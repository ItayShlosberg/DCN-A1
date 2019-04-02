import numpy as np
import sklearn
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from pandas import DataFrame
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, check_X_y
from sklearn.model_selection import KFold

class LinearRegressor(BaseEstimator, RegressorMixin):
    """
    Implements Linear Regression prediction and closed-form parameter fitting.
    """

    def __init__(self, reg_lambda=0.1):
        self.reg_lambda = reg_lambda

    def predict(self, X):
        """
        Predict the class of a batch of samples based on the current weights.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :return:
            y_pred: np.ndarray of shape (N,) where each entry is the predicted
                value of the corresponding sample.
        """
        X = check_array(X)
        check_is_fitted(self, 'weights_')

        # TODO: Calculate the model prediction, y_pred

        y_pred = None
        # ====== YOUR CODE: ======
        y_pred = np.dot(X, self.weights_)
        # ========================

        return y_pred

    def fit(self, X, y):
        """
        Fit optimal weights to data using closed form solution.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :param y: A tensor of shape (N,) where N is the batch size.
        """
        X, y = check_X_y(X, y)

        # TODO: Calculate the optimal weights using the closed-form solution
        # Use only numpy functions.

        w_opt = None
        # ====== YOUR CODE: ======
        # we need to calculate [(X^T * X)^-1] * [X^T] * [y] , with regularization
        res = np.dot(X.T, X)
        res = np.add(res, self.reg_lambda * np.eye(res.shape[0]))
        res = np.linalg.inv(res) # -1 operation

        w_opt = np.dot(res, np.dot(X.T, y))
        # ========================

        self.weights_ = w_opt
        return self

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)


class BiasTrickTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        :param X: A tensor of shape (N,D) where N is the batch size or of shape
            (D,) (which assumes N=1).
        :returns: A tensor xb of shape (N,D+1) where xb[:, 0] == 1
        """
        X = check_array(X)

        # TODO: Add bias term to X as the first feature.

        xb = None
        # ====== YOUR CODE: ======
        xb = np.hstack((np.ones(X.shape[0])[:,np.newaxis],X))
        # ========================

        return xb


class BostonFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Generates custom features for the Boston dataset.
    """
    def __init__(self, degree=2):
        self.degree = degree

        # TODO: Your custom initialization, if needed
        # Add any hyperparameters you need and save them as above
        # ====== YOUR CODE: ======
        # ========================

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transform features to new features matrix.
        :param X: Matrix of shape (n_samples, n_features_).
        :returns: Matrix of shape (n_samples, n_output_features_).
        """
        X = check_array(X)
        # check_is_fitted(self, ['n_features_', 'n_output_features_'])

        # TODO: Transform the features of X into new features in X_transformed
        # Note: You can count on the order of features in the Boston dataset
        # (this class is "Boston-specific"). For example X[:,1] is the second
        # feature ('ZN').

        X_transformed = None
        # ====== YOUR CODE: ======
        X_poly = PolynomialFeatures(degree=self.degree)
        X_transformed = X_poly.fit_transform(X)
        # ========================

        return X_transformed


def top_correlated_features(df: DataFrame, target_feature, n=5):
    """
    Returns the names of features most strongly correlated (correlation is
    close to 1 or -1) with a target feature. Correlation is Pearson's-r sense.

    :param df: A pandas dataframe.
    :param target_feature: The name of the target feature.
    :param n: Number of top features to return.
    :return: A tuple of
        - top_n_features: Sequence of the top feature names
        - top_n_corr: Sequence of correlation coefficients of above features
        Both the returned sequences should be sorted so that the best (most
        correlated) feature is first.
    """

    # TODO: Calculate correlations with target and sort features by it

    # ====== YOUR CODE: ======
    
    data = df[df.columns[df.columns != target_feature]].values
    target = df[target_feature].values

    data_mean0 = data - data.mean(axis=0)
    target_mean0 = target - target.mean()
    
    var_data = np.sqrt((data_mean0 ** 2).sum(axis=0))
    var_target = np.sqrt((target_mean0 ** 2).sum())
    
    ro_mat = (data_mean0 * target_mean0.reshape(-1, 1)).sum(axis=0) / (var_data * var_target)
    
    best_indices = np.abs(ro_mat).argsort()[:-n-1:-1]
    
    top_n_features = df.columns[best_indices]
    top_n_corr = ro_mat[best_indices]
    # ========================

    return top_n_features, top_n_corr


def cv_best_hyperparams(model: BaseEstimator, X, y, k_folds,
                        degree_range, lambda_range):
    """
    Cross-validate to find best hyperparameters with k-fold CV.
    :param X: Training data.
    :param y: Training targets.
    :param model: sklearn model.
    :param lambda_range: Range of values for the regularization hyperparam.
    :param degree_range: Range of values for the degree hyperparam.
    :param k_folds: Number of folds for splitting the training data into.
    :return: A dict containing the best model parameters,
        with some of the keys as returned by model.get_params()
    """

    # TODO: Do K-fold cross validation to find the best hyperparameters
    #
    # Notes:
    # - You can implement it yourself or use the built in sklearn utilities
    #   (recommended). See the docs for the sklearn.model_selection package
    #   http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
    # - If your model has more hyperparameters (not just lambda and degree)
    #   you should add them to the search.
    # - Use get_params() on your model to see what hyperparameters is has
    #   and their names. The parameters dict you return should use the same
    #   names as keys.
    # - You can use MSE or R^2 as a score.

    # ====== YOUR CODE: ======
    best_params = None
    best_loss = -1
    
    for degree in degree_range:
        for lambda_val in lambda_range:
            # create the hyper-parameters for the model
            params = dict(linearregressor__reg_lambda=lambda_val, 
                          bostonfeaturestransformer__degree=degree)

            model.set_params(**params)

            # k-fold split
            kf = KFold(n_splits=k_folds)

            loss = 0
            for fold_ind_train, fold_ind_test in kf.split(X):
                
                # getting the fold values
                train_X_fold = X[fold_ind_train]
                train_y_fold = y[fold_ind_train]
                
                test_X_fold = X[fold_ind_test]
                test_y_fold = y[fold_ind_test]

                model.fit(train_X_fold, train_y_fold)

                y_pred = model.predict(test_X_fold)
                
                mse = np.mean((test_y_fold - y_pred) ** 2)

                loss += mse

            # first loop - initialization
            if best_loss < 0:
                best_params = params
                best_loss = loss
            
            else:
                best_loss = min(loss, best_loss)
                if loss > best_loss:
                    best_params = best_params 
                else:
                    best_params = params
                
    # ========================

    return best_params
