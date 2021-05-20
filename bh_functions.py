""" Ignore Warnings """
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

""" Imports """
import numpy as np
import pandas as pd
import sobol_seq
from scipy.stats.distributions import entropy
from scipy.stats import ks_2samp
import matplotlib.pylab as plt
import seaborn as sns
from scipy.stats import uniform, randint
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, RandomizedSearchCV # Cross validation
import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score


import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix 






""" surrogate models """
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

# Gaussian Process Regression (Kriging)
# modified version of kriging to make a fair comparison with regard
# to the number of hyperparameter evaluations
from sklearn.gaussian_process import GaussianProcessRegressor


""" cross-validation
Cross validation is used in each of the rounds to approximate the selected 
surrogate model over the data samples that are available. 
The evaluated parameter combinations are randomly split into two sets. An 
in-sample set and an out-of-sample set. The surrogate is trained and its 
parameters are tuned to an in-sample set, while the out-of-sample performance 
is measured (using a selected performance metric) on the out-of-sample set. 
This out-of-sample performance is then used as a proxy for the performance 
on the full space of unevaluated parameter combinations. In the case of the 
proposed procedure, this full space is approximated by the randomly selected 
pool.
"""

""" Defaults Algorithm Tuning Constants """
_N_EVALS = 10
_N_SPLITS = 5
_CALIBRATION_THRESHOLD = 1.00
_P_VALUE_REJECT = 0.05

_TIME_STEPS = 100

""" Functions """
def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)] * a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def evaluate_bh_on_set(parameter_combinations, method, threshold):
    y = np.zeros(parameter_combinations.shape[0])
    num_params = parameter_combinations.shape[1]
    
    real_data = get_BH_real_data()
    
    if num_params == 9:
        for i, (trend_2, trend_1, switching_parameter, alpha, bias_2, weight_past_profits, bias_1, rational_expectation_cost, risk_free_return) in enumerate(parameter_combinations):
            response = BH_abm(trend_2 = trend_2,
                              trend_1 = trend_1,
                              switching_parameter = switching_parameter,
                              alpha = alpha,
                              bias_2 = bias_2,
                              weight_past_profits = weight_past_profits,
                              bias_1 = bias_1,
                              rational_expectation_cost = rational_expectation_cost,
                              risk_free_return = risk_free_return)
            y[i] = BH_constraint(response, real_data)

    if method == "regression":
        return y
    if method == "classification":
        return (y > threshold).astype(int)

def get_BH_real_data():
    """C-Support Vector Classification.
    Parameters
    ----------
    C : float, optional (default=1.0)
        penalty parameter C of the error term.
    kernel : string, optional
         Description of this members.
    Attributes
    ----------
    `bar_` : array-like, shape = [n_features]
        Brief description of this attribute.
    Examples
    --------
    >>> clf = Foo()
    >>> clf.fit()
    []
    See also
    --------
    OtherClass
    """
    """ Get real data sample """
    data_close = pd.read_csv("sp500.csv")
    data_close.index = data_close['Date']
    data_close.drop('Date', axis=1, inplace=True)

    sample = data_close[-_TIME_STEPS:]
    sample = np.log(sample['Adj Close']).diff(1).dropna()

    return sample

def BH_abm(trend_2, 
           trend_1,
           switching_parameter, 
           alpha, 
           bias_2, 
           weight_past_profits,
           bias_1, 
           rational_expectation_cost, 
           risk_free_return, 
           T=_TIME_STEPS,
           _RNG_SEED=0):
    """C-Support Vector Classification.
    Parameters
    ----------
    C : float, optional (default=1.0)
        penalty parameter C of the error term.
    kernel : string, optional
         Description of this members.
    Attributes
    ----------
    `bar_` : array-like, shape = [n_features]
        Brief description of this attribute.
    Examples
    --------
    >>> clf = Foo()
    >>> clf.fit()
    []
    See also
    --------
    OtherClass
    """
    
    """ Default Response Value """
    response = np.array([0.0])

    """ Fixed Parameters (used inside positive price constraint) """
    share_type_1 = 0.5
    init_pdev_fund = 0.2

    """Set Fixed Parameters"""
    dividend_stream = 0.8
    dividend = dividend_stream
    init_wtype_1 = 0
    init_wtype_2 = 0
    sigma2 = 0.002
    fund_price = dividend_stream / (risk_free_return - 1)
    """ Check that the price is positive """
    if (share_type_1 * (trend_1 * init_pdev_fund + bias_1)) + ((1 - share_type_1) * (trend_2 * init_pdev_fund + bias_2)) > 0:
        """ Set RNG Seed"""
        np.random.seed(_RNG_SEED)
        random_dividend = np.random.uniform(low = -0.005, high = 0.005, size = T)

        """ Preallocate Containers """
        X = np.zeros(T)
        P = np.zeros(T)
        N1 = np.zeros(T)

        """ Run simulation """
        for time in range(T):
            # Update fraction of share_type_2
            share_type_2 = 1 - share_type_1

            # Produce Forecast
            forecast = share_type_1 * (trend_1 * init_pdev_fund + bias_1) + share_type_2 * (trend_2 * init_pdev_fund + bias_2)

            # Realized equilibrium_price
            equilibrium_price_realized = forecast / risk_free_return

            # Accumulated type 1 profits
            init_wtype_1 = weight_past_profits * init_wtype_1 + (
                                                                    equilibrium_price_realized - risk_free_return * init_pdev_fund) * (
                                                                    trend_1 * dividend + bias_1 - risk_free_return * init_pdev_fund) / (
                                                                    alpha * sigma2) - rational_expectation_cost

            # Accumulated type 2 profits
            init_wtype_2 = weight_past_profits * init_wtype_2 + (
                                                                    equilibrium_price_realized - risk_free_return * init_pdev_fund) * (
                                                                    trend_2 * dividend + bias_2 - risk_free_return * init_pdev_fund) / (
                                                                    alpha * sigma2)

            # Update fractions
            share_type_1 = np.exp(switching_parameter * init_wtype_1) / (np.exp(switching_parameter * init_wtype_1) + np.exp(switching_parameter * init_wtype_2))
            share_type_2 = 1 - share_type_1

            # Set initial conditions for next period
            dividend = init_pdev_fund
            random_dividend_fluctuation = random_dividend[time]
            init_pdev_fund = equilibrium_price_realized + random_dividend_fluctuation
            #init_pdev_fund = equilibrium_price_realized 

            # set constraint on unstable diverging behaviour
            if init_pdev_fund > 100:
                init_pdev_fund = np.nan
            elif init_pdev_fund < 0:
                init_pdev_fund = np.nan

            """ Record Results """
            # Prices
            X[time] = init_pdev_fund
            P[time] = X[time] + fund_price
            N1[time] = share_type_1

        if X[~np.isnan(X)].shape[0] == T:
            response = np.diff(np.log(X[~np.isnan(X)]))
            #pd.DataFrame(response).to_csv("response.csv")
            
    return response

def BH_constraint(sim_series, sample):
    """
    """
    p_value = 0.0 # Reject LOW P VALUE = REJECT NULL HYPOTHESIS THAT DISTS ARE EQUAL

    # Check that the length is equal to the number of returns
    if sim_series.size == _TIME_STEPS - 1:
        
        np.random.seed(0)

        D, p_value = ks_2samp(sample, sim_series)

    return p_value

def set_surrogate_as_gbt_reg():
    """ 
    Set the surrogate model as Gradient Boosted Decision Trees. A helper 
    function to set the surrogate model and parameter space as Gradient Boosted 
    Decision Trees.

    Parameters
    ----------
    None
    
    Returns
    -------
    surrogate_model :
        The surrogate model object.
    surrogate_parameter_space :
        The parameter space of the surrogate model to be explored.
    """

    # Set the surrogate model as GradientBoostingRegressorm loss is a combination of L1 and L2 regularization
    surrogate_model = GradientBoostingRegressor(random_state = 0)
    
    # Define the parameter space of the surrogate model
    surrogate_parameter_space = {"n_estimators": [100,200,300], # n_estimators
                                 "learning_rate": [0.5,0.1,0.01,0.05],   # learning_rate
                                 "max_depth": [3,4,5],  # max_depth
                                 "subsample": [0.5,0.7,1]} # subsample

    return surrogate_model, surrogate_parameter_space

def set_surrogate_as_gbt_cla():
    """ 
    Set the surrogate model as Gradient Boosted Decision Trees. A helper 
    function to set the surrogate model and parameter space as Gradient Boosted 
    Decision Trees.

    Parameters
    ----------
    None
    
    Returns
    -------
    surrogate_model :
        The surrogate model object.
    surrogate_parameter_space :
        The parameter space of the surrogate model to be explored.
    """

    # Set the surrogate model as GradientBoostingRegressorm loss is a combination of L1 and L2 regularization
    surrogate_model = GradientBoostingClassifier(random_state = 0)
    
    # Define the parameter space of the surrogate model
    surrogate_parameter_space = {"n_estimators": [100,200,300], # n_estimators
                                 "learning_rate": [0.5,0.1,0.01,0.05],   # learning_rate
                                 "max_depth": [3,4,5],  # max_depth
                                 "subsample": [0.5,0.7,1]} # subsample

    return surrogate_model, surrogate_parameter_space

# finished-------------------------------------------------------------------------------------------------------------------
def set_surrogate_as_XG_reg():
    """ 
    Set the surrogate model as XGBoost. A helper 
    function to set the surrogate model and parameter space as XGBoost.

    Parameters
    ----------
    None
    
    Returns
    -------
    surrogate_model :
        The surrogate model object.
    surrogate_parameter_space :
        The parameter space of the surrogate model to be explored.
    """

    # Set the surrogate model as GradientBoostingRegressorm loss is a combination of L1 and L2 regularization
    surrogate_model = xgb.XGBRegressor(objective = 'reg:linear',missing = None, seed = 0)
    
    # Define the parameter space of the surrogate model
    surrogate_parameter_space = {'max_depth':[3,4,5],
                                  'learning_rate':[0.5,0.1,0.01,0.05],
                                  'gamma':[0,0.25,1.0],
                                  'reg_lambda':[0,1.0,10.0,20,100],
                                  'scale_pos_weight':[1,3,5]}

    return surrogate_model, surrogate_parameter_space

# -------------------------------------------------------------------------------------------------------------------


# finished-------------------------------------------------------------------------------------------------------------------
def set_surrogate_as_XG_cla():
    """ 
    Set the surrogate model as XGBoost. A helper 
    function to set the surrogate model and parameter space as XGBoost.

    Parameters
    ----------
    None
    
    Returns
    -------
    surrogate_model :
        The surrogate model object.
    surrogate_parameter_space :
        The parameter space of the surrogate model to be explored.
    """

    # Set the surrogate model as GradientBoostingRegressorm loss is a combination of L1 and L2 regularization
    surrogate_model = xgb.XGBClassifier(objective = 'binary:logistic',missing = None, seed = 0)
    
    # Define the parameter space of the surrogate model   
    surrogate_parameter_space = {'max_depth':[3,4,5],
                                  'learning_rate':[0.5,0.1,0.01,0.05],
                                  'gamma':[0,0.25,1.0],
                                  'reg_lambda':[0,1.0,10.0,20,100],
                                  'scale_pos_weight':[1,3,5]}

    return surrogate_model, surrogate_parameter_space

#-----------------------------------------------------------------------------------------------------------------------------------

def set_surrogate_as_ANN_cla(input_dim):
    """ 
    Set the surrogate model as ANN. A helper 
    function to set the surrogate model and parameter space as ANN.

    Parameters
    ----------
    None
    
    Returns
    -------
    surrogate_model :
        The surrogate model object.
    surrogate_parameter_space :
        The parameter space of the surrogate model to be explored.
    """

    # Set the surrogate model as ANN with the following topology
    # initialize sequential model and add layers
    surrogate_model = keras.Sequential()
    surrogate_model.add(layers.Dense(8, activation='elu',input_shape=(input_dim,)))
    surrogate_model.add(layers.Dense(8, activation='elu'))
    surrogate_model.add(layers.Dense(8, activation='elu'))
    surrogate_model.add(layers.Dropout(0.4))
    surrogate_model.add(layers.Dense(8, activation='elu'))
    surrogate_model.add(layers.Dropout(0.4))
    surrogate_model.add(layers.Dense(2, activation='softmax'))
    
    return surrogate_model

def set_surrogate_as_ANN_reg(input_dim):
    """ 
    Set the surrogate model as ANN. A helper 
    function to set the surrogate model and parameter space as ANN.

    Parameters
    ----------
    None
    
    Returns
    -------
    surrogate_model :
        The surrogate model object.
    surrogate_parameter_space :
        The parameter space of the surrogate model to be explored.
    """

    # Set the surrogate model as ANN with the following topology
    # initialize sequential model and add layers
    surrogate_model = keras.Sequential()
    surrogate_model.add(layers.Dense(8, activation='elu',input_shape=(input_dim,)))
    surrogate_model.add(layers.Dense(8, activation='elu'))
    surrogate_model.add(layers.Dense(8, activation='elu'))
    surrogate_model.add(layers.Dropout(0.4))
    surrogate_model.add(layers.Dense(8, activation='elu'))
    surrogate_model.add(layers.Dropout(0.4))
    surrogate_model.add(layers.Dense(1, activation='relu'))
    
    return surrogate_model

def custom_metric_regression(y_hat, y):
    return mean_squared_error(y, y_hat)

def custom_metric_binary(y_hat, y):
    TP = 0
    
    for i in range(len(y_hat)): 
        if y[i]==y_hat[i]==1:
            TP += 1
    pos = (y == 1).sum()

    return TP/(pos)

def fit_surrogate_model(model, method, X, y, batch):
    """ 
    Fit a surrogate model to the X,y parameter combinations in the real valued case.
    
    Parameters
    ----------
    X :
        Parameter combinations to train the model on.
    y :
        Output of the abm for these parameter combinations.
        
    Returns
    -------
    surrogate_model_fitted : 
        A surrogate model fitted.
    """
    
    # Get the surrogate model and parameter space
    if model == "Gradient boost":
        if method == "regression":
            surrogate_model, surrogate_parameter_space = set_surrogate_as_gbt_reg()
            # Run GridSearchCV       
            grid_search = GridSearchCV(estimator = surrogate_model,
                                          param_grid = surrogate_parameter_space,
                                          scoring = 'neg_mean_squared_error',
                                          verbose = 0,
                                          n_jobs = 10,
                                          cv = 3
                                          )
        if method == "classification":
            surrogate_model, surrogate_parameter_space = set_surrogate_as_gbt_cla()
            # Run randomized parameter search
            grid_search = GridSearchCV(estimator = surrogate_model,
                                          param_grid = surrogate_parameter_space,
                                          scoring = 'roc_auc',
                                          verbose = 0,
                                          n_jobs = 10,
                                          cv = 3
                                          )
        surrogate_model_tuned = grid_search.fit(X, y)

        # Set the hyperparameters  of the surrograte model to the optimised hyperparameters from the random search
        surrogate_model.set_params(n_estimators = surrogate_model_tuned.best_params_["n_estimators"],
                                   learning_rate = surrogate_model_tuned.best_params_["learning_rate"],
                                   max_depth = surrogate_model_tuned.best_params_["max_depth"],
                                   subsample = surrogate_model_tuned.best_params_["subsample"])

        # Fit the surrogate model
        surrogate_model.fit(X, y)
        
    # 1 --------------------------------------------------------------------------------------------------

    if model == "XGBoost":
        if method == "regression":
            surrogate_model, surrogate_parameter_space = set_surrogate_as_XG_reg()
            # Run GridSearchCV       
            optimal_params = GridSearchCV(estimator = surrogate_model,
                                          param_grid = surrogate_parameter_space,
                                          scoring = 'neg_mean_squared_error',
                                          verbose = 0,
                                          n_jobs = 10,
                                          cv = 3
                                          ) 
        if method == "classification":
            surrogate_model, surrogate_parameter_space = set_surrogate_as_XG_cla()
            # Run GridSearchCV       
            optimal_params = GridSearchCV(estimator = surrogate_model,
                                          param_grid = surrogate_parameter_space,
                                          scoring = 'roc_auc',
                                          verbose = 0,
                                          n_jobs = 10,
                                          cv = 3
                                          )
        
        surrogate_model_tuned = optimal_params.fit(X, y)

        # Set the hyperparameters  of the surrograte model to the optimised hyperparameters from the random search
        surrogate_model.set_params(gamma = surrogate_model_tuned.best_params_["gamma"],
                                   learning_rate = surrogate_model_tuned.best_params_["learning_rate"],
                                   max_depth = surrogate_model_tuned.best_params_["max_depth"],
                                   scale_pos_weight = surrogate_model_tuned.best_params_["scale_pos_weight"],
                                   reg_lambda = surrogate_model_tuned.best_params_["reg_lambda"])

        # Fit the surrogate model
        surrogate_model.fit(X, y)


   # 2 ------------------------------------------------------------------------------------------

    if model == "ANN":
        if method == "regression":
            input_dim = X.shape[1]
            surrogate_model = set_surrogate_as_ANN_reg(input_dim)
            surrogate_model.compile(loss='mean_squared_error', optimizer= Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999,                                               epsilon=1e-07))
            # Fit the surrogate model
            surrogate_model.fit(X, y, batch_size = 10, epochs=20, 
                           verbose=0, validation_split=0.0)
        if method == "classification":
            input_dim = X.shape[1]
            surrogate_model = set_surrogate_as_ANN_cla(input_dim)
            surrogate_model.compile(loss='categorical_crossentropy', optimizer= Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999,                                               epsilon=1e-07))
        
            y_to_categorical = keras.utils.to_categorical(y, 2)
            # Fit the surrogate model
            surrogate_model.fit(X, y_to_categorical, batch_size = 10, epochs=20, 
                           verbose=0, validation_split=0.0)
            
    return surrogate_model

def calibration_condition(y, calibration_threshold):
    return (y > calibration_threshold).astype(int)

def fit_entropy_classifier(X, y, method, model, calibration_threshold, batch):
    """ 
    Fit a surrogate model to the X,y parameter combinations in the binary case.
    
    Parameters
    ----------
    X:
        Parameter combinations to train the model on.
    y:
        Output of the abm for these parameter combinations (0 or 1).
    method:
        if current y is regression value or classification value.
    calibration_threshold:
        the threshhold to turn real value y into binary y.
        
    Returns
    -------
    surrogate_model_fitted : 
        A surrogate model fitted.
    """
    if method ==  "regression":
        y_binary = calibration_condition(y, calibration_threshold)
    else:
        y_binary = y
    
    if model == "Gradient boost":    
        clf, clf_space = set_surrogate_as_gbt_cla()
            
        # Run randomized parameter search
        grid_search = GridSearchCV(estimator = clf,
                                      param_grid = clf_space,
                                      scoring = 'roc_auc',
                                      verbose = 0,
                                      n_jobs = 10,
                                      cv = 3
                                      )
        clf_tuned = grid_search.fit(X, y_binary)

        # Set the hyperparameters  of the surrograte model to the optimised hyperparameters from the random search
        clf.set_params(n_estimators = clf_tuned.best_params_["n_estimators"],
                                   learning_rate = clf_tuned.best_params_["learning_rate"],
                                   max_depth = clf_tuned.best_params_["max_depth"],
                                   subsample = clf_tuned.best_params_["subsample"])
        # Fit the surrogate model
        clf.fit(X, y_binary)

    if model == "XGBoost":    
        clf, clf_space = set_surrogate_as_XG_cla()
            
        # Run randomized parameter search
        grid_search = GridSearchCV(estimator = clf,
                                      param_grid = clf_space,
                                      scoring = 'roc_auc',
                                      verbose = 0,
                                      n_jobs = 10,
                                      cv = 3
                                      )
        clf_tuned = grid_search.fit(X, y_binary)

        # Set the hyperparameters  of the surrograte model to the optimised hyperparameters from the random search
        clf.set_params(gamma = clf_tuned.best_params_["gamma"],
                           learning_rate = clf_tuned.best_params_["learning_rate"],
                           max_depth = clf_tuned.best_params_["max_depth"],
                           scale_pos_weight = clf_tuned.best_params_["scale_pos_weight"],
                           reg_lambda = clf_tuned.best_params_["reg_lambda"])
        # Fit the surrogate model
        clf.fit(X, y_binary)        
        
    if model == "ANN":
        input_dim = X.shape[1]
        clf = set_surrogate_as_ANN_cla(input_dim)
        clf.compile(loss='categorical_crossentropy',optimizer= Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999, epsilon=1e-07))
        
        y_to_categorical = keras.utils.to_categorical(y, 2)
        
        # Fit the surrogate model
        clf.fit(X, y_to_categorical, batch_size = 10, epochs=20, 
                       verbose=0, validation_split=0.0)

    return clf

def get_sobol_samples(n_dimensions, samples, parameter_support):
    """
    """
    # Get the range for the support
    support_range = parameter_support[:, 1] - parameter_support[:, 0]

    # Generate the Sobol samples
    random_samples = sobol_seq.i4_sobol_generate(n_dimensions, samples)

    # Compute the parameter mappings between the Sobol samples and supports
    sobol_samples = np.vstack([
        np.multiply(s, support_range) + parameter_support[:, 0]
        for s in random_samples])

    return sobol_samples

def get_unirand_samples(n_dimensions, samples, parameter_support):
    """
    """
    # Get the range for the support
    support_range = parameter_support[:, 1] - parameter_support[:, 0]

    # Generate the Sobol samples
    random_samples = np.random.rand(n_dimensions,samples).T

    # Compute the parameter mappings between the Sobol samples and supports
    unirand_samples = np.vstack([
        np.multiply(s, support_range) + parameter_support[:, 0]
        for s in random_samples])

    return unirand_samples

def get_round_selections(evaluated_set_X, evaluated_set_y,
                         unevaluated_set_X,
                         predicted_binary, num_predicted_positives,
                         batch, calibration_threshold,
                         method, model, trained_model):
    """
    Update evaluated_set_X, evaluated_set_y and unevaluated_set_X
    
    Parameters
    ----------
    evaluated_set_X:
        evaluated X set to be updated.
    evaluated_set_y:
        evaluated y set to be updated.
    unevaluated_set_X:
        unevaluated_set_X to be updated.
    predicted_binary:
        predicted labels of unevaluated_set_X by surrogate model.
    num_predicted_positives:
        number of positive labels in predicted_binary.
    batch:
        size of batch needed to be added to evaluated sets
    method:
        if current y is regression value or classification value.
    calibration_threshold:
        the threshhold to turn real value y into binary y. will only be used is method is "regression"
        
    Returns
    -------
    updated evaluated_set_X, evaluated_set_y and unevaluated_set_X
    """

    if num_predicted_positives >= batch:
        round_selections = batch
        selections = np.where(predicted_binary == True)[0]
        selections = np.random.permutation(selections)[:round_selections]
        print("Parameters bundles of indices: ", selections, " are randomly selected from positive prediction candidates, and will added to the evaluated sets.")
        to_be_evaluated = unevaluated_set_X[selections]
        unevaluated_set_X = np.delete(unevaluated_set_X, selections, 0)

    else:
        # select all predicted positives indices
        selections = np.where(predicted_binary == True)[0]
        print("Parameters bundles of indices: ", selections, " are the only ones with positive prediction candidates. They will added to the evaluated sets.")
        to_be_evaluated = unevaluated_set_X[selections]
        unevaluated_set_X = np.delete(unevaluated_set_X, selections, 0)
        
        # select remainder according to entropy weighting
        budget_shortfall = int(batch - num_predicted_positives)
        selections = get_new_labels_entropy(evaluated_set_X, evaluated_set_y,
                                                      unevaluated_set_X,
                                                      calibration_threshold,
                                                      budget_shortfall, method, model, batch, trained_model)
        print("Parameters bundles of indices: ", selections, " in the rest of the pool are added to the evaluated sets using uncertainty sampling.")
        to_be_evaluated = np.vstack([to_be_evaluated,unevaluated_set_X[selections]])
        unevaluated_set_X = np.delete(unevaluated_set_X, selections, 0)
        
    evaluated_set_X = np.vstack([evaluated_set_X, to_be_evaluated])
    evaluated_set_y = np.append(evaluated_set_y, evaluate_bh_on_set(to_be_evaluated, method, calibration_threshold))

    return evaluated_set_X, evaluated_set_y, unevaluated_set_X

def get_new_labels_entropy(evaluated_set_X, evaluated_set_y,
                           unevaluated_X, calibration_threshold,
                           number_of_new_labels, method, model, batch, trained_model):
    """ Get a set of parameter combinations according to their predicted label entropy
    """
    if method == "classification":
        if model == "ANN":
            y_hat_probability = trained_model.predict(unevaluated_X)
        else:
            y_hat_probability = trained_model.predict_proba(unevaluated_X)
        y_hat_entropy = np.array(list(map(entropy, y_hat_probability)))
        y_hat_entropy /= y_hat_entropy.sum()
    if method == "regression":
        clf = fit_entropy_classifier(evaluated_set_X, evaluated_set_y, method, model, calibration_threshold, batch)
        if model == "ANN":
            y_hat_probability = clf.predict(unevaluated_X)
        else:
            y_hat_probability = clf.predict_proba(unevaluated_X)
        y_hat_entropy = np.array(list(map(entropy, y_hat_probability)))
        y_hat_entropy /= y_hat_entropy.sum()
        
    unevaluated_X_size = unevaluated_X.shape[0]

    selections = np.random.choice(a=unevaluated_X_size,
                                  size=number_of_new_labels,
                                  replace=False,
                                  p=y_hat_entropy)
    return selections

def active_learning(evaluated_set_X, evaluated_set_y, unevaluated_set_X, budget, method, model, threshold, batch):
    '''
    return final evaluated_set_X, evaluated_set_y constructed through active learning algorithm.
    
    Parameters
    ----------
    evaluated_set_X, evaluated_set_y:
        Inital evaluated_set_X, evaluated_set_y to be constructed. 
    unevaluated_set_X :
        unevaluated_set_X pool to draw from to be added to evaluated_set_X.
    budget: 
        number of parameters will be evaluated in total. budget = size(evaluated_set_X return).
    method:
        surrogate model type: "regression", "classification".
    model: 
        surrogate model: "Gradient boost",
    threshold:
        threshold to convert real value y to binanru classification when needed. Will only be used then method is 
        "regression".
    batch:
        default update batch size
        
    Returns
    -------
    evaluated_set_X, evaluated_set_y
    
    '''
    eva_num = evaluated_set_y.size
    
    while eva_num < budget:

        print("--------------------------------------------------------")
        print("There are still {} parameter unevaluated within budget.".format(budget - eva_num))
        update_size = np.min([budget - eva_num,batch])    
        
        my_model = fit_surrogate_model(model, method, evaluated_set_X, evaluated_set_y, batch)
        if method == "regression":
            unevaluated_set_y_hat = my_model.predict(unevaluated_set_X)
            predicted_binary = unevaluated_set_y_hat > threshold
            num_predicted_positives = predicted_binary.astype(int).sum()
        if method == "classification":
            if model == "ANN":
                predicted_binary = my_model.predict_classes(unevaluated_set_X)  
            else:
                predicted_binary = my_model.predict(unevaluated_set_X)
            num_predicted_positives = predicted_binary.sum()
            predicted_binary = predicted_binary.astype(bool)

        print("Surrogate model predicted {} positive labels out of {}: ".format(num_predicted_positives,predicted_binary.size))

        evaluated_set_X, evaluated_set_y, unevaluated_set_X = get_round_selections(evaluated_set_X, evaluated_set_y,
                                 unevaluated_set_X,
                                 predicted_binary, num_predicted_positives,
                                 update_size, threshold,
                                 method, model, my_model)

        eva_num = eva_num + update_size
    return evaluated_set_X, evaluated_set_y

def print_y_test_info(y_test, method, threshold):
    '''
    '''
    if method == "regression":
        print("There are {} out of {} in the y test set greater than {}.".format((y_test>threshold).astype(int).sum(),y_test.size, threshold))
    if method == "classification":
        print("There are {} out of {} in the y test set that are positive.".format(y_test.sum(),y_test.size))
        
def print_evaluated_set_y_info(evaluated_set_y,method, threshold):
    '''
    '''
    if method == "regression":
        print("There are {} out of {} in the evaluated y set that are greater than {}."
          .format((evaluated_set_y>threshold).astype(int).sum(), evaluated_set_y.size, threshold))
    if method == "classification":
        print("There are {} out of {} in the evaluated y set that are positive."
          .format(evaluated_set_y.sum(), evaluated_set_y.size))

print ("Imported successfully")