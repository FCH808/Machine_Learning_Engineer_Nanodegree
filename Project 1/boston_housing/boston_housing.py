"""Load the Boston dataset and examine its target (label) distribution."""

# Load libraries
import numpy as np
import pylab as pl
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cross_validation import ShuffleSplit, train_test_split, KFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor


################################
### ADD EXTRA LIBRARIES HERE ###
################################


def load_data():
    """Load the Boston dataset."""

    boston = datasets.load_boston()
    return boston


def explore_city_data(city_data):
    """Calculate the Boston housing statistics."""

    # Get the labels and features from the housing data
    housing_prices = city_data.target
    housing_features = city_data.data

    ###################################
    ### Step 1. YOUR CODE GOES HERE ###
    ###################################

    # Please calculate the following values using the Numpy library

    # Size of data (number of houses)?
    n_rows = np.shape(city_data.data)[0]
    # Number of features?
    n_features = np.shape(city_data.data)[0]
    # Minimum price?
    price_min = np.min(city_data.target)
    # Maximum price?
    price_max = np.max(city_data.target)
    # Calculate mean price?
    price_mean = np.mean(city_data.target)
    # Calculate median price?
    price_median = np.median(city_data.target)
    # Calculate standard deviation?
    price_std = np.std(city_data.target)


def performance_metric(label, prediction):
    """Calculate and return the appropriate error performance metric."""

    ###################################
    ### Step 2. YOUR CODE GOES HERE ###
    ###################################

    # http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics
    return mean_squared_error(label, prediction)


def split_data(city_data):
    """Randomly shuffle the sample set. Divide it into 70 percent training and 30 percent testing data."""

    # Get the features and labels from the Boston housing data
    X, y = city_data.data, city_data.target

    ###################################
    ### Step 3. YOUR CODE GOES HERE ###
    ###################################
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
    return X_train, y_train, X_test, y_test


def learning_curve(depths, X_train, y_train, X_test, y_test):
    """Calculate the performance of the model after a set of training data."""

    # print "Decision Tree with Max Depth: "
    # print depth

    # We will vary the training set size so that we have 50 different sizes
    # Make as an argument to only create once and use for all call in the loop.
    num_of_sizes = 50
    sizes = np.linspace(1, len(X_train), num_of_sizes)

    total_size = num_of_sizes * len(depths)

    # Initialize a empty Numpy arrays for all training/test errors
    train_err_all = np.empty(0)
    test_err_all = np.empty(0)

    # Repeat each depth value 50 times (num_of_sizes)
    depths_all = np.repeat(depths, num_of_sizes)

    # Repeat the sizes sequence (np.linspace()..) so each step is labeled in each depth
    size_all = np.resize(sizes, total_size)

    fill_index = 0

    for depth in depths:
        test_err, train_err = train_decision_tree(sizes, depth, X_test, X_train, y_test, y_train)

        train_err_all = np.append(train_err_all, train_err)
        test_err_all = np.append(test_err_all, test_err)

        fill_index += num_of_sizes

    learning_curve_df = pd.DataFrame({'Depth': depths_all,
                                      'Training Error': train_err_all,
                                      'Test Error': test_err_all,
                                      'Size': size_all})

    learning_curve_graph_pd(learning_curve_df)


def train_decision_tree(sizes, depth, X_test, X_train, y_test, y_train):
    """
    Args:
        sizes   (Numpy array): Array of training sample sizes to train on.
        depth   (int): The maximum depth of the DecisionTreeRegressor
        X_test  (Numpy array): Test set features
        X_train (Numpy array): Training set features
        y_test  (Numpy array): Test set target variable
        y_train (Numpy array): Training set target variable

    Returns:
        test_err  (Numpy array): Test set predictions.
        train_err (Numpy array): Training set predictions.
    """

    train_err = np.zeros(len(sizes))
    test_err = np.zeros(len(sizes))

    for i, s in enumerate(sizes):
        # Create and fit the decision tree regressor model
        regressor = DecisionTreeRegressor(max_depth=depth)
        regressor.fit(X_train[:s], y_train[:s])

        # Find the performance on the training and testing set
        train_err[i] = performance_metric(y_train[:s], regressor.predict(X_train[:s]))
        test_err[i] = performance_metric(y_test, regressor.predict(X_test))

    return test_err, train_err


def learning_curve_graph_pd(df):
    # TODO: Add docstrings
    # key = each Depth value
    # grp is each sub-dataframe filtered by the Depth value of 'key'
    fig = plt.figure(figsize=(16, 18))

    for i, key_group_pair in enumerate(df.groupby(['Depth'])):
        key, grp = key_group_pair
        each_ax = fig.add_subplot(5, 2, i + 1)
        each_ax.set_ylim([0, 80])
        each_ax.set_title('Depth: {}'.format(i + 1))
        each_ax.plot(grp['Size'], grp['Test Error'])
        each_ax.plot(grp['Size'], grp['Training Error'])

    plt.suptitle('Train & Test set error as a function of training set size', fontsize=20)
    plt.legend(loc='best')
    plt.xlabel('Training set size', fontsize=20, horizontalalignment='right')
    plt.show()


def learning_curve_graph(sizes, train_err, test_err, show=True):
    """Plot training and test error as a function of the training size."""

    pl.figure()
    pl.title('Decision Trees: Performance vs Training Size')
    pl.plot(sizes, test_err, lw=2, label='test error')
    pl.plot(sizes, train_err, lw=2, label='training error')
    pl.legend()
    pl.xlabel('Training Size')
    pl.ylabel('Error')
    pl.show()


def model_complexity(X_train, y_train, X_test, y_test):
    """Calculate the performance of the model as model complexity increases."""

    print "Model Complexity: "

    # We will vary the depth of decision trees from 2 to 25
    max_depth = np.arange(1, 25)
    train_err = np.zeros(len(max_depth))
    test_err = np.zeros(len(max_depth))

    for i, d in enumerate(max_depth):
        # Setup a Decision Tree Regressor so that it learns a tree with depth d
        regressor = DecisionTreeRegressor(max_depth=d)

        # Fit the learner to the training data
        regressor.fit(X_train, y_train)

        # Find the performance on the training set
        train_err[i] = performance_metric(y_train, regressor.predict(X_train))

        # Find the performance on the testing set
        test_err[i] = performance_metric(y_test, regressor.predict(X_test))

    # Plot the model complexity graph
    model_complexity_graph(max_depth, train_err, test_err)


def model_complexity_graph(max_depth, train_err, test_err):
    """Plot training and test error as a function of the depth of the decision tree learn."""

    pl.figure()
    pl.title('Decision Trees: Performance vs Max Depth')
    pl.plot(max_depth, test_err, lw=2, label='test error')
    pl.plot(max_depth, train_err, lw=2, label='training error')
    pl.legend()
    pl.xlabel('Max Depth')
    pl.ylabel('Error')
    pl.show()


def fit_predict_model(city_data, verbose=True):
    # TODO: Add docstrings
    """Find and tune the optimal model. Make a prediction on housing data."""

    # Get the features and labels from the Boston housing data
    X, y = city_data.data, city_data.target

    # Setup a Decision Tree Regressor
    regressor = DecisionTreeRegressor()

    parameters = {'max_depth': (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)}

    ###################################
    ### Step 4. YOUR CODE GOES HERE ###
    ###################################
    # TODO: Add train/test split to fit_predict() to fit on testing data only as part of best practices.
    # TODO: Move KFold justification to report outside script.
    # 1. Find the best performance metric
    # should be the same as your performance_metric procedure
    # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html

    mse_scorer = make_scorer(performance_metric,  # Use our performance metric, which is just mean squared error(mse)
                             greater_is_better=False)  # False because we are trying to minimize a loss function, MSE

    # 2. Use gridearch to fine tune the Decision Tree Regressor and find the best model
    # http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html#sklearn.grid_search.GridSearchCV


    # We'll choose the more often used 10-fold cross validation instead of the default of 3.
    # We'd like to avoid having our K-fold split being too low to avoid having higher variance between models trained.
    #   If K is too low, the amount of data for training the model (66% in the case of 3-fold) would be lower (which
    #   might be a problem with our smaller data set.
    # We also don't want to k-fold splits to be too high, since as K approaches N (the number of data points in our data
    #   set), we would have more and more overlap in training data used to train on. This would make all of the models
    #   more and more correlated with each other as K approaches N. This might cause another type of higher variance in
    #   our predicted models since "In general, if the variables are correlated, then the variance of their sum is
    #   the sum of their covariances" 1, 2
    # 1. https://en.wikipedia.org/wiki/Variance#Sum_of_correlated_variables
    # 2. http://stats.stackexchange.com/questions/61783/variance-and-bias-in-cross-validation-why-does-leave-one-out-cv-have-higher-var


    # Ideally we would want to split once around 70/30 train/test.
    # Then do Kfold cross-validation on a 70% training set only, then test once only on the 30% hold-out test set.

    # Or even more ideally, further split the training set into a training/validation sets,
    # train on the training set while scoring on the hold-out validation set to adjust our model parameters.
    # Then once done fine-tuning, run our model once on the 30% hold-out test set to get an idea of the out-of-sample
    #   performance.
    # But again, this would require more data since so many splits would reduce the predictive power of our model,
    #   causing it to have higher variance.


    # Note: In this particular function, we are simply training a final model using the full training set and
    #   10-fold cross validation.

    kfold_split = KFold(np.shape(X)[0],
                        n_folds=10,
                        random_state=None)

    reg = GridSearchCV(estimator=regressor,
                       param_grid=parameters,
                       scoring=mse_scorer,
                       n_jobs=4,
                       cv=kfold_split)

    # Fit the learner to the training data
    if verbose == True:
        print "Final Model: "
        print reg.fit(X, y)

        print "*" * 80
        print "Best Estimator: {0}".format(reg.best_estimator_)
        print ""
        print "Best Params: {0}".format(reg.best_params_)
        print "Mean Square Error of Best Model {0}".format(reg.best_score_)
        print "*" * 80

        # Use the model to predict the output of a particular sample
        # Changed to address changes in sklearn:
        # DeprecationWarning: Passing 1d arrays as data is deprecated in 0.17 and willraise ValueError in 0.19.
        x = np.array([11.95, 0.00, 18.100, 0, 0.6590, 5.6090, 90.00, 1.385, 24, 680.0, 20.20, 332.09, 12.13]).reshape(1,
                                                                                                                      -1)
        y = reg.predict(x)
        print "House: " + str(x)
        print "Prediction: " + str(y)
    else:
        # Just train the model using gridSearch but don't print anything if verbose is false.
        reg.fit(X, y)

    # Return our gridSearch object containing our best model.
    return reg


def fit_predict_many(city_data, n=100, verbose=False):
    # TODO: Add docstrings
    # Here, we can use many iterations of random 10-fold validation splits to try to control our model variance when
    #   choosing a best parameter.
    best_max_depth_all = []
    for i in xrange(n):
        one_pass = fit_predict_model(city_data, verbose=verbose)
        best_max_depth_all.append(one_pass)
    return best_max_depth_all


def get_xval_data_from_GridScore(oneGridScoreObject):
    # TODO: add docstrings

    best_depth = oneGridScoreObject.best_params_['max_depth']
    best_depth_kfold_scores = oneGridScoreObject.grid_scores_[best_depth - 1].cv_validation_scores

    # TODO: Remove if calculating summary stats in the end instead
    # best_loss = oneGridScoreObject.best_score_
    # best_std = np.std(oneGridScoreObject.grid_scores_[best_depth - 1].cv_validation_scores)

    return best_depth, best_depth_kfold_scores


def get_xval_data_from_list_Gridscores(manyGridScoreObjects):
    # TODO: add docstrings

    best_max_depths_all = {}

    for eachGridScoreObject in manyGridScoreObjects:

        # Pull out the loss for each fold in each best 'max depth' of each run.
        best_depth, best_depth_kfold_scores = get_xval_data_from_GridScore(eachGridScoreObject)

        try:
            best_max_depths_all[best_depth] = np.append(best_max_depths_all[best_depth], best_depth_kfold_scores)
        except KeyError:
            best_max_depths_all[best_depth] = best_depth_kfold_scores

    # TODO: remove if aggregating all and getting summary stats on original kfold datapoints
    # http://stats.stackexchange.com/questions/25848/how-to-sum-a-standard-deviation
    # http://en.wikipedia.org/wiki/Sum_of_normally_distributed_random_variables
    # http://mathworld.wolfram.com/NormalSumDistribution.html
    # Here we can 1. average the means for each best value of 'max depth', and
    #             2. sum the variances, then take the square root of the summed variances
    #                to get the average standard deviation of the standard deviations.

    # max_depths_all['avg_loss_']

    return best_max_depths_all


def create_df_from_mismatch_dict(best_max_depths_all):

    max_depths_all = []
    scores_all = []

    for key, value in best_max_depths_all.iteritems():
        max_depths = [key] * len(value)
        scores = list(value)

        max_depths_all += max_depths
        scores_all += scores

    return pd.DataFrame({'Max Depth': max_depths_all, 'Loss': scores_all})


def plot_all_distributions(best_depths):
    # TODO: Add docstrings

    best_max_depths_all = get_xval_data_from_list_Gridscores(best_depths)

    best_max_depths_df = create_df_from_mismatch_dict(best_max_depths_all)

    best_max_depths_stats = best_max_depths_df.groupby(['Max Depth'], as_index=False)\
        .aggregate([np.mean, np.std])\
        .reset_index()


    # TODO: fix this plot

    plt.xlim([0, 11])
    plt.gca().invert_yaxis()

    plt.scatter(best_max_depths_df['Max Depth'],
                best_max_depths_df['Loss'], alpha=0.25)

    plt.scatter(best_max_depths_stats['Max Depth'],
                best_max_depths_stats['Loss']['mean'],
                color='red',
                s=20, marker='x')

    plt.errorbar(best_max_depths_stats['Max Depth'],
                 best_max_depths_stats['Loss']['mean'],
                 yerr=best_max_depths_stats['Loss']['std'])

    plt.show()


def plot_hist_best_max_depths(best_depths):
    # TODO: Add docstrings
    best_depths_list = [model.best_params_['max_depth'] for model in best_depths]

    plt.figure(figsize=(12, 9))
    # Remove unneeded plot frame lines and ticks.
    ax = plt.subplot(1, 1, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.xticks(fontsize=14)
    plt.xlabel('Best Max Depth', fontsize=16)
    plt.ylabel('Count', fontsize=16)

    plt.hist(best_depths_list, color="#3F5D7D", bins=range(11))
    plt.title(
        'Best "Max Depth" parameter settings found over {0} different runs of gridSearch'.format(len(best_depths)))

    plt.show()


def plot_prediction_distribution(best_depths):
    # TODO: Add docstrings
    from scipy.stats.kde import gaussian_kde

    x = np.array([11.95, 0.00, 18.100, 0, 0.6590, 5.6090, 90.00, 1.385, 24, 680.0, 20.20, 332.09, 12.13]).reshape(1, -1)
    y_predictions = [model.predict(x)[0] for model in best_depths]

    plt.figure(figsize=(12, 9))
    # Remove unneeded plot frame lines and ticks.
    ax = plt.subplot(1, 1, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.xticks(fontsize=14)
    plt.xlabel('Predicted Value', fontsize=16)
    plt.ylabel('Probability', fontsize=16)

    # http://stackoverflow.com/questions/15415455/plotting-probability-density-function-by-sample-with-matplotlib
    # this create the kernel, given an array it will estimate the probability over that values
    kde = gaussian_kde(y_predictions)
    # these are the values over which your kernel will be evaluated
    dist_space = np.linspace(min(y_predictions), max(y_predictions), 100)

    plt.plot(dist_space, kde(dist_space), color="#3F5D7D")
    plt.title(
        'Probability Density for Prediction from each Best Max Depth Model over {0} different runs of gridSearch'.format(
            len(best_depths)))

    plt.show()


def main():
    """Analyze the Boston housing data. Evaluate and validate the
    performanance of a Decision Tree regressor on the housing data.
    Fine tune the model to make prediction on unseen data."""

    # Load data
    city_data = load_data()

    # Explore the data
    explore_city_data(city_data)

    # Training/Test dataset split
    X_train, y_train, X_test, y_test = split_data(city_data)

    # Learning Curve Graphs

    # Move logic/flow control out of main() and pass list as argument to our learning_curve function.
    max_depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # learning_curve(max_depths, X_train, y_train, X_test, y_test)

    # Model Complexity Graph
    # model_complexity(X_train, y_train, X_test, y_test)

    # Tune and predict Model
    fit_predict_model(city_data)

    # Tune and predict Model over many iterations.
    many_iter = fit_predict_many(city_data, n=20)
    # plot_hist_best_max_depths(many_iter)
    # plot_prediction_distribution(many_iter)

    plot_all_distributions(many_iter)


if __name__ == "__main__":
    main()
