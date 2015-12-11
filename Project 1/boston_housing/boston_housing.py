"""Load the Boston dataset and examine its target (label) distribution."""

# Load libraries
import numpy as np
import pylab as pl
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats.kde import gaussian_kde
from sklearn import datasets
from sklearn.cross_validation import ShuffleSplit, train_test_split, KFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

plt.style.use('bmh')

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
    print 'Number of houses in data set: {}'.format(n_rows)

    # Number of features?
    n_features = np.shape(city_data.data)[1]
    print 'Number of features: {}'.format(n_features)

    # Minimum price?
    price_min = np.min(city_data.target)
    print 'Minimum house price in dataset: {0}'.format(price_min)

    # Maximum price?
    price_max = np.max(city_data.target)
    print 'Maximum house price in dataset: {0}'.format(price_max)

    # Calculate mean price?
    price_mean = np.mean(city_data.target)
    print 'Average house price: {0}'.format(price_mean)

    # Calculate median price?
    price_median = np.median(city_data.target)
    print 'Median house price: {0}'.format(price_median)

    # Calculate standard deviation?
    price_std = np.std(city_data.target)
    print 'House price standard deviation: {0}'.format(price_std)


def performance_metric(label, prediction):
    """Calculate and return the appropriate error performance metric."""

    ###################################
    ### Step 2. YOUR CODE GOES HERE ###
    ###################################

    return mean_squared_error(label, prediction)


def split_data(city_data, train_size=0.7, random=True):
    '''Randomly shuffle the sample set. Divide it into 70 percent training and
        30 percent testing data.

        Optionally, random flag can be set to False to return deterministic
            results for report analysis. Set to True by default for running
            multiple times in Monte Carlo Simulations.

    Args:
        city_data: (Numpy array): Boston Housing data from sklearn.datasets
        train_size (int): Size of training set to be returned.
        random (Boolean): Random seed flag for deterministic results.

    Returns:
        X_train (Numpy array): Training set features.
        y_train (Numpy array): Training set target.
        X_test (Numpy array): Test set features.
        y_test (Numpy array): Test set target.

    '''
    # Get the features and labels from the Boston housing data
    X, y = city_data.data, city_data.target

    ###################################
    ### Step 3. YOUR CODE GOES HERE ###
    ###################################

    random_state = None

    # Set a seed for the depth analysis graphs only
    if random != True:
        random_state = 333

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=random_state)
    return X_train, y_train, X_test, y_test



def learning_curve(depths, X_train, y_train, X_test, y_test):
    '''Calculate the performance of the model after a set of training data.

    1. Takes a list of depths to train
    2. Calls a helper function (train_decision_tree) to train a series
     of models at each depth.
    3. At each depth, a model is trained on varying amounts of training set
     sizes.
    4. All results are aggregated into a Pandas data frame.

    Example output format with 2 entries would be:

    {'Depth': [1, 1],
     'Training Error': [-0.2, -0.3],
     'Test Error': [-0.3, -0.25],
     'Size': [1, 2]}

     5. Results are plotted using the helper function learning_curve_graph_pd()

    Args:
        depths (List): List of depths
        X_train (Numpy array): Training set features.
        y_train (Numpy array): Training set target.
        X_test (Numpy array): Test set features.
        y_test (Numpy array): Test set target.

    Returns:
        null

    '''

    # We will vary the training set size so that we have 50 different sizes
    # Make as an argument to only create once and use for all call in the loop.
    num_of_sizes = 50
    sizes = np.round(np.linspace(1, len(X_train), 50))
    # Convert to ints to avoid numpy DeprecationWarning
    sizes = [int(x) for x in sizes]

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

        # Cast to int to avoid DeprecationWarning from numpy 1.8
        regressor.fit(X_train[:int(s)], y_train[:int(s)])

        # Find the performance on the training and testing set
        train_err[i] = performance_metric(y_train[:s], regressor.predict(X_train[:s]))
        test_err[i] = performance_metric(y_test, regressor.predict(X_test))

    return test_err, train_err


def learning_curve_graph_pd(df):
    '''Plots error values as a function of training set size for each depth from
        the provided dataframe.

    Example dataframe with 2 entries.

    {'Depth': [1, 1],
     'Training Error': [-0.2, -0.3],
     'Test Error': [-0.3, -0.25],
     'Size': [1, 2]}

    Groups the dataframe by Depth, then plots error and training/test error on
        one plot.

    Args:
        df (Pandas dataframe): Pandas dataframe

    Returns:
        null
    '''

    # key = each Depth value
    # grp is each sub-dataframe filtered by the Depth value of 'key'
    fig = plt.figure(figsize=(16, 18))
    ax = fig.add_subplot(1,1,1)

    for i, key_group_pair in enumerate(df.groupby(['Depth'])):
        key, grp = key_group_pair
        each_ax = fig.add_subplot(5, 2, i + 1)
        #each_ax.set_ylim([0, 80])
        each_ax.set_title('Depth: {}'.format(i + 1))
        each_ax.plot(grp['Size'], grp['Test Error'])
        each_ax.plot(grp['Size'], grp['Training Error'])

    plt.suptitle('Train & Test set error as a function of training set size', fontsize=20)
    plt.legend(loc='best')

    # Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    ax.set_ylabel('Mean Squared Error',fontsize=20)
    ax.set_xlabel('Training set size', fontsize=20, horizontalalignment='center')

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


def model_complexity_graph(max_depths, train_err, test_err):
    """Plot training and test error as a function of the depth of the decision tree learn."""
    # Change to matplotlib.pyplot for pyplot.xticks() to accept a list in ipython

    plt.figure(figsize=(12, 9))
    plt.title('Decision Trees: Performance vs Max Depth')
    plt.plot(max_depths, test_err, lw=2, label='test error')
    plt.plot(max_depths, train_err, lw=2, label='training error')
    plt.xticks(max_depths)
    plt.legend()
    plt.xlabel('Max Depth')
    plt.ylabel('Error')
    plt.show()


def fit_predict_model(city_data, verbose=True):
    '''
    1. Splits the city_data into training/testing splits.
    2. Creates a performance scorer and k-fold object for training.
    3. Trains a DecisionTreeRegressor() model on training data using
     GridScoreCV()
     3a. If verbose is True, trains then predicts on one predefined sample then
      prints training info to stdout.
    4. Return fully trained GridScoreCV object containing best model.

    Args:
        city_data (Numpy array): Boston Dataset from sklearn.datasets
        verbose (Boolean): Whether to print out model statistics when done
            training the model.

    Returns:
        reg (GridScoreCV): A pre-trained GridScoreCV object.

    '''
    """Find and tune the optimal model. Make a prediction on housing data."""

    # Get the features and labels from the Boston housing data
    # Not needed in this version.
    # X, y = city_data.data, city_data.target

    # Setup a Decision Tree Regressor
    regressor = DecisionTreeRegressor()

    parameters = {'max_depth': (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)}

    ###################################
    ### Step 4. YOUR CODE GOES HERE ###
    ###################################

    # Training/Test dataset split
    # We'll give a larger train size split here since we will be doing K-fold grid search cross validation
    # over the training set and our dataset is not very big to being with.
    X_train, y_train, X_test, y_test = split_data(city_data, train_size=0.90)

    # 1. Find the best performance metric
    # should be the same as your performance_metric procedure
    # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html

    mse_scorer = make_scorer(performance_metric,  # Use our performance metric, which is just mean squared error(mse)
                             greater_is_better=False)  # False because we are trying to minimize a loss function, MSE

    # 2. Use gridearch to fine tune the Decision Tree Regressor and find the best model
    # http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html#sklearn.grid_search.GridSearchCV


    kfold_split = KFold(np.shape(X_train)[0],
                        n_folds=5,
                        random_state=None)

    reg = GridSearchCV(estimator=regressor,
                       param_grid=parameters,
                       scoring=mse_scorer,
                       n_jobs=4,
                       cv=kfold_split)

    # Fit the learner to the training data
    if verbose == True:
        print "Final Model: "
        print reg.fit(X_train, y_train)

        print "*" * 80
        print "Best Estimator: {0}".format(reg.best_estimator_)
        print ""
        print "Best Params: {0}".format(reg.best_params_)
        print "Mean Square Error of Best Model {0}".format(reg.best_score_)
        print "*" * 80

        # Use the model to predict the output of a particular sample
        # Changed to address changes in sklearn:
        # DeprecationWarning: Passing 1d arrays as data is deprecated in 0.17 and willraise ValueError in 0.19.
        x = np.array([11.95, 0.00, 18.100, 0, 0.6590, 5.6090, 90.00, 1.385, 24,
                      680.0, 20.20, 332.09, 12.13]).reshape(1, -1)
        y = reg.predict(x)
        print "House: " + str(x)
        print "Prediction: " + str(y)
    else:
        # Just train the model using gridSearch but don't print anything if verbose is false.
        reg.fit(X_train, y_train)

    # Return our gridSearch object containing our best model.
    return reg


def fit_predict_many(city_data, n=100, verbose=False):
    '''Helper function to train 'n' models on the provided using the
    fit_predict_model to handle prediction internals.

    Args:
        city_data (Numpy array): Boston Dataset from sklearn.datasets
        n (int): Number of models to train.
        verbose (Boolean): Whether to print out model statistics when done
            training the model.

    Returns:
        best_max_depth_all (List):  List of 'n' GridScoreCV sklearn objects,
            each containing a best trained model on the dataset.

    '''

    best_max_depth_all = []
    for i in xrange(n):
        one_pass = fit_predict_model(city_data, verbose=verbose)
        best_max_depth_all.append(one_pass)

    return best_max_depth_all


def get_xval_data_from_GridScore(oneGridScoreObject):
    '''Extracts the best depth and a list of error scores from that depth
     from the pre-trained GridScoreCV object.

    Args:
        oneGridScoreObject (GridScoreCV): A pre-trained GridScoreCV object.

    Returns:
        best_depth (int): Best depth of the best model in the GridScoreCV
                          object

        best_depth_kfold_scores (List): List of the error scores from the best
                            depth.

    '''

    best_depth = oneGridScoreObject.best_params_['max_depth']

    # Array indexing starts at 0, so the index is [max_depth - 1]
    best_depth_kfold_scores = oneGridScoreObject.grid_scores_[best_depth - 1].cv_validation_scores

    return best_depth, best_depth_kfold_scores


def get_xval_data_from_list_Gridscores(manyGridScoreObjects):
    ''' Takes a list of pre-trained GridScoreCV objects and extracts the best
        depth and error scores from each GridScoreCV object.

    Multiple error results from same max depth are concatenated.

    Return as a dictionary with the following example format.

    Example of errors from 2 3-fold models with best max depth of 5 and
                           1 3-fold model  with best max depth of 6.

    example_dict = {5: [-0.4, -0.5, -0.2, -0.4, -0.2, -0.5],
                    6: [-0.3, -0.4, -0.4]}

    Args:
        manyGridScoreObjects (List): List of pre-trained GridScoreCV objects

    Returns:
        best_max_depths_all (dict): Python dictionary with best_depth as key
            and a list of error scores from best performing depth of each model.
    '''

    best_max_depths_all = {}

    for eachGridScoreObject in manyGridScoreObjects:

        # Pull out the loss for each fold in each best 'max depth' of each run.
        best_depth, best_depth_kfold_scores = get_xval_data_from_GridScore(eachGridScoreObject)

        try:
            best_max_depths_all[best_depth] = np.append(best_max_depths_all[best_depth], best_depth_kfold_scores)
        except KeyError:
            best_max_depths_all[best_depth] = best_depth_kfold_scores

    return best_max_depths_all


def unpack_dict_to_df(best_max_depths_all):
    ''' Unpacks dictionary into pandas dataframe.

    Example: Unpacks data from:

    dict({2, [-0.4, -0.3],
         4, [-0.5, -0.2]})

    To:

    pd.DataFrame({'Max Depth': [2, 2, 4, 4]
                  'Loss': [-0.4, -0.3, -0.5, -0.2]})

    Args:
        best_max_depths_all (Dict): Python dictionary containing best max depth
            as keys, and error values in list as values.

    Returns:
        (Pandas dataframe): Pandas dataframe with dictionary in tidy long data
            format.
    '''

    max_depths_all = []
    scores_all = []

    for key, value in best_max_depths_all.iteritems():
        max_depths = [key] * len(value)
        scores = list(value)

        max_depths_all += max_depths
        scores_all += scores

    return pd.DataFrame({'Max Depth': max_depths_all, 'Loss': scores_all})


def plot_all_distributions(best_depths):
    '''Takes a list of pre-trained GridScoreObjects, extracts error rates for
    each model from each "best" folds. Calculates the average error, and
    standard deviation from the best depth in each 5-fold run to plot as well.

     From each model, each fold's error is plotted. The average of each depth
     and standard deviation are also plotted.

    Args:
        best_depths (List): List of GridScoreCV objects

    Returns:
        null

    '''

    # iter - Store number of runs/GridSearchCV objects
    # k - Pull out the first GridSearchCV object and grab the k-folds used.
    best_depths_params = {'iter': len(best_depths),
                          'k': best_depths[0].cv.n_folds}

    best_max_depths_all = get_xval_data_from_list_Gridscores(best_depths)

    best_max_depths_df = unpack_dict_to_df(best_max_depths_all)

    best_max_depths_stats = best_max_depths_df.groupby(['Max Depth'], as_index=False)\
        .aggregate([np.mean, np.std])\
        .reset_index()

    # Add a bit of jitter to the max depth for each fold to avoid overplotting
    best_max_depths_df['Jitter Max Depth'] = best_max_depths_df['Max Depth']\
        .apply(lambda x: x*(1+np.random.uniform(-0.02, 0.02)))

    plt.figure(figsize=(12, 9))
    plt.xlim([0, 11])
    plt.xticks(range(0, 11))

    plt.gca().invert_yaxis()

    plt.scatter(best_max_depths_df['Jitter Max Depth'],
                best_max_depths_df['Loss'], alpha=0.3,
                color='#6734bd')

    plt.errorbar(best_max_depths_stats['Max Depth'],
                 best_max_depths_stats['Loss']['mean'],
                 yerr=best_max_depths_stats['Loss']['std'],
                 color='#bd6734',
                 capsize=10,
                 elinewidth=2,
                 markeredgewidth=1)

    plt.scatter(best_max_depths_stats['Max Depth'],
                best_max_depths_stats['Loss']['mean'],
                color='#bd6734',
                s=70, marker='o')

    plt.title('Mean Squared Error of best model of each fold \n'
              ' in each run:  {k}-fold CV,  {iter} iterations'.format(**best_depths_params))

    plt.xlabel('Best Max Depth')
    plt.ylabel('Mean Squared Error')

    plt.show()


def plot_hist_best_max_depths(best_depths):
    '''Takes a list of pre-trained GridScoreObjects, extracts the max_depth
    value found in each object, then plots a histogram from the resulting
    values.

    Args:
        best_depths (List): List of GridScoreCV objects

    Returns:
        null
    '''
    best_depths_list = [model.best_params_['max_depth'] for model in best_depths]

    plt.figure(figsize=(12, 9))
    # Remove unneeded plot frame lines and ticks.
    ax = plt.subplot(1, 1, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


    # Move the ticks over 0.5 units to be centered under bar; add ticks for every depth
    plt.xticks(np.arange(1.5, 11.5, 1), range(1, 12), fontsize=14)
    plt.xlabel('Best Max Depth', fontsize=16)
    plt.ylabel('Count', fontsize=16)

    plt.hist(best_depths_list, color="#3F5D7D", bins=range(1, 12), edgecolor='k')
    plt.title('Best "Max Depth": {0} Runs of GridSearchCV'.format(len(best_depths)))

    plt.show()


def plot_prediction_distribution(best_depths):
    ''' Takes a list of pre-trained GridScoreObjects, makes a predictions on one
    predefined data point with each object, then plots kernel density estimation
    plot with the predictions. Adds average value.

    Args:
        best_depths (List): List of GridScoreCV objects

    Returns:
        null
    '''


    x = np.array([11.95, 0.00, 18.100, 0, 0.6590, 5.6090, 90.00, 1.385, 24, 680.0, 20.20, 332.09, 12.13]).reshape(1, -1)
    y_predictions = [model.predict(x)[0] for model in best_depths]
    y_predictions_mean = np.mean(y_predictions)

    plt.figure(figsize=(12, 9))
    # Remove unneeded plot frame lines and ticks.
    ax = plt.subplot(1, 1, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    # http://stackoverflow.com/questions/15415455/plotting-probability-density-function-by-sample-with-matplotlib
    # https://en.wikipedia.org/wiki/Kernel_density_estimation
    # In statistics, kernel density estimation (KDE) is a non-parametric way to estimate the probability density function of a random variable.

    kde = gaussian_kde(y_predictions)
    # these are the values over which your kernel will be evaluated
    # Add 2 to min/max to extend the curve
    dist_space = np.linspace(min(y_predictions)-2, max(y_predictions)+2, 100)
    y = kde(dist_space)



    plt.plot(dist_space, y, color="#348ABD")
    plt.fill_between(dist_space, 0, y, color='#348ABD', alpha=0.4)

    # Plot the average prediction of all of the models.
    plt.vlines(y_predictions_mean,
               ymin=0, ymax=1,
               colors=['#bd6734'],
               linestyles="--",
               lw=2,
               label='Avg. Predicted Price: {:,}'.format(int(round(y_predictions_mean, 3)*1000)))

    # Plot the actual predictions as well.
    dist_space_predictions = kde(y_predictions)

    plt.vlines(y_predictions,
               ymin=0,
               ymax=dist_space_predictions,
               colors=['#348ABD'],
               linestyles='--',
               lw=0.5,
               label='Each Predicted Price')

    plt.legend()
    plt.title(
        'Estimated Probability Density for Predictions of Each Best Max Depth Model \n'
        ' over {0} different runs of gridSearch'.format(len(best_depths)))
    plt.xlabel('Predicted Value', fontsize=16)
    plt.ylabel('Probability', fontsize=16)
    plt.xticks(fontsize=14)


def plot_price(city_data):
    ''' Plots a simple histogram with custom bins, xlabels, and labels.

    Args:
        city_data (Numpy array): Boston Dataset from sklearn.datasets

    Returns:
        null

    '''
    plt.figure(figsize=(12,9))
    bins = range(5, 51, 1)
    plt.hist(city_data.target, bins, color="#3F5D7D")
    plt.xticks(range(5, 51, 2))
    plt.title('Histogram of Full Dataset Prices')
    plt.xlabel('Price (in $1000 bins)')
    plt.ylabel('Frequency')
    plt.show()


def main():
    """Analyze the Boston housing data. Evaluate and validate the
    performanance of a Decision Tree regressor on the housing data.
    Fine tune the model to make prediction on unseen data."""

    # Load data
    city_data = load_data()

    # Explore the data
    explore_city_data(city_data)

    # Plot histogram of price data
    plot_price(city_data)

    # Training/Test dataset split
    X_train, y_train, X_test, y_test = split_data(city_data)

    # Learning Curve Graphs

    # Move logic/flow control out of main() and pass list as argument to our learning_curve function.
    max_depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    learning_curve(max_depths, X_train, y_train, X_test, y_test)

    # Model Complexity Graph
    model_complexity(X_train, y_train, X_test, y_test)

    # Tune and predict one Model
    fit_predict_model(city_data)

    # Tune and predict Model over many iterations.
    many_iter = fit_predict_many(city_data, n=100)

    # Plot best max depth over many runs
    plot_hist_best_max_depths(many_iter)

    # Plot kernel density estimate for predictions from many models
    plot_prediction_distribution(many_iter)

    # Plot spread of error of all models
    plot_all_distributions(many_iter)


if __name__ == "__main__":
    main()
