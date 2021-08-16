import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef


def print_unique_values_from_columns(df: pd.DataFrame, column: str) -> None:
    """
    - Prints number of unique values in a Dataframe column.
    - Prints all unique values in a Dataframe column.
     :return None
     """
    unique_values = df[column].unique()
    number_unique_values = df[column].nunique()
    print(f""" 
The {column}-column contains:
Number of unique values:{number_unique_values}
Values: {unique_values}""")


def normalizing_array(array:np.ndarray) -> np.ndarray:
    """Normalizes a Numpy array.

    :return Normalized Numpy array.
    """
    norm = np.linalg.norm(array)
    normalize_array = array / norm
    return normalize_array


def predicting_scoring_model(forest_fitted: RandomForestClassifier, X_train: np.ndarray, y_train: np.ndarray,
                                             X_test: np.ndarray, y_test: np.ndarray,
                                             confusion_matrix_title: str):
    """ Prints and plots different model evaluations techniques.

    Prints the score on the training set, the score on the test set,
    the cross-validation-score, Matthews-corrcoëf,
    and plots the confusion matrix with title as parameter.

    :return Fitted RandomForestClassifier-model. """

    # Computing scores
    train_score = forest_fitted.score(X_train, y_train)
    test_score = forest_fitted.score(X_test, y_test)
    y_pred = forest_fitted.predict(X_test)
    cv_score = cross_val_score(forest_fitted, X_train, y_train, cv=5)

    # Printing the scores.
    print(f"Score on training set: {train_score}")
    print(f"Score on test set: {test_score}")
    print("""
cross_validation_score:""")
    print("Accuracy: {:.2f}% (+/- {:.2f})".format(cv_score.mean() * 100, cv_score.std() * 100))
    print("""
classification_report:""")
    print(classification_report(y_test, y_pred))
    print("""
matthews_corrcoef:""")
    print(matthews_corrcoef(y_test, forest_fitted.predict(X_test)))

    # Plotting confusion matrix
    plot_confusion_matrix(forest_fitted, X_test, y_test, values_format = '.0f')
    plt.title(f"Confusion matrix on {model_title}")
    plt.show()