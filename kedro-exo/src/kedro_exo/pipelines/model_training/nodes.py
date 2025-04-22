"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.19.12
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn import set_config
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
import optuna
import pickle
from typing import Tuple

def split_data(
        df: pd.DataFrame, 
        test_size: float = 0.15,
        random_state: int = 42
) -> Tuple[Tuple
           pd.DataFrame,  pd.DataFrame, pd.DataFrame,
           pd.Series, pd.Series, pd.Series
           ]:
    """Split DataFrame into train, validation and test sets for features and target.

    Args:
        df: Datafraime containing features and target column.
        test_size: Fraction of data to reserve for the test split.
        random_state: Seed for reproducible splits.

    Returns:
        Tuple in order:
        - X_train: DataFrame of training features
        - X_val: DataFrame of validation features
        - X_test: DataFrame of test features
        - y_train: Series of trainig target
        - y_val: Series of validation target
        - y_test: Series of test target
    """
    # Separate input features and target
    X = df.drop("Transported", axis=1)
    y = df["Transported"]
    # Split off the test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    # Split the remaining part into train and validation
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=test_size, random_state=random_state)
    return X_train, X_val,  X_test, y_train, y_val, y_test

def 
