"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.19.12
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn import set_config
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
import optuna

# Ensure all transformers output pandas DataFrame with column names
set_config(transform_output="pandas")


def split_data(
    df: pd.DataFrame, test_size: float = 0.15, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
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

    # First split off the test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    # Split the remaining part into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=test_size, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_column_groups(_: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Identify numericak and categorical columns.

    Args:
        _ : Not used.

    Returns:
        Tuple with two lists:
        - ''cat_cols'' - categorical feature names.
        - ''num_cols'' - numerical feature names.
    """

    num_cols = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    cat_cols = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side"]
    return num_cols, cat_cols


def creat_preprocessor(cat_cols: list[str], num_cols: list[str]) -> ColumnTransformer:
    """Build ColumnTransformer for feature engineering.
    - Categorical columns -> One-hot encoding.
    - Numerical columns -> Min-max scaling.
    - Remaining columns -> passthrough.

    Args:
        cat_cols: Names of categorical feature columns.
        num_cols: Names of numerical feature columns.

    Returns:
        Configured ColumnTransformer objet.
    """

    preprocessor = ColumnTransformer(
        transformers=[
            # (step name, transformer, column list to apply transformation to)
            ("onehot_encoding", OneHotEncoder(sparse_output=False), cat_cols),
            ("minmax_scaling", MinMaxScaler(), num_cols),
        ],
        remainder="passthrough",
    )
    return preprocessor


def make_pipeline(cat_cols: list[str], num_cols: list[str]) -> Pipeline:
    """Combine feature pipeline with an un-tuned ``LGBMClassifier``.

    Args:
        cat_cols: Categorical feature names.
        num_cols: Numerical feature names.

    Returns:
        sklearn-Pipeline with two steps:
        'feature_engineering' and  'model'.
    """
    feature_engineering = creat_preprocessor(cat_cols, num_cols)

    model_pipeline = Pipeline(
        steps=[
            ("feature_engineering", feature_engineering),
            ("model", LGBMClassifier()),
        ]
    )
    return model_pipeline


def tune_hyperparameters(  # noqa: PLR0913
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cat_cols: list[str],
    num_cols: list[str],
    n_trials: int = 100,
    random_state: int = 42,
) -> dict[str, any]:
    """Search for the best LightGBM hyperparameters with Optuna.

    Args:
        X_train: Training features.
        y_train: Training target.
        X_val: Validation features.
        y_val: Validation target.
        cat_cols: Categorical feature names.
        num_cols: Numerical feature names.
        n_trials: Number of Optuna trials.
        random_state: Seed for reproducibility.

    Returns:
        Dictionary of best hyperparameters (prefixed with ``model__``).
    """

    def _objective(trial: optuna.Trial) -> float:
        params = {
            # meta-parameters
            "model__objective": trial.suggest_categorical(
                "model__objective", ["binary"]
            ),
            "model__metric": trial.suggest_categorical(
                "model__metric", ["binary_logloss"]
            ),
            "model__boosting_type": trial.suggest_categorical(
                "model__boosting_type", ["gbdt"]
            ),
            "model__verbosity": trial.suggest_categorical("model__verbosity", [-1]),
            "model__random_state": random_state,
            # hyperparameters
            "model__num_leaves": trial.suggest_int("model__num_leaves", 10, 100),
            "model__learning_rate": trial.suggest_float(
                "model__learning_rate", 0.01, 0.1, log=True
            ),
            "model__feature_fraction": trial.suggest_float(
                "model__feature_fraction", 0.1, 1.0
            ),
            "model__bagging_fraction": trial.suggest_float(
                "model__bagging_fraction", 0.1, 1.0
            ),
            "model__bagging_freq": trial.suggest_int("model__bagging_freq", 1, 10),
            "model__min_child_samples": trial.suggest_int(
                "model__min_child_samples", 1, 50
            ),
        }

        pipeline = make_pipeline(cat_cols, num_cols).set_params(**params)
        pipeline.fit(X_train, y_train)
        return pipeline.score(X_val, y_val)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params


def train_model(
    best_params: dict[str, any],
    cat_cols: list[str],
    num_cols: list[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Fit a model with the best hyperparameters.

    Args:
        best_params: Dictionary from ``tune_hyperparameters``.
        cat_cols: Categorical feature names.
        num_cols: Numerical feature names.
        X_train: Training features.
        y_train: Training target.

    Returns:
        Trained sklearn-Pipeline.
    """
    model_pipeline = make_pipeline(cat_cols, num_cols).set_params(**best_params)
    model_pipeline.fit(X_train, y_train)
    return model_pipeline


def evaluate_model(
    model_pipeline: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> float:
    """Compute accuracy of the model on the validation set.

    Args:
        model_pipeline: Pipeline from the train_model.
        X_val: Validation features.
        y_val: Validation target.

    Returns:
        Accuracy score in the range 0 - 1.
    """
    return model_pipeline.score(X_val, y_val)
