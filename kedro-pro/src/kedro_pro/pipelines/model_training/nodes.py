import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Ensure transformers return DataFrame, not ndarray
set_config(transform_output="pandas")


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Split the cleaned dataset into training, validation and test sets.

    The function:
    1. Holds out 15 % of the data for the final test set.
    2. Splits the remaining data into training and validation
       using the same 15 % fraction.
    3. Keeps the random state fixed to ensure reproducibility.

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    TEST_SIZE = 0.15
    RANDOM_STATE = 42

    X = df.drop(columns=["Transported"])
    y = df["Transported"]

    # Step 1 – test split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Step 2 – validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def define_feature_columns() -> tuple[list[str], list[str]]:
    """Return the categorical and numerical feature lists."""
    cat_columns = [
        "HomePlanet",
        "CryoSleep",
        "Destination",
        "VIP",
        "Deck",
        "Side",
    ]
    num_columns = [
        "Age",
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "VRDeck",
    ]
    return cat_columns, num_columns


def make_feature_pipeline(
    cat_columns: list[str],
    num_columns: list[str],
) -> ColumnTransformer:
    """Build a preprocessing pipeline that

    1. One-hot-encodes categorical features.
    2. Scales numerical features to [0, 1].

    Args:
        cat_columns: Names of categorical feature columns.
        num_columns: Names of numerical feature columns.

    Returns:
        Configured ColumnTransformer instance.
    """
    return ColumnTransformer(
        transformers=[
            (
                "onehot_encoding",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_columns,
            ),
            ("minmax_scaling", MinMaxScaler(), num_columns),
        ]
    )


def tune_hyperparameters(
    feature_engineering: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict[str, float]:
    """Search for the LightGBM hyper-parameters that maximise validation accuracy.

    The search space replicates the ranges used during experimentation.
    Optuna evaluates N_TRIALS random-seed-controlled trials and returns
    the parameter set that gives the highest accuracy on the
    validation split.

    Args:
        feature_engineering: Fully configured ColumnTransformer that
            handles one-hot encoding and feature scaling.
        X_train: Training features.
        y_train: Training target.
        X_val: Validation features.
        y_val: Validation target.

    Returns:
        dict[str, float]: Mapping of parameter names to their optimal
        values, ready to be passed into ``Pipeline.set_params``.
    """
    N_TRIALS = 100
    RANDOM_STATE = 42

    def objective(trial: optuna.Trial) -> float:
        params = {
            # Meta-parameters (fixed choices)
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
            "model__random_state": trial.suggest_categorical(
                "model__random_state", [RANDOM_STATE]
            ),
            # Hyper-parameters (to be tuned)
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

        model_pipeline = Pipeline(
            [
                ("feature_engineering", feature_engineering),
                ("model", LGBMClassifier()),
            ]
        ).set_params(**params)

        model_pipeline.fit(X_train, y_train.values.ravel())
        return model_pipeline.score(X_val, y_val.values.ravel())

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=N_TRIALS)

    return study.best_params


def train_final_model(  # noqa
    feature_engineering: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    best_params: dict[str, float],
) -> tuple[Pipeline, float]:
    """Build the final LightGBM pipeline, fit it, and report validation accuracy.

    The pipeline combines the previously created 'feature_engineering'
    transformer with an LGBMClassifier initialised by 'best_params'.

    Args:
        feature_engineering: Transformer that one-hot-encodes categorical
            features and scales numerical ones.
        X_train: Training features.
        y_train: Training target.
        X_val: Validation features.
        y_val: Validation target.
        best_params: Hyper-parameters returned by 'tune_hyperparameters'.

    Returns:
        tuple:
            * model_pipeline - fitted pipeline ready for inference;
            * val_accuracy - accuracy on the validation split.
    """
    model_pipeline = Pipeline(
        [
            ("feature_engineering", feature_engineering),
            ("model", LGBMClassifier()),
        ]
    ).set_params(**best_params)

    model_pipeline.fit(X_train, y_train)
    val_accuracy = model_pipeline.score(X_val, y_val)

    return model_pipeline, val_accuracy
