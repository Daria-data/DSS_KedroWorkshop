import importlib

import optuna
import pandas as pd
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


def tune_hyperparameters(  # noqa
    feature_engineering: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cfg: dict[str, object],
) -> dict[str, object]:
    """Dynamically load model class and tune its parameters as per config.

    Description:
        1. Load n_trials and random_state from config.
        2.Start with cfg["init_params"] as base.
        3. Build search space from cfg["tune_hyperparameters"].
        4. Run Optuna study and return merged best_params.

    Args:
        feature_engineering: Preprocessing transformer.
        X_train, y_train: Training split.
        X_val, y_val: Validation split.
        cfg: Dict with 'class', 'init_params', 'tune_hyperparameters'.

    Returns:
        dict[str, object]: Mapping of hyper-parameters giving best validation score.
    """
    # --- 1. unpack config --------------------------------------------------
    model_path: str = cfg["class"]              # import path
    init_params: dict[str, object] = cfg.get("init_params", {})
    tp_cfg = cfg["tune_hyperparameters"]
    n_trials = tp_cfg.get("n_trials")
    random_state = tp_cfg.get("random_state")

    # --- 2. dynamic import -------------------------------------------------
    module_name, class_name = model_path.rsplit(".", 1)
    ModelClass = getattr(importlib.import_module(module_name), class_name)

    def objective(trial: optuna.Trial) -> float:
        # copy base params for each trial
        trial_params = init_params.copy()

        for name, spec in tp_cfg.items():
            if name in ("n_trials", "random_state"):
                continue
            if isinstance(spec, list):
                # categorical parameter
                trial_params[name] = trial.suggest_categorical(name, spec)
            elif isinstance(spec, dict) and "low" in spec and "high" in spec:
                low, high = spec["low"], spec["high"]
                if isinstance(low, int) and isinstance(high, int):
                    trial_params[name] = trial.suggest_int(name, low, high)
                else:
                    log_flag = bool(spec.get("log", False))
                    trial_params[name] = trial.suggest_float(name, low, high, log=log_flag)

        model_pipeline = Pipeline(
            [
                ("feature_engineering", feature_engineering),
                ("model", ModelClass(**trial_params)),
            ]
        )

        model_pipeline.fit(X_train, y_train.values.ravel())
        return model_pipeline.score(X_val, y_val.values.ravel())

    # --- 4. run study ------------------------------------------------------
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials)

    return study.best_params


def train_final_model(  # noqa
    feature_engineering: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    best_params: dict[str, object],
    cfg: dict[str, object],
) -> tuple[Pipeline, float]:
    """Build the final model pipeline from config and tuned params, fit it, and report validation accuracy.

    Description:
        1. Dynamically load the model class from cfg["model"]["class"].
        2. Merge cfg["model"]["init_params"] with best_params.
        3. Construct a Pipeline([("feature_engineering", ...), ("model", ModelClass(**merged_params))]).
        4. Fit and evaluate on the validation split.

    Args:
        feature_engineering: Transformer that one-hot-encodes categorical
            features and scales numerical ones.
        X_train: Training features.
        y_train: Training target.
        X_val: Validation features.
        y_val: Validation target.
        best_params: Hyper-parameters returned by 'tune_hyperparameters'.
        cfg: Same branch dict used for tuning.

    Returns:
        tuple:
            * model_pipeline - fitted pipeline ready for inference;
            * val_accuracy - accuracy on the validation split.
    """
    # 1. Import model class dynamically
    module_name, class_name = cfg["class"].rsplit(".", 1)
    ModelClass = getattr(importlib.import_module(module_name), class_name)

    # 2. Prepare constructor params
    init_params = dict(cfg.get("init_params", {}))
    model_params = {**init_params, **best_params}

    # 3. Build and fit pipeline
    model_pipeline = Pipeline(
        [
            ("feature_engineering", feature_engineering),
            ("model", ModelClass(**model_params)),
        ]
    )
    model_pipeline.fit(X_train, y_train.values.ravel())
    val_accuracy = model_pipeline.score(X_val, y_val.values.ravel())

    return model_pipeline, val_accuracy
