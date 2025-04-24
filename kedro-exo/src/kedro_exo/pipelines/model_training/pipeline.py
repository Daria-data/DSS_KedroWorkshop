from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import (
    split_data,
    get_column_groups,
    creat_preprocessor,
    tune_hyperparameters,
    train_model,
    evaluate_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # split raw cleaned dataframe into train / val / test
            node(
                split_data,
                inputs="clean_train",
                outputs=[
                    "X_train",
                    "X_val",
                    "X_test",
                    "y_train",
                    "y_val",
                    "y_test",
                ],
                name="split_data",
            ),
            # detect categorical and numerical columns
            node(
                get_column_groups,
                inputs="X_train",
                outputs=["cat_cols", "num_cols"],
                name="get_column_groups",
            ),
            # build feature engineering transformer
            node(
                creat_preprocessor,
                inputs=["cat_cols", "num_cols"],
                outputs="preprocessor",
                name="creat_preprocessor",
            ),
            # optuna tuning - looking for the best hyperparameters
            node(
                tune_hyperparameters,
                inputs=["X_train", "y_train", "X_val", "y_val", "cat_cols", "num_cols"],
                outputs="best_params",
                name="tune_hyperparameters",
                tags=["optuna"],
            ),
            # train finam model using best params
            node(
                train_model,
                inputs=[
                    "best_params",
                    "cat_cols",
                    "num_cols",
                    "X_train",
                    "y_train",
                ],
                outputs="model_pipeline",
                name="train_model",
            ),
            # evaluate on validation set
            node(
                evaluate_model,
                inputs=["model_pipeline", "X_val", "y_val"],
                outputs="val_accuracy",
                name="evaluate_model",
            ),
        ]
    )
