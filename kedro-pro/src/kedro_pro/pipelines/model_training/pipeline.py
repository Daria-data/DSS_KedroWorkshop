"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.19.12

1. Splits the cleaned dataset into training, validation and test sets.
2. Defines categotrical and numerical feature lists.
3. Builds the feature-engineering transformer.
4. Runs Optuna to select the best LightGBM hyper-parameters.
5. Fits the final model to reports validation accuracy.
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa

from .nodes import (
    split_data,
    define_feature_columns,
    make_feature_pipeline,
    tune_hyperparameters,
    train_final_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Assemble the model-training pipeline grouped under the model_training namespace.

    Returns:
        Kedro Pipeline instance ready to be registered.
    """
    return pipeline(
        [
            node(
                func=split_data,
                inputs="clean_train",
                outputs=[
                    "X_train",
                    "y_train",
                    "X_val",
                    "y_val",
                    "X_test",
                    "y_test",
                ],
                name="split_data",
                tags=("split",),
            ),
            node(
                func=define_feature_columns,
                inputs=None,
                outputs=["cat_columns", "num_columns"],
                name="define_feature_columns",
                tags=("features",),
            ),
            node(
                func=make_feature_pipeline,
                inputs=["cat_columns", "num_columns"],
                outputs="feature_engineering",
                name="make_feature_pipeline",
                tags=("preprocess",),
            ),
            node(
                func=tune_hyperparameters,
                inputs=[
                    "feature_engineering",
                    "X_train",
                    "y_train",
                    "X_val",
                    "y_val",
                ],
                outputs="best_params",
                name="tune_hyperparameters",
                tags=("tuning",),
            ),
            node(
                func=train_final_model,
                inputs=[
                    "feature_engineering",
                    "X_train",
                    "y_train",
                    "X_val",
                    "y_val",
                    "best_params",
                ],
                outputs=["best_model", "val_accuracy"],
                name="train_final_model",
                tags=("train",),
            ),
        ],
        namespace="model_training",
        inputs=["clean_train"],
        outputs=["best_model", "val_accuracy"],
    )
