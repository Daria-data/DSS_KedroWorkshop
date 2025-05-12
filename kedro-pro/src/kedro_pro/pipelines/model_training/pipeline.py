"""
Multi-model training pipeline.

1. Split the cleaned dataset (once).
2. Build feature engineering transformer (once).
3. For every model key under `models` in parameters:
   • Run Optuna tuning.
   • Fit final model.
"""
from pathlib import Path

import yaml

from kedro.pipeline import node, Pipeline, pipeline  # noqa

from .nodes import (
    split_data,
    define_feature_columns,
    make_feature_pipeline,
    tune_hyperparameters,
    train_final_model,
)

# Helper: read conf/base/parameters_model_training.yml to get model list
PARAMS_FILE = Path(__file__).resolve().parents[3] / "conf" / "base" / "parameters_model_training.yml"
with open(PARAMS_FILE, "r", encoding="utf-8") as fp:
    MODELS = list(yaml.safe_load(fp)["models"].keys())

def _make_model_nodes(model_key: str) -> list[node]:
    """Return tuning & training nodes for a single model."""
    suffix = f"_{model_key.lower()}"

    return [
        node(
            func=tune_hyperparameters,
            inputs=[
                "feature_engineering",
                "X_train",
                "y_train",
                "X_val",
                "y_val",
                f"params:models.{model_key}",
            ],
            outputs=f"best_params{suffix}",
            name=f"tune_hyperparameters{suffix}",
            tags=("tuning", model_key),
        ),
        node(
            func=train_final_model,
            inputs=[
                "feature_engineering",
                "X_train",
                "y_train",
                "X_val",
                "y_val",
                f"best_params{suffix}",
                f"params:models.{model_key}",
            ],
            outputs=[f"best_model{suffix}", f"val_accuracy{suffix}"],
            name=f"train_final_model{suffix}",
            tags=("train", model_key),
        ),
    ]

def create_pipeline(**kwargs) -> Pipeline:
    """Build the multi-model training pipeline."""
    common_nodes = [ # same input for all models
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
        ]
    model_nodes = []
    for model in MODELS:
        model_nodes.extend(_make_model_nodes(model))

    return Pipeline(common_nodes + model_nodes, namespace="model_training")

    # return pipeline(
    #     [    #         node(
    #             func=tune_hyperparameters,
    #             inputs=[
    #                 "feature_engineering",
    #                 "X_train",
    #                 "y_train",
    #                 "X_val",
    #                 "y_val",
    #                 #"parameters_model_training",
    #                 "params:models.LightGBM",
    #             ],
    #             outputs="best_params",
    #             name="tune_hyperparameters",
    #             tags=("tuning",),
    #         ),
    #         node(
    #             func=train_final_model,
    #             inputs=[
    #                 "feature_engineering",
    #                 "X_train",
    #                 "y_train",
    #                 "X_val",
    #                 "y_val",
    #                 "best_params",
    #                 #"parameters_model_training",
    #                 "params:models.LightGBM",
    #             ],
    #             outputs=["best_model", "val_accuracy"],
    #             name="train_final_model",
    #             tags=("train",),
    #         ),
    #     ],
