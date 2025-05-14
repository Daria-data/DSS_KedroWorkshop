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

def _get_model_keys() -> list[str]:
     """Load list of models from parameters_model_training.yml"""
     project_root = Path(__file__).resolve().parents[4]
     params_path = project_root / "conf" / "base" / "parameters_model_training.yml"
     with open(params_path, encoding="utf-8") as fp:
        MODELS = list(yaml.safe_load(fp)["models"].keys())
        return MODELS

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
    models = _get_model_keys()
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
    for model in models:
        model_nodes.extend(_make_model_nodes(model))

    return pipeline(
        common_nodes + model_nodes,
        namespace="model_training",
        inputs=["clean_train"],
        parameters={f"models.{m}" for m in models},
    )