"""
This is a boilerplate pipeline 'model_evaluation'
generated using Kedro 0.19.12

Steps:

1. transform_test_set: apply feature-engineering to X_test and extract the fitted LightGBM estimator.
2. predict_and_metrics: compute confusion matrix and ROC-AUC.
3. plot_confusion_matrix: generate confusion-matrix heatmap.
4. plot_roc_curve: generate ROC curve.
5. interpret_model: calculate SHAP values.
6. shap_summary_plot: SHAP summary (dot) plot.
7. shap_bar_plot: SHAP feature-importance bar plot.
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa

from .nodes import (
    transform_test_set,
    predict_and_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    interpret_model,
    shap_summary_plot,
    shap_bar_plot,
)


def create_pipeline(**kwargs) -> Pipeline:  # noqa: D401
    """Assemble the model_evaluation pipeline grouped under the model_evaluation namespace."""
    return pipeline(
        [
            # ───── preparation ──────────────────────────────────────
            node(
                func=transform_test_set,
                inputs=["best_model", "X_test"],
                outputs=["X_test_transformed", "model"],
                name="transform_test_set",
                tags=("prep",),
            ),
            # ───── metrics ───────────────────────────────────────────
            node(
                func=predict_and_metrics,
                inputs=["model", "X_test_transformed", "y_test"],
                outputs=["cm", "auc_score", "fpr", "tpr"],
                name="predict_and_metrics",
                tags=("metrics",),
            ),
            # ───── plots: confusion matrix & ROC ─────────────────────
            node(
                func=plot_confusion_matrix,
                inputs="cm",
                outputs="confusion_matrix_plot",
                name="plot_confusion_matrix",
                tags=("plot", "cm"),
            ),
            node(
                func=plot_roc_curve,
                inputs=["fpr", "tpr", "auc_score"],
                outputs="roc_curve_plot",
                name="plot_roc_curve",
                tags=("plot", "roc"),
            ),
            # ───── SHAP values and plots ─────────────────────────────
            node(
                func=interpret_model,
                inputs=["model", "X_test_transformed"],
                outputs="shap_values",
                name="interpret_model",
                tags=("shap",),
            ),
            node(
                func=shap_summary_plot,
                inputs=["shap_values", "X_test_transformed"],
                outputs="shap_summary_plot",
                name="shap_summary_plot",
                tags=("plot", "shap"),
            ),
            node(
                func=shap_bar_plot,
                inputs=["shap_values", "X_test_transformed"],
                outputs="shap_bar_plot",
                name="shap_bar_plot",
                tags=("plot", "shap"),
            ),
        ],
        namespace="model_evaluation",
        inputs=["best_model", "X_test", "y_test"],
        outputs=[
            "cm",
            "auc_score",
            "confusion_matrix_plot",
            "roc_curve_plot",
            "shap_values",
            "shap_summary_plot",
            "shap_bar_plot",
        ],
    )
