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

MODEL_KEYS = ["LightGBM", "RandomForest"]

def _make_eval_nodes(model_key: str) -> list[node]:
    """Return evaluation nodes for one model suffix."""
    sfx = f"_{model_key.lower()}"

    return [
        # 1. Preparation
        node(
            transform_test_set,
            inputs=[f"best_model{sfx}", "X_test"],
            outputs=[f"X_test_transformed{sfx}", f"model{sfx}"],
            name=f"transform_test_set{sfx}",
        ),
        # 2. Metrics
        node(
            predict_and_metrics,
            inputs=[f"model{sfx}", f"X_test_transformed{sfx}", "y_test"],
            outputs=[f"cm{sfx}", f"auc_score{sfx}", f"fpr{sfx}", f"tpr{sfx}"],
            name=f"predict_and_metrics{sfx}",
        ),
        # 3. Confusion-matrix heatmap
        node(
            plot_confusion_matrix,
            inputs=f"cm{sfx}",
            outputs=f"confusion_matrix_plot{sfx}",
            name=f"plot_confusion_matrix{sfx}",
        ),
        # 4. ROC curve
        node(
            plot_roc_curve,
            inputs=[f"fpr{sfx}", f"tpr{sfx}", f"auc_score{sfx}"],
            outputs=f"roc_curve_plot{sfx}",
            name=f"plot_roc_curve{sfx}",
        ),
        # 5. SHAP values
        node(
            interpret_model,
            inputs=[f"model{sfx}", f"X_test_transformed{sfx}"],
            outputs=f"shap_values{sfx}",
            name=f"interpret_model{sfx}",
        ),
        # 6. SHAP plots
        node(
            shap_summary_plot,
            inputs=[f"shap_values{sfx}", f"X_test_transformed{sfx}"],
            outputs=f"shap_summary_plot{sfx}",
            name=f"shap_summary_plot{sfx}",
        ),
        node(
            shap_bar_plot,
            inputs=[f"shap_values{sfx}", f"X_test_transformed{sfx}"],
            outputs=f"shap_bar_plot{sfx}",
            name=f"shap_bar_plot{sfx}",
        ),
    ]

def create_pipeline(**kwargs) -> pipeline:  # noqa: D401
    """Assemble the model_evaluation pipeline grouped under the model_evaluation namespace."""
    nodes: list[node] = []
    for key in MODEL_KEYS:
        nodes.extend(_make_eval_nodes(key))

    return pipeline(nodes, namespace="model_evaluation")
