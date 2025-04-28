import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn import set_config
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline

set_config(transform_output="pandas")


def transform_test_set(
    model_pipeline: Pipeline,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, LGBMClassifier]:
    """Apply the feature-engineering step to X_test.

    Args:
        model_pipeline: Pipeline loaded from best_model.pkl.
        X_test: Raw test-split features produced by model_training pipeline.

    Returns:
        tuple:
            - X_test_transformed: transformed features (overwriting the raw version);
            - model: underlying LGBMClassifier estimator.
    """
    X_test_transformed = model_pipeline["feature_engineering"].transform(X_test)
    model = model_pipeline["model"]
    return X_test_transformed, model


def predict_and_metrics(
    model: LGBMClassifier,
    X_test_transformed: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Generate predictions and compute evaluation metrics.

    Args:
        model: Trained LightGBM classifier.
        X_test_transformed: Encoded and scaled test features.
        y_test: True target values for the test split.

    Returns:
        tuple:
            * cm: '2x2' confusion-matrix normalised by true class.
            * auc_score: ROC-AUC value on X_test.
            * fpr: array of false-positive-rate values (for ROC).
            * tpr: array of true-positive-rate values (for ROC).
    """
    y_pred = model.predict(X_test_transformed)
    y_pred_probs = model.predict_proba(X_test_transformed)[:, 1]

    cm = confusion_matrix(y_test, y_pred, normalize="true")
    # Calculate the false positive rate, true positive rate
    fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
    auc_score = roc_auc_score(y_test, y_pred_probs)

    return cm, auc_score, fpr, tpr


def plot_confusion_matrix(cm: np.ndarray) -> any:
    """Create a heatmap of the confusion matrix.

    Args:
        cm: '2x2' confusion matrix from function 'predict_and_metrics'.

    Returns:
        Matplotlib figure with annotated heatmap.
    """
    classes = ["Not Transported", "Transported"]
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")

    ax.set(
        title="Confusion Matrix",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=range(2),
        yticks=range(2),
        xticklabels=classes,
        yticklabels=classes,
    )
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{cm[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
) -> any:
    """Plot the ROC curve for the classifier.

    Args:
        fpr: False-positive-rate values from 'predict_and_metrics'.
        tpr: True-positive-rate values from 'predict_and_metrics'.
        auc_score: ROC-AUC value for annotation.

    Returns:
        Matplotlib figure containing the ROC curve.
    """
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC (AUC = {auc_score:.2f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="ROC Curve",
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def interpret_model(
    model: LGBMClassifier,
    X_test_transformed: pd.DataFrame,
) -> np.ndarray:
    """Calculate SHAP values for X_test using the LightGBM estimator.

    Args:
        model: Trained LightGBM classifier.
        X_test_transformed: Encoded and scaled test features.

    Returns:
        2-D numpy array of SHAP values (shape = n_samples * n_features, same shape as X_test).
    """
    booster = model.booster_
    if "objective" not in booster.params:
        booster.params["objective"] = model.get_params().get("objective", "binary")
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_test_transformed)
    return shap_values


def shap_summary_plot(
    shap_values: np.ndarray,
    X_test_transformed: pd.DataFrame,
) -> any:
    """Create SHAP summary (dot) plot.

    Args:
        shap_values: SHAP values from "compute_shap_values".
        X_test_transformed: Encoded and scaled test features.

    Returns:
        Matplotlib figure with the summary plot.
    """
    shap.summary_plot(shap_values, X_test_transformed, show=False, plot_size=0.2)
    return plt.gcf()


def shap_bar_plot(
    shap_values: np.ndarray,
    X_test_transformed: pd.DataFrame,
) -> any:
    """Create SHAP feature-importance bar plot.

    Args:
        shap_values: SHAP values from "compute_shap_values".
        X_test_transformed: Encoded and scaled test features.

    Returns:
        Matplotlib figure with the bar plot.
    """
    shap.summary_plot(
        shap_values, X_test_transformed, plot_type="bar", show=False, plot_size=0.2
    )
    return plt.gcf()
