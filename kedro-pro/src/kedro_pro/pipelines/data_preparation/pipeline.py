"""
This is a boilerplate pipeline 'data_cleaning'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import (
    add_cabin_features,
    fill_missing_values,
    drop_unused_columns,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Build data-preparation pipeline.

    The pipeline performs three sequential steps:

    1. 'add_cabin_features' - parse *Cabin* into *Deck* and *Side*.
    2. 'fill_missing_values' - impute NaNs.
    3. 'drop_unused_columns' - remove columns not needed.

    Namespace "data_cleaning" lets Kedro Viz display these three nodes as one logical group.

    Returns:
        A Kedro Pipeline object.
    """
    return pipeline(
        [
            node(
                add_cabin_features,
                inputs="raw_train",
                outputs="df_with_cabin",
                name="add_cabin_features",
            ),
            node(
                fill_missing_values,
                inputs="df_with_cabin",
                outputs="df_filled",
                name="fill_missing_values",
            ),
            node(
                drop_unused_columns,
                inputs="df_filled",
                outputs="clean_train",
                name="drop_unused_columns",
            ),
        ],
        namespace="data_preparation_pipeline",
        inputs="raw_train",
        outputs=["clean_train"],
    )
