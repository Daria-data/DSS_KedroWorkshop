from kedro.pipeline import Pipeline, node, pipeline #noqa

from .nodes import (
    extract_cabin_parts,
    fill_missing,
    drop_unused_cols,
)

def create_pipeline(**kwargs) -> Pipeline:
    """Build the data-preparation pipeline.

    Returns:
        Pipeline: Ordered set of nodes that transforms
        raw_train -> clean_train.
    """
    return pipeline(
        [
            # Parse Deck and Side from Cabin
            node(
                func=extract_cabin_parts,
                inputs="raw_train",
                outputs="df_with_cabin",
                name="extract_cabin_parts_node",
            ),
            # Fill missing values
            node(
                func=fill_missing,
                inputs="df_with_cabin",
                outputs="df_filled",
                name="fill_missing_node",
            ),
            # Drop unused columns
            node(
                func=drop_unused_cols,
                inputs="df_filled",
                outputs="clean_train",
                name="drop_unused_cols_node",
            ),
        ]
    )
