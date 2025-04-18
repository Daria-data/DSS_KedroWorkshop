"""
This is a boilerplate pipeline 'my_first_pipeline'
generated using Kedro 0.19.12
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import my_first_node


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                my_first_node,
                inputs="train_df",
                outputs="train_df_head",
                name="sample_train_df",
            )
        ]
    )
