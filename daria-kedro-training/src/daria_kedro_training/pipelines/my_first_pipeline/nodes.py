"""
This is a boilerplate pipeline 'my_first_pipeline'
generated using Kedro 0.19.12
"""

import numpy as np
import pandas as pd


def my_first_node(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return the first 5 rows of the input DataFrame.

    input_df : pd.DataFrame
    """

    return input_df.head()
