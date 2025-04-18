"""
This is a boilerplate pipeline 'data_preparation'
generated using Kedro 0.19.12
"""
import pandas as pd

def extract_cabin_parts(df: pd.DataFrame) -> pd.DataFrame:
    """Parse `Cabin` into `Deck` and `Side`.

    Args:
        df: Raw dataframe with a `Cabin` column.

    Returns:
        DataFrame with two new columns:
        - Deck: first segment before the first slash.
        - Side: last segment after the last slash.
    """
    df = df.copy()

    # Deck is the first element before the first slash
    df["Deck"] = df["Cabin"].apply(
        lambda x: x.split("/")[0] if not pd.isna(x) else None
    )

    # Side is the last element after the last slash
    df["Side"] = df["Cabin"].apply(
        lambda x: x.split("/")[-1] if not pd.isna(x) else None
    )

    return df

def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values with median for numeric and mode for categorical.

    Args:
        df: DataFrame produced by `extract_cabin_parts`.

    Returns:
        DataFrame with no missing values in
        `Age`, `HomePlanet`, `Destination`, `Deck`, `Side`.
    """
    df = df.copy()

    # Numerical column
    df["Age"].fillna(df["Age"].median(), inplace=True)

    # Categorical columns
    for col in ["HomePlanet", "Destination", "Deck", "Side"]:
        df[col].fillna(df[col].mode()[0], inplace=True)

    return df

def drop_unused_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that are not needed downstream.

    Args:
        df: Cleaned passengers dataframe.

    Returns:
        DataFrame without `PassengerId`, `Cabin`, `Name`.
    """
    return df.drop(columns=["PassengerId", "Cabin", "Name"])
