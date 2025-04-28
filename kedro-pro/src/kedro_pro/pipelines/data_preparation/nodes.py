import pandas as pd


def add_cabin_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create 'Deck' and 'Side' columns parsed from the 'Cabin' field.

    Args:
        df: Raw passengers dataframe that contains a 'Cabin' column
            formatted like "B/123/P".

    Returns:
        A copy of the dataframe with two new columns:
            * Deck — the first segment before the first slash.
            * Side — the last segment after the last slash.
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


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing data using median (numeric) and mode (categorical).

    Args:
        df: DataFrame returned by `add_cabin_features`.

    Returns:
        DataFrame with no missing values for the selected columns.
    """
    df = df.copy()

    # Numerical: median
    df["Age"].fillna(df["Age"].median())

    # Categorical: mode
    df["HomePlanet"].fillna(df["HomePlanet"].mode()[0])
    df["Destination"].fillna(df["Destination"].mode()[0])
    df["Deck"].fillna(df["Deck"].mode()[0])
    df["Side"].fillna(df["Side"].mode()[0])

    # Fill na constant values for the remaining columns we are going to use
    df["VIP"] = df["VIP"].fillna(False).astype("boolean")
    df["CryoSleep"] = df["CryoSleep"].fillna(False).astype("boolean")
    df['VRDeck'].fillna(0)
    df['RoomService'].fillna(0)
    df['FoodCourt'].fillna(0)
    df['ShoppingMall'].fillna(0)
    df['Spa'].fillna(0)
    df['VRDeck'].fillna(0)

    return df


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that are not needed for downstream processing.

    Args:
        df: Cleaned dataframe.

    Returns:
        DataFrame without `PassengerId`, `Cabin`, `Name`.
    """
    return df.drop(columns=["PassengerId", "Cabin", "Name"])
