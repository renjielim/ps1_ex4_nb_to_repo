import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

from cProfile import label
import matplotlib.pyplot as plt
import seaborn as sns


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise missing values and data types for each column in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.

    Returns
    -------
    pd.DataFrame
        A transposed summary table showing:
        - Total missing values per column
        - Percentage of missing values per column
        - Data type of each column
    """
    total = df.isnull().sum()
    percent = df.isnull().sum() / df.isnull().count() * 100

    tt = pd.concat([total, percent], axis=1, keys=["Total", "Percent"])
    tt["Types"] = [str(df[col].dtype) for col in df.columns]

    return tt.T  # Transpose for readability


def family(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a new column 'Family Size' in the DataFrame by summing 'SibSp' and 'Parch' and adding 1.
    """
    df = df.copy()
    df["Family Size"] = df["SibSp"] + df["Parch"] + 1

    return df


def frequency_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise the most frequent item, its count, and percent of total for each column.
    """
    total = df.count()
    tt = pd.DataFrame(total, columns=["Total"])
    items, vals = [], []

    for col in df.columns:
        try:
            counts = df[col].value_counts(dropna=False)
            items.append(counts.index[0])
            vals.append(counts.iloc[0])
        except Exception:
            items.append(None)
            vals.append(0)

    tt["Most frequent item"] = items
    tt["Frequency"] = vals
    tt["Percent from total"] = np.round(tt["Frequency"] / tt["Total"] * 100, 3)
    return tt.T


def unique_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise non-missing and unique value counts for each column.
    """
    total = df.count()
    tt = pd.DataFrame(total, columns=["Total"])
    tt["Uniques"] = [df[col].nunique() for col in df.columns]
    return tt.T


def combine_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine train and test DataFrames and label their source in a new 'set' column.
    """
    all_df = pd.concat([train_df, test_df], axis=0)
    all_df["set"] = "train"
    all_df.loc[all_df["Survived"].isna(), "set"] = "test"
    return all_df


def family_add(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a new column 'Family Size' in the DataFrame by summing 'SibSp' and 'Parch' and adding 1.
    """
    df = df.copy()
    df["Family Size"] = df["SibSp"] + df["Parch"] + 1

    return df


def add_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'Age Interval' and 'Fare Interval' columns based on defined numeric bins.
    """
    df = df.copy()

    # Age bins
    age_bins = [-float("inf"), 16, 32, 48, 64, float("inf")]
    df["Age Interval"] = pd.cut(df["Age"], bins=age_bins, labels=[0, 1, 2, 3, 4])

    # Fare bins
    fare_bins = [-float("inf"), 7.91, 14.454, 31, float("inf")]
    df["Fare Interval"] = pd.cut(df["Fare"], bins=fare_bins, labels=[0, 1, 2, 3])

    return df


def add_sex_pclass(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'Sex_Pclass' column combining the first letter of 'Sex' and 'Pclass'.

    Example: Male, class 2 → 'M_C2'
    """
    df = df.copy()
    df["Sex_Pclass"] = df.apply(
        lambda row: row["Sex"][0].upper() + "_C" + str(row["Pclass"]), axis=1
    )
    return df


def parse_names(row):
    try:
        text = row["Name"]
        split_text = text.split(",")
        family_name = split_text[0]
        next_text = split_text[1]
        split_text = next_text.split(".")
        title = (split_text[0] + ".").lstrip().rstrip()
        next_text = split_text[1]
        if "(" in next_text:
            split_text = next_text.split("(")
            given_name = split_text[0]
            maiden_name = split_text[1].rstrip(")")
            return pd.Series([family_name, title, given_name, maiden_name])
        else:
            given_name = next_text
            return pd.Series([family_name, title, given_name, None])
    except Exception as ex:
        print(f"Exception: {ex}")


def add_name_parts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply parse_names() to each row to extract name components
    and add columns: Family Name, Title, Given Name, Maiden Name.
    """
    df = df.copy()
    df[["Family Name", "Title", "Given Name", "Maiden Name"]] = df.apply(
        lambda row: parse_names(row), axis=1
    )
    return df


def add_family_and_title_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Add 'Family Type' from 'Family Size'
    - Classify Family Type as Single / Small / Large
    - Add and standardize 'Titles' from 'Title'
    Returns the modified list of DataFrames.
    """
    df = df.copy()

    # Family Type
    df["Family Type"] = df["Family Size"]
    df.loc[df["Family Size"] == 1, "Family Type"] = "Single"
    df.loc[(df["Family Size"] > 1) & (df["Family Size"] < 5), "Family Type"] = "Small"
    df.loc[df["Family Size"] >= 5, "Family Type"] = "Large"

    # Titles
    df["Titles"] = df["Title"]
    df["Titles"] = df["Titles"].replace(
        {"Mlle.": "Miss.", "Ms.": "Miss.", "Mme.": "Mrs."}
    )
    df["Titles"] = df["Titles"].replace(
        [
            "Lady.",
            "the Countess.",
            "Capt.",
            "Col.",
            "Don.",
            "Dr.",
            "Major.",
            "Rev.",
            "Sir.",
            "Jonkheer.",
            "Dona.",
        ],
        "Rare",
    )

    return df


def encode_sex(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode 'Sex' as numeric (female=1, male=0) for each DataFrame in dfs.
    """
    df = df.copy()
    df["Sex"] = df["Sex"].map({"female": 1, "male": 0}).astype(int)
    return df


def get_features(df: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    """
    Return only the predictor columns from a DataFrame.
    """
    return df[predictors].copy()


def get_target(df: pd.DataFrame, target: str) -> np.ndarray:
    """
    Return the target column as a NumPy array.
    """
    return df[target].values


def train_and_predict_rf(train_X, train_Y, valid_X=None):
    """
    Train a Random Forest classifier and make predictions.
    """
    clf = RandomForestClassifier(n_jobs=-1, random_state=42, n_estimators=100)
    clf.fit(train_X, train_Y)

    results = {
        "preds_tr": clf.predict(train_X),
        "preds_val": clf.predict(valid_X) if valid_X is not None else None,
        "model": clf,
    }
    return results


COLOR_LIST = ["#A5D7E8", "#576CBC", "#19376D", "#0b2447"]


def plot_count_pairs(data_df, feature, title, hue="set"):
    f, ax = plt.subplots(1, 1, figsize=(8, 4))
    sns.countplot(x=feature, data=data_df, hue=hue, palette=COLOR_LIST)
    plt.grid(color="black", linestyle="-.", linewidth=0.5, axis="y", which="major")
    ax.set_title(f"Number of passengers / {title}")
    f.savefig(
        f"figures/Number of passengers, {title}, by {hue}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def plot_distribution_pairs(data_df, feature, title, hue="set"):
    f, ax = plt.subplots(1, 1, figsize=(8, 4))
    for i, h in enumerate(data_df[hue].unique()):
        g = sns.histplot(
            data_df.loc[data_df[hue] == h, feature], color=COLOR_LIST[i], ax=ax, label=h
        )
    ax.set_title(f"Number of passengers / {title}")
    g.legend()
    f.savefig(
        f"figures/Number of passengers, {title}, by {hue}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
