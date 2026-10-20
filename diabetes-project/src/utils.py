from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def _sklearn_missing_array(values):
    """Convert nullable values to np.nan in a new array, preserving the source."""
    array = np.asarray(values, dtype=object)
    return np.where(pd.isna(array), np.nan, array)


def build_preprocessor(scale_numeric=False):
    """Create a fresh, unfitted transformer for the eight common predictors."""
    numeric_steps = [
        ("missing", FunctionTransformer(
            _sklearn_missing_array, feature_names_out="one-to-one"
        )),
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical = Pipeline([
        ("missing", FunctionTransformer(
            _sklearn_missing_array, feature_names_out="one-to-one"
        )),
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore", drop=None, sparse_output=False
        )),
    ])
    return ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), ["age", "bmi"]),
        ("categorical", categorical, [
            "sex", "race_ethnicity", "smoking_status",
            "physical_activity", "heart_attack", "stroke",
        ]),
    ], remainder="drop")


def load_xpt(path, columns=None):
    df = pd.read_sas(Path(path), format="xport")
    df = df.replace(2.0 ** -260, 0.0)
    return df if columns is None else df[columns].copy() 


def _map_input_codes(series, mapping, unknown=()):
    """Map documented codes and reject unexpected values before conversion."""
    unexpected = series.notna() & ~series.isin([*mapping, *unknown])
    if unexpected.any():
        raise ValueError(f"Unexpected codes in {series.name!r}: "
                         f"{series.loc[unexpected].unique().tolist()}")
    return series.map(mapping)


def map_nhanes_inputs(df):
    """Copy the eight existing predictors into the common input schema."""
    result = df[[
        "RIDAGEYR", "RIAGENDR", "RIDRETH3", "BMXBMI", "smoking_status",
        "physical_activity", "heart_attack", "stroke",
    ]].rename(columns={
        "RIDAGEYR": "age", "RIAGENDR": "sex",
        "RIDRETH3": "race_ethnicity", "BMXBMI": "bmi",
    }).copy()
    result["sex"] = _map_input_codes(df["RIAGENDR"], {1: 1, 2: 2}).astype("Int64")
    # RIDRETH3 code 5 is unused; do not confuse this with RIDRETH1.
    result["race_ethnicity"] = _map_input_codes(df["RIDRETH3"], {
        1: "Hispanic", 2: "Hispanic", 3: "White NH", 4: "Black NH",
        6: "Other", 7: "Other",
    }).astype("string")
    if not result["age"].dropna().between(18, 80).all():
        raise ValueError("NHANES adult age must already be topcoded at 80")
    for column, values in {
        "smoking_status": ["current", "former", "never"],
        "physical_activity": [0, 1], "heart_attack": [0, 1], "stroke": [0, 1],
    }.items():
        if not result[column].dropna().isin(values).all():
            raise ValueError(f"Unexpected values in {column}")
    return result


def map_brfss_inputs(df):
    """Map BRFSS 2015/2017 inputs without filtering rows or modifying df."""
    age = df["_AGE80"].copy()
    if not age.dropna().between(18, 80).all():
        raise ValueError("BRFSS _AGE80 must be in 18-80")
    bmi = df["_BMI5"]
    if not bmi.dropna().between(1, 9999).all():
        raise ValueError("Unexpected _BMI5 value")
    return pd.DataFrame({
        "age": age,
        "sex": _map_input_codes(df["SEX"], {1: 1, 2: 2}, [9]).astype("Int64"),
        "race_ethnicity": _map_input_codes(df["_RACEGR3"], {
            1: "White NH", 2: "Black NH", 3: "Other", 4: "Other", 5: "Hispanic",
        }, [9]).astype("string"),
        "bmi": bmi / 100,
        "smoking_status": _map_input_codes(df["_SMOKER3"], {
            1: "current", 2: "current", 3: "former", 4: "never",
        }, [9]).astype("string"),
        "physical_activity": _map_input_codes(df["_PA150R2"], {
            1: 1, 2: 0, 3: 0,
        }, [9]).astype("Int64"),
        "heart_attack": get_yes_no(df["CVDINFR4"]),
        "stroke": get_yes_no(df["CVDSTRK3"]),
    }, index=df.index)


def get_brfss_diabetes_label(series):
    """Return diagnosed non-gestational diabetes labels, not disease absence.

    Pregnancy-only diabetes (2) and prediabetes/borderline (4) are not
    positive cases. Unknown/refused/missing responses stay pd.NA.
    """
    return _map_input_codes(
        series, {1: 1, 2: 0, 3: 0, 4: 0}, unknown=[7, 9]
    ).astype("Int64").rename("diabetes_label")


def get_diabetes_label(df):
    """Return observed-rule labels; the negative rule is a project assumption.

    Any observed positive criterion wins, even with other components missing.
    Label 0 requires DIQ010 in {2, 3} and observed HbA1c <6.5; glucose
    is not required. All other cases remain unknown. Inputs are unchanged.
    """
    positive = (
        df["DIQ010"].eq(1)
        | df["DIQ050"].eq(1)
        | df["DIQ070"].eq(1)
        | df["LBXGH"].ge(6.5)
        | (df["glucose_usable"].eq(True) & df["LBXGLU"].ge(126))
    ).fillna(False)
    negative = (
        df["DIQ010"].isin([2, 3])
        & df["LBXGH"].notna()
        & df["LBXGH"].lt(6.5)
        & ~positive
    ).fillna(False)
    label = pd.Series(pd.NA, index=df.index, dtype="Int64", name="diabetes_label")
    label.loc[negative] = 0
    label.loc[positive] = 1
    return label


def get_yes_no(series):
    """Map NHANES Yes/No codes to nullable integers; reject unexpected codes."""
    unexpected = series.notna() & ~series.isin([1, 2, 7, 9])
    if unexpected.any():
        raise ValueError(
            f"Unexpected codes in {series.name!r}: "
            f"{series.loc[unexpected].unique().tolist()}"
        )
    result = pd.Series(pd.NA, index=series.index, dtype="Int64", name=series.name)
    result.loc[series.eq(1)] = 1
    result.loc[series.eq(2)] = 0
    return result


def get_smoking_status(df):
    """Return smoking status, preserving unknown responses as pd.NA."""
    status = pd.Series(pd.NA, index=df.index, dtype="string", name="smoking_status")
    smoked = df["SMQ020"].eq(1)
    status.loc[df["SMQ020"].eq(2)] = "never"
    status.loc[smoked & df["SMQ040"].eq(3)] = "former"
    status.loc[smoked & df["SMQ040"].isin([1, 2])] = "current"
    return status


def get_physical_activity(df):
    """Return recreational minutes and nullable activity status without modifying df.

    Valid Yes responses require 1-7 days and 10 to <1440 minutes/day
    (CDC questionnaire hard limits, not the observed sample ranges).
    weekly_minutes stays unknown if either branch is unknown; a known
    contribution of at least 150 equivalent minutes still proves status 1.
    """
    def branch_minutes(question, days, minutes):
        result = pd.Series(pd.NA, index=df.index, dtype="Float64")
        result.loc[df[question].eq(2)] = 0.0
        valid = (
            df[question].eq(1)
            & df[days].isin(range(1, 8))
            & df[minutes].ge(10)
            & df[minutes].lt(1440)
        ).fillna(False)
        result.loc[valid] = df.loc[valid, days] * df.loc[valid, minutes]
        return result

    moderate = branch_minutes("PAQ665", "PAQ670", "PAD675")
    vigorous = branch_minutes("PAQ650", "PAQ655", "PAD660")
    weekly = moderate + 2 * vigorous
    # A lower bound for classification only; missing minute outputs stay NA.
    known_minutes = moderate.fillna(0) + 2 * vigorous.fillna(0)
    status = pd.Series(pd.NA, index=df.index, dtype="Int64")
    status.loc[weekly.lt(150).fillna(False)] = 0
    status.loc[known_minutes.ge(150)] = 1
    return pd.DataFrame({
        "moderate_minutes": moderate,
        "vigorous_minutes": vigorous,
        "weekly_minutes": weekly,
        "physical_activity": status,
    })


def get_glucose_usable(df):
    """Return a nullable boolean flag without changing the input.

    Missing required information takes precedence over a failed condition.
    Beverage times are required only for a Yes (1); No (2) needs no time.
    Other beverage response codes remain unknown.
    """
    def no_recent_beverage(question, hours, minutes):
        answer = df[question]
        elapsed = df[hours] + df[minutes] / 60
        known_time = df[[hours, minutes]].notna().all(axis=1)
        result = pd.Series(pd.NA, index=df.index, dtype="boolean")
        result.loc[answer.eq(2)] = True
        answered_yes = answer.eq(1) & known_time
        result.loc[answered_yes] = elapsed.loc[answered_yes].ge(8)
        return result

    coffee_ok = no_recent_beverage("PHQ020", "PHACOFHR", "PHACOFMN")
    alcohol_ok = no_recent_beverage("PHQ030", "PHAALCHR", "PHAALCMN")
    required = ["LBXGLU", "PHDSESN", "PHAFSTHR", "PHAFSTMN", "WTSAF2YR"]
    unknown = df[required].isna().any(axis=1) | coffee_ok.isna() | alcohol_ok.isna()
    food_fast_hours = df["PHAFSTHR"] + df["PHAFSTMN"] / 60
    usable = (
        df["LBXGLU"].notna()
        & df["PHDSESN"].eq(0)
        & food_fast_hours.between(8, 24, inclusive="left")
        & df["WTSAF2YR"].gt(0)
        & coffee_ok
        & alcohol_ok
    )
    return usable.astype("boolean").mask(unknown, pd.NA).rename("glucose_usable")
