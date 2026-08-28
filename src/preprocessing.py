"""
Reusable preprocessing for the Disease Risk Classifier.
Now uses SMOTENC for mixed data types.
"""

from pathlib import Path
from typing import Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Feature groups
CONTINUOUS = ["BMI", "MentHlth", "PhysHlth", "Age"]
BINARY = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex"
]
ORDINAL = ["GenHlth", "Education", "Income"]

ALL_FEATURES = CONTINUOUS + BINARY + ORDINAL
TARGET = "Diabetes_012"

# Indices of categorical features (needed by SMOTENC)
# Order in ALL_FEATURES: 0-3 continuous, 4-17 binary, 18-20 ordinal
CATEGORICAL_INDICES = list(range(4, 21))   # everything except the 4 continuous


def prepare_data(
    data_path: str | Path = "../data/raw/diabetes_012_health_indicators_BRFSS2015.csv",
    test_size: float = 0.20,
    random_state: int = 42,
    use_binary_target: bool = True,
    apply_smote: bool = False,
    save_dir: Optional[str | Path] = "../data/processed",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:

    df = pd.read_csv(data_path)

    # Binary target: 0 = no diabetes, 1 = prediabetes or diabetes
    if use_binary_target:
        df["target"] = (df[TARGET] > 0).astype(int)
        target_col = "target"
    else:
        target_col = TARGET

    X = df[ALL_FEATURES].copy()
    y = df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Scale only continuous features
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[CONTINUOUS] = scaler.fit_transform(X_train[CONTINUOUS])
    X_test[CONTINUOUS] = scaler.transform(X_test[CONTINUOUS])

    # SMOTENC (handles mixed continuous + categorical correctly)
    if apply_smote:
        smote = SMOTENC(
            categorical_features=CATEGORICAL_INDICES,
            random_state=random_state
        )
        X_res, y_res = smote.fit_resample(X_train, y_train)
        X_train = pd.DataFrame(X_res, columns=ALL_FEATURES)
        y_train = pd.Series(y_res, name=target_col)

    # Save
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        suffix = "_binary" if use_binary_target else "_multiclass"
        smote_suffix = "_smotenc" if apply_smote else ""

        X_train.to_csv(save_dir / f"X_train{suffix}{smote_suffix}.csv", index=False)
        X_test.to_csv(save_dir / f"X_test{suffix}.csv", index=False)
        y_train.to_frame().to_csv(save_dir / f"y_train{suffix}{smote_suffix}.csv", index=False)
        y_test.to_frame().to_csv(save_dir / f"y_test{suffix}.csv", index=False)
        joblib.dump(scaler, save_dir / f"scaler{suffix}.joblib")

        print(f"Saved processed data to {save_dir}")

    print(f"\nTrain shape: {X_train.shape} | Test shape: {X_test.shape}")
    print(f"Target distribution (train):\n{y_train.value_counts(normalize=True).sort_index().round(4)}")
    if apply_smote:
        print("(SMOTENC applied – train set is now balanced)")

    return X_train, X_test, y_train, y_test, scaler