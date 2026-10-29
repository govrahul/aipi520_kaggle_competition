from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_FOLDER = Path(__file__).resolve().parents[1] / "Data"
STUDENT_DATA_FOLDER = DATA_FOLDER / "for_students"
RANDOM_STATE = 520

train_labels = pd.read_csv(STUDENT_DATA_FOLDER / "train.csv", encoding="latin1")
features = pd.read_csv(DATA_FOLDER / "merged.csv", encoding="latin1")
data = train_labels.merge(
    features,
    on="account.id",
    how="left",
    validate="one_to_one",
)

if data["past_subscription"].isna().any():
    raise ValueError("Some training accounts are missing merged features.")

data["has_subs"] = data["season_count_subs"].notna().astype(int)
data["has_tickets"] = data["season_count_tickets"].notna().astype(int)
data[["season_count_subs", "season_count_tickets"]] = data[
    ["season_count_subs", "season_count_tickets"]
].fillna(0)

# The history flags capture whether these year ranges exist; retain only counts,
# which also express duration and are defined as zero for accounts without history.
features_to_drop = [
    "account.id",
    "earliest_season_start_year_subs",
    "latest_season_start_year_subs",
    "earliest_season_start_year_tickets",
    "latest_season_start_year_tickets",
    "past_subscription",  # Identical to has_subs in the merged data.
]
data = data.drop(columns=features_to_drop)

count_features = ["season_count_subs", "season_count_tickets"]
continuous_features = [
    "amount.donated.2013",
    "amount.donated.lifetime",
    "no.donations.lifetime",
    "mean_price_level_subs",
    "mean_subscription_tier",
    "mean_multiple_subs",
    "mean_no_seats_subs",
    "mean_price_level_tickets",
    "mean_multiple_tickets",
    "mean_no_seats_tickets",
]
indicator_features = ["has_subs", "has_tickets"]
model_features = count_features + continuous_features + indicator_features

preprocessor = ColumnTransformer(
    transformers=[
        (
            "counts",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                    ("scaler", StandardScaler()),
                ]
            ),
            count_features,
        ),
        (
            "continuous",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            continuous_features,
        ),
        ("indicators", "passthrough", indicator_features),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]
)

X_train, X_validation, y_train, y_validation = train_test_split(
    data[model_features],
    train_labels["label"],
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=train_labels["label"],
)

model.fit(X_train, y_train)
validation_probabilities = model.predict_proba(X_validation)[:, 1]
train_probs = model.predict_proba(X_train)[:, 1]
validation_auroc = roc_auc_score(y_validation, validation_probabilities)
train_auroc = roc_auc_score(y_train, train_probs)

print(f"Train AUROC: {train_auroc:.4f}")
print(f"Validation AUROC: {validation_auroc:.4f}")

model.fit(data[model_features], train_labels["label"])

test_ids = pd.read_csv(STUDENT_DATA_FOLDER / "test.csv", encoding="latin1")
test_data = test_ids.rename(columns={"ID": "account.id"}).merge(
    features,
    on="account.id",
    how="left",
    validate="one_to_one",
)

if test_data["past_subscription"].isna().any():
    raise ValueError("Some test accounts are missing merged features.")

test_data["has_subs"] = test_data["season_count_subs"].notna().astype(int)
test_data["has_tickets"] = test_data["season_count_tickets"].notna().astype(int)
test_data[["season_count_subs", "season_count_tickets"]] = test_data[
    ["season_count_subs", "season_count_tickets"]
].fillna(0)

test_probabilities = model.predict_proba(test_data[model_features])[:, 1]
submission = pd.DataFrame(
    {
        "ID": test_ids["ID"],
        "Predicted": test_probabilities,
    }
)
submission.to_csv(DATA_FOLDER / "log_reg_submission.csv", index=False)
print(f"Saved {len(submission)} predictions to log_reg_submission.csv")