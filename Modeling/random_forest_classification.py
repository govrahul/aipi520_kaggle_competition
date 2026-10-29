from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


DATA_FOLDER = Path(__file__).resolve().parents[1] / "Data"
STUDENT_DATA_FOLDER = DATA_FOLDER / "for_students"
RANDOM_STATE = 520

train_labels = pd.read_csv(STUDENT_DATA_FOLDER / "train.csv", encoding="latin1")
features = pd.read_csv(DATA_FOLDER / "merged.csv", encoding="latin1")
subscriptions = pd.read_csv(
    STUDENT_DATA_FOLDER / "subscriptions.csv",
    encoding="latin1",
)
subscriptions["season_start_year"] = pd.to_numeric(
    subscriptions["season"].str[:4],
    errors="coerce",
)
if subscriptions["season_start_year"].isna().any():
    raise ValueError("Some subscription seasons have an invalid start year.")
if subscriptions.duplicated(["account.id", "season"]).any():
    raise ValueError("Subscription history has multiple records per account-season.")

# Labels target the season immediately after the final observed subscription season.
subscription_reference_year = int(subscriptions["season_start_year"].max()) + 1
subscription_history = subscriptions.groupby("account.id").agg(
    subscription_frequency=("season", "nunique"),
    first_subscription_year=("season_start_year", "min"),
    last_subscription_year=("season_start_year", "max"),
)
subscription_history["subscription_recency_years"] = (
    subscription_reference_year - subscription_history["last_subscription_year"]
)
subscription_history["subscription_tenure_years"] = (
    subscription_history["last_subscription_year"]
    - subscription_history["first_subscription_year"]
)
for year_window in (3, 5):
    recent_subscriptions = subscriptions[
        subscriptions["season_start_year"]
        >= subscription_reference_year - year_window
    ]
    subscription_history[f"subscription_seasons_last_{year_window}_years"] = (
        recent_subscriptions.groupby("account.id")["season"].nunique()
    )

subscriptions["package"] = subscriptions["package"].fillna("Unknown")
package_shares = pd.crosstab(
    subscriptions["account.id"],
    subscriptions["package"],
    normalize="index",
)
package_shares.columns = [
    f"package_share_{package.lower().replace(' ', '_')}"
    for package in package_shares.columns
]
subscription_history = subscription_history.join(package_shares)
subscription_history = subscription_history.drop(
    columns=["first_subscription_year", "last_subscription_year"]
)
recency_frequency_features = [
    "subscription_frequency",
    "subscription_recency_years",
    "subscription_tenure_years",
    "subscription_seasons_last_3_years",
    "subscription_seasons_last_5_years",
]
package_preference_features = package_shares.columns.tolist()
subscription_history = subscription_history.reset_index()

features = features.merge(
    subscription_history,
    on="account.id",
    how="left",
    validate="one_to_one",
)
features[
    [
        "subscription_frequency",
        "subscription_tenure_years",
        "subscription_seasons_last_3_years",
        "subscription_seasons_last_5_years",
    ]
] = features[
    [
        "subscription_frequency",
        "subscription_tenure_years",
        "subscription_seasons_last_3_years",
        "subscription_seasons_last_5_years",
    ]
].fillna(0)
features[package_preference_features] = features[
    package_preference_features
].fillna(0)

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

features_to_drop = [
    "account.id",
    "earliest_season_start_year_subs",
    "latest_season_start_year_subs",
    "earliest_season_start_year_tickets",
    "latest_season_start_year_tickets",
    "past_subscription",
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
baseline_features = count_features + continuous_features + indicator_features
model_features = baseline_features + recency_frequency_features


def create_model(include_recency_frequency, include_package_preferences=False):
    transformers = [
        (
            "counts",
            SimpleImputer(strategy="constant", fill_value=0),
            count_features,
        ),
        (
            "continuous",
            SimpleImputer(strategy="median"),
            continuous_features,
        ),
        ("indicators", "passthrough", indicator_features),
    ]
    if include_recency_frequency:
        transformers.append(
            (
                "subscription_history",
                SimpleImputer(strategy="median"),
                recency_frequency_features,
            )
        )
    if include_package_preferences:
        transformers.append(
            (
                "package_preferences",
                SimpleImputer(strategy="constant", fill_value=0),
                package_preference_features,
            )
        )

    return Pipeline(
        steps=[
            ("preprocessor", ColumnTransformer(transformers=transformers)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=6,
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

X_train, X_validation, y_train, y_validation = train_test_split(
    data[
        baseline_features
        + recency_frequency_features
        + package_preference_features
    ],
    train_labels["label"],
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=train_labels["label"],
)

baseline_model = create_model(include_recency_frequency=False)
baseline_model.fit(X_train[baseline_features], y_train)
baseline_validation_probabilities = baseline_model.predict_proba(
    X_validation[baseline_features]
)[:, 1]
baseline_validation_auroc = roc_auc_score(
    y_validation, baseline_validation_probabilities
)
print(f"Baseline validation AUROC: {baseline_validation_auroc:.4f}")

model = create_model(include_recency_frequency=True)
model.fit(X_train[model_features], y_train)
validation_probabilities = model.predict_proba(X_validation[model_features])[:, 1]
train_probabilities = model.predict_proba(X_train[model_features])[:, 1]
validation_auroc = roc_auc_score(y_validation, validation_probabilities)
train_auroc = roc_auc_score(y_train, train_probabilities)

print(f"Train AUROC: {train_auroc:.4f}")
print(f"Validation AUROC: {validation_auroc:.4f}")
print(
    "Validation AUROC change from recency/frequency features: "
    f"{validation_auroc - baseline_validation_auroc:+.4f}"
)

package_model = create_model(
    include_recency_frequency=True,
    include_package_preferences=True,
)
package_features = model_features + package_preference_features
package_model.fit(X_train[package_features], y_train)
package_validation_probabilities = package_model.predict_proba(
    X_validation[package_features]
)[:, 1]
package_validation_auroc = roc_auc_score(
    y_validation, package_validation_probabilities
)
print(
    "Validation AUROC with package preferences: "
    f"{package_validation_auroc:.4f} "
    f"({package_validation_auroc - validation_auroc:+.4f} vs recency/frequency)"
)

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

test_data[
    [
        "subscription_frequency",
        "subscription_tenure_years",
        "subscription_seasons_last_3_years",
        "subscription_seasons_last_5_years",
    ]
] = test_data[
    [
        "subscription_frequency",
        "subscription_tenure_years",
        "subscription_seasons_last_3_years",
        "subscription_seasons_last_5_years",
    ]
].fillna(0)
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
submission_path = DATA_FOLDER / "random_forest_submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Saved {len(submission)} predictions to {submission_path.name}")
