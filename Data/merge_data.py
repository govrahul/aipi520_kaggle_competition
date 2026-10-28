"""
Create aggregated and merged datasets from the competition CSV files.

Run from any directory with:

    python Data/merge_data.py

Use ``--input-dir`` and ``--output-dir`` when the default directories are not
appropriate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def aggregate_subscriptions(subscriptions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate subscription history to one row per account."""
    subscriptions = subscriptions.copy()
    subscriptions["multiple.subs"] = subscriptions["multiple.subs"].map(
        {"no": 0, "yes": 1}
    )
    subscriptions["season_start_year"] = subscriptions["season"].str[:4].astype(int)

    return subscriptions.groupby("account.id", as_index=False).agg(
        season_count=("season", "count"),
        earliest_season_start_year=("season_start_year", "min"),
        latest_season_start_year=("season_start_year", "max"),
        mean_price_level=("price.level", "mean"),
        mean_subscription_tier=("subscription_tier", "mean"),
        mean_multiple_subs=("multiple.subs", "mean"),
        mean_no_seats=("no.seats", "mean"),
    )


def aggregate_tickets(tickets: pd.DataFrame) -> pd.DataFrame:
    """Filter and aggregate ticket history to one row per account."""
    tickets = tickets.copy()
    tickets["multiple.tickets"] = tickets["multiple.tickets"].map(
        {"no": 0, "yes": 1}
    )
    tickets["season_start_year"] = tickets["season"].str[:4].astype(int)
    tickets = tickets[tickets["price.level"].isin(["0", "1", "2", "3", "4", "4.0"])]
    tickets["price.level"] = tickets["price.level"].astype(float)

    return tickets.groupby("account.id", as_index=False).agg(
        season_count=("season", "count"),
        earliest_season_start_year=("season_start_year", "min"),
        latest_season_start_year=("season_start_year", "max"),
        mean_price_level=("price.level", "mean"),
        mean_multiple_tickets=("multiple.tickets", "mean"),
        mean_no_seats=("no.seats", "mean"),
    )


def build_merged_data(input_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Build and save the aggregate and merged datasets."""
    output_dir.mkdir(parents=True, exist_ok=True)
    accounts = pd.read_csv(input_dir / "account.csv", encoding="latin1")
    subscriptions = pd.read_csv(input_dir / "subscriptions.csv")
    tickets = pd.read_csv(input_dir / "tickets_all.csv")

    subscriptions_agg = aggregate_subscriptions(subscriptions)
    tickets_agg = aggregate_tickets(tickets)

    subscriptions_agg.to_csv(output_dir / "subscriptions_agg.csv", index=False)
    tickets_agg.to_csv(output_dir / "tickets_agg.csv", index=False)

    merged = (
        accounts.merge(
            subscriptions_agg,
            how="left",
            on="account.id",
            suffixes=("", "_subs"),
        )
        .merge(
            tickets_agg,
            how="left",
            on="account.id",
            suffixes=("_subs", "_tickets"),
        )
    )
    merged["past_subscription"] = merged["season_count_subs"].notna().astype(int)
    merged.to_csv(output_dir / "merged.csv", index=False)

    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate subscription and ticket history and merge it with accounts."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "for_students",
        help="Directory containing source CSV files (default: Data/for_students).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for generated CSV files (default: Data).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    merged = build_merged_data(input_dir, output_dir)
    print(f"Saved merged data for {len(merged):,} accounts to {output_dir / 'merged.csv'}")
    print(f"Saved subscription aggregates to {output_dir / 'subscriptions_agg.csv'}")
    print(f"Saved ticket aggregates to {output_dir / 'tickets_agg.csv'}")


if __name__ == "__main__":
    main()