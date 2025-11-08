import pandas as pd

def aggregate_features(df):
    # Convert date columns safely
    for col in ["date", "subscription_start_date", "subscription_end_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Handle subscription duration
    if {"subscription_start_date", "subscription_end_date"}.issubset(df.columns):
        df["subscription_duration_days"] = (
            (df["subscription_end_date"] - df["subscription_start_date"])
            .dt.days.fillna(0)
        )
    else:
        df["subscription_duration_days"] = 0

    # Detect numeric columns automatically
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # Add revenue column manually if exists
    if "monthly_revenue" in df.columns and "monthly_revenue" not in numeric_cols:
        df["monthly_revenue"] = pd.to_numeric(df["monthly_revenue"], errors="coerce")
        numeric_cols.append("monthly_revenue")

    # Always ensure we group by customer_id
    if "customer_id" not in df.columns:
        raise KeyError("Dataset must contain 'customer_id' column!")

    # Build aggregation dictionary dynamically
    agg_dict = {col: "mean" for col in numeric_cols if col != "customer_id"}

    # If still no numeric columns, fallback to duration
    if not agg_dict:
        agg_dict = {"subscription_duration_days": "mean"}

    # Perform aggregation
    agg_features = df.groupby("customer_id").agg(agg_dict).reset_index()

    # Add subscription duration explicitly if not already in aggregation
    if "subscription_duration_days" not in agg_features.columns:
        agg_features["subscription_duration_days"] = df["subscription_duration_days"].mean()

    return agg_features
