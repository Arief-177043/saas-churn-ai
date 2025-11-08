import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

model = joblib.load("models/lightgbm_churn_model.pkl")
explainer = shap.TreeExplainer(model)

def explain(sample_df):
    shap_values = explainer.shap_values(sample_df)
    shap.summary_plot(shap_values, sample_df, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig("models/shap_summary.png")
    return shap_values
