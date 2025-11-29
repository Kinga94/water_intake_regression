import shap
import matplotlib.pyplot as plt

def compute_shap_values(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return shap_values

def shap_summary_plot(shap_values, X):
    shap.summary_plot(shap_values, X)

def shap_bar_plot(shap_values, X):
    shap.summary_plot(shap_values, X, plot_type="bar")
    plt.show()

def shap_dependence(feature_name, shap_values, X):
    shap.dependence_plot(feature_name, shap_values, X)
    plt.show()