from matplotlib import pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns


class Correlations:
    @staticmethod
    def perform_correlation_matrix(dataset):
        print("\n===== Step - Correlation Matrix =====")
        num_cols = dataset.data_frame.select_dtypes(include=['float64', 'int64']).columns
        correlation = dataset.data_frame[num_cols].corr()
        print("Correlation matrix:\n", correlation)
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm",
                    annot_kws={"size": 7})  # smaller numbers
        # sns.heatmap(correlation, annot=True, cmap="coolwarm")
        plt.title("Correlation Matrix")
        plt.show()

    @staticmethod
    def perform_pearson_correlation_matrix(dataset):
        print("\n===== Step - Pearson correlations with significance (Pearson, p < 0.05) =====")
        num_cols = dataset.data_frame.select_dtypes(include=['float64', 'int64']).columns
        significant_num = []

        # Loop through all pairs of numeric columns
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                col1 = num_cols[i]
                col2 = num_cols[j]

                # Drop NaN values for this pair
                valid_data = dataset.data_frame[[col1, col2]].dropna()
                if valid_data.empty:
                    continue

                corr, p = pearsonr(valid_data[col1], valid_data[col2])

                if p < 0.05:
                    significant_num.append((col1, col2, corr, p))
                    print(f"{col1} ↔ {col2}: corr={corr:.3f}, p={p:.4g}")

        print(f"There are {len(significant_num)} significant correlations.")

    @staticmethod
    def compute_cramers_v(dataset):
        print("===== Step - Computing Cramer's V =====")

        from scipy.stats import chi2_contingency
        import pandas as pd
        import numpy as np

        # Funkcja pomocnicza
        def cramers_v(x, y):
            confusion_matrix = pd.crosstab(x, y)
            chi2, _, _, _ = chi2_contingency(confusion_matrix)
            n = confusion_matrix.sum().sum()
            phi2 = chi2 / n
            r, k = confusion_matrix.shape
            return np.sqrt(phi2 / min(k - 1, r - 1))

        # 1. Znajdź kolumny kategoryczne (np. zakodowane LabelEncoderem)
        categorical_columns = []
        for column in dataset.data_frame.columns:
            if dataset.data_frame[column].nunique() < 20:
                categorical_columns.append(column)

        print("Categorical columns:", categorical_columns)

        # 2. Policz Cramer's V dla każdej pary kolumn
        for i in range(len(categorical_columns)):
            for j in range(i + 1, len(categorical_columns)):
                col1 = categorical_columns[i]
                col2 = categorical_columns[j]

                value = cramers_v(dataset.data_frame[col1], dataset.data_frame[col2])
                print(f"{col1} ↔ {col2}: Cramer's V = {value:.3f}")