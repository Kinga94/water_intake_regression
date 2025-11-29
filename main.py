from matplotlib import pyplot as plt
from xgboost import XGBRegressor


from settings.configuration import DATASET_PATH, MODEL_OUTPUT_COLUMN_NAME, TEST_DATASET_SPLIT, RANDOM_STATE, \
    COLUMNS_TO_DROP, COLUMNS_TO_ENCODE
from tools.correlations import Correlations
from tools.data_preparation import DatasetPreparation, DataPreProcessing
from tools.model import Model
from tools.training_data_preparation import TrainingDataPreparation
from tools.shap import compute_shap_values, shap_summary_plot, shap_bar_plot, shap_dependence

def get_output_distribution(dataset, output_column_name):
    dataset.get_unique_elements_in_selected_column(output_column_name)
    plt.hist(dataset.data_frame[output_column_name])
    plt.title(f"Distribution of '{output_column_name}' values")
    plt.xlabel(output_column_name)
    plt.ylabel("Number of rows")
    plt.show()


def plot_dataset_histogram(dataset):
    dataset.data_frame.hist(bins=30, figsize=(14, 10))
    plt.tight_layout()
    plt.show()

def perform_descriptive_statistics(dataset):
    print("\n===== Step - Performing descriptive statistics =====")
    num_cols = dataset.data_frame.select_dtypes(include=['float64', 'int64']).columns
    stats = dataset.data_frame[num_cols].agg(['mean', 'median', 'std']).T
    print(stats)


def get_correlations(dataset):
    Correlations.perform_correlation_matrix(dataset)
    Correlations.perform_pearson_correlation_matrix(dataset)
    Correlations.compute_cramers_v(dataset)

def main():
    # Collect data to DataFrame and remove unnecessary data
    dataset = DatasetPreparation(dataset_location=DATASET_PATH,
                                 remove_empty_columns=True,
                                 remove_unnecessary_columns=True)

    # Get information about number of possible output values and dataset class split
    get_output_distribution(dataset, MODEL_OUTPUT_COLUMN_NAME)
    # Wnioski: 
    # Rozkład jest zbliżony do normalnego, bez wyraźnych odchyleń
    # Histogram sugeruje pewną dwumodalność – jeden szczyt w okolicach 2,6–2,7 L, a drugi mniejszy przy ~3,6 L.
    # Może to wskazywać na istnienie dwóch grup w populacji, np. osób o różnym poziomie aktywności fizycznej lub różnych nawykach związanych z nawodnieniem.
    # Warto rozwazyć standaryzację danych przed trenowaniem modelu, aby poprawić jego wydajność.

    plot_dataset_histogram(dataset)

    perform_descriptive_statistics(dataset)
    get_correlations(dataset)
    # Wnioski: 
    # Niektore cechy wykazują silne korelacje ze zmienną Water intake (Fat Percantage - ujemna korelacja, Waga, Wzrost korelacje dodatnie), co sugeruje, że mogą być one istotne dla modelu predykcyjnego.

    preprocessed_data_frame = DataPreProcessing(data_frame=dataset.data_frame,
                                                output_column_name=MODEL_OUTPUT_COLUMN_NAME,
                                                columns_to_drop=COLUMNS_TO_DROP,
                                                columns_to_encode=COLUMNS_TO_ENCODE)
    features_data, output_data = preprocessed_data_frame.process_data()

    training_data_preparation = TrainingDataPreparation(features_data=features_data,
                                                        output_data=output_data,
                                                        test_data_split=TEST_DATASET_SPLIT,
                                                        random_state=RANDOM_STATE,
                                                        scale_data=False)
    train_test_dataset = training_data_preparation.process_data()
    x_train, x_test, y_train, y_test = train_test_dataset

    model = XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
    )
    regression_model = Model(model,
                             dataset=train_test_dataset,
                             model_name="XGB Regression")
    regression_model.fit()
    regression_model.score()
    regression_model.metrics()

    best_params = Model.optimize_xgb_regression_model(train_test_dataset)
    optimized_model = XGBRegressor(**best_params)
    regression_mode_optimized = Model(optimized_model,
                                      dataset=train_test_dataset,
                                      model_name="XGB Regression optimized")
    regression_mode_optimized.fit()
    regression_mode_optimized.metrics()

# Wnioski:
# Optymalizacja hiperparametrów za pomocą Optuna przyniosła  poprawę w wydajności modelu XGBRegressor, co podkreśla znaczenie dostosowywania parametrów modelu do specyfiki danych.

# Warto rozważyć dalsze eksperymenty z innymi technikami optymalizacji oraz różnymi modelami, aby jeszcze bardziej poprawić dokładność predykcji.
# Warto równiez rozważyć implementację walidacji krzyżowej podczas optymalizacji hiperparametrów, aby zapewnić bardziej stabilne i wiarygodne wyniki.
# Warto również przyjżeć się cechom o najwyższej ważności i rozważyć ich dalszą analizę lub inżynierię cech w celu poprawy wydajności modelu. - przy uzyciu Shapp
# Mozna rowniez popracowac nad wartosciami odsstajacymi 

 
    shap_values = compute_shap_values(optimized_model, features_data)
    shap_summary_plot(shap_values, features_data)
    shap_bar_plot(shap_values, features_data)
    shap_dependence("Gender",shap_values, features_data)
    shap_dependence("Session_Duration (hours)",shap_values, features_data)


print("END")

if __name__ == "__main__":
    main()
