import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor


class Model:
    def __init__(self, model_object, dataset, model_name):
        self.model_object = model_object
        self.x_train, self.x_test, self.y_train, self.y_test = dataset
        self.model_name = model_name

    def fit(self):
        print(f"\n===== Step - Training {self.model_name} model =====")
        self.model_object.fit(self.x_train, self.y_train)
        print(f"Model trained successfully.")

    def score(self):
        print(f"\n===== Step - Testing {self.model_name} model =====")
        result = self.model_object.score(self.x_test, self.y_test)
        print(f"Score: {result * 100.0}%")
        return result

    def metrics(self):
        predictions = self.model_object.predict(self.x_test)
        rmse = np.sqrt(mean_squared_error(self.y_test, predictions))
        print("RMSE value:", rmse)
        mean_abs_value = mean_absolute_error(self.y_test, predictions)
        print("MAE value:", mean_abs_value)

    @staticmethod
    def optimize_xgb_regression_model(train_dataset):
        def objective(trial):
            x_train, x_valid, y_train, y_valid = train_dataset
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 10)
            }
            model = XGBRegressor(**params)
            model.fit(x_train, y_train)
            preds = model.predict(x_valid)
            rmse = np.sqrt(mean_squared_error(y_valid, preds))
            return rmse
        study = optuna.create_study(direction="minimize", sampler=TPESampler())
        study.optimize(objective, n_trials=50)
        best_params = study.best_trial.params
        best_trial_rmse = study.best_trial.value
        print("Best trial parameters:", best_params)
        print("Best trial RMSE:", best_trial_rmse)
        return best_params