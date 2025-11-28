from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class TrainingDataPreparation:
    def __init__(self, features_data, output_data, test_data_split, random_state, scale_data):
        self.features_data = features_data
        self.output_data = output_data
        self.test_data_split = test_data_split
        self.random_state = random_state
        self.scale_data = scale_data

    def process_data(self):
        print("\n===== Step - Preparing training data =====")
        x_train, x_test, y_train, y_test = train_test_split(
            self.features_data, self.output_data, test_size=self.test_data_split, random_state=42)
        print("Number of training samples:", len(x_train))
        print("Number of test samples:", len(x_test))
        if self.scale_data:
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)
        return x_train, x_test, y_train, y_test
