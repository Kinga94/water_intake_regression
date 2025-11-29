import pandas
from sklearn.preprocessing import LabelEncoder


class DatasetPreparation:
    def __init__(self,
                 dataset_location: str,
                 remove_empty_columns: bool = True,
                 remove_unnecessary_columns: bool = True,):
        self.dataset_location = dataset_location
        self.data_frame = self.read_data()
        self.print_basic_information()
        self.remove_unnecessary_data(remove_empty_columns, remove_unnecessary_columns)


    def read_data(self):
        return pandas.read_csv(self.dataset_location)

    def print_basic_information(self, number_of_rows_to_display: int = 20):
        print("\n===== ORIGINAL DATASET INFORMATION =====")
        print(f"Dataset location: {self.dataset_location}\n")
        print("• Column names:")
        print(list(self.data_frame.columns), "\n")
        print("• Dataset shape (rows, columns):")
        print(self.data_frame.shape, "\n")
        print("• Data types:")
        print(self.data_frame.dtypes, "\n")
        print("• Dataset statistics:")
        print(self.data_frame.describe(), "\n")
        print("===== =====\n")

    def remove_empty_columns(self):
        print("\n===== Step - Removing empty columns =====")
        empty_columns = []
        for column in self.columns:
            if self.data_frame[column].isnull().all():
                empty_columns.append(column)
        if empty_columns:
            print("Removing empty columns:", empty_columns)
            self.data_frame = self.data_frame.drop(columns=empty_columns)
        else:
            print("There are no empty columns in the dataset. Skipping ...")

    def remove_duplicates(self):
        print("\n===== Step - Removing duplicated rows =====")
        duplicate_rows = []
        for i in range(len(self.data_frame)):
            if self.data_frame.duplicated().iloc[i]:
                duplicate_rows.append(i)
        if duplicate_rows:
            print("Removing duplicated frames. Duplicated frames list:", duplicate_rows)
            self.data_frame = self.data_frame.drop_duplicates().reset_index(drop=True)
        else:
            print("There are no duplicate rows in the dataset. Skipping ...")

    def remove_unnecessary_data(self, remove_empty_columns = True, remove_duplicated_frames = True):
        if remove_empty_columns:
            self.remove_empty_columns()
        if remove_duplicated_frames:
            self.remove_duplicates()

    @property
    def columns(self):
        return self.data_frame.columns

    def get_unique_elements_in_selected_column(self, column_name):
        print(f"\n===== Step - Counting unique values in '{column_name}' column =====")
        value_counts = self.data_frame[column_name].value_counts()
        for value, count in value_counts.items():
            print(f"Value: {value}, Number of representants: {count}")

class DataPreProcessing:
    def __init__(self,
                 data_frame: pandas.DataFrame,
                 output_column_name,
                 columns_to_drop,
                 columns_to_encode):
        self.data_frame = data_frame
        self.output_column_name = output_column_name
        self.columns_to_drop = columns_to_drop
        self.columns_to_encode = columns_to_encode

    def process_data(self):
        print("\n===== Step - Processing data =====")
        if self.columns_to_drop:
            print("Dropping columns:", self.columns_to_drop)
            self.data_frame = self.data_frame.drop(self.columns_to_drop, axis=1)
        if self.columns_to_encode:
            for column in self.columns_to_encode:
                print("Encoding column:", column)
                label_encoder = LabelEncoder()
                self.data_frame[column] = label_encoder.fit_transform(self.data_frame[column])
        features_data = self.data_frame.drop(self.output_column_name, axis=1)
        output_data = self.data_frame[self.output_column_name]
        print("Features data:\n ", features_data)
        print("Target data:\n ", output_data)
        return features_data, output_data
