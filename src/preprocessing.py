import pandas as pd
from src.preprocess_functions import FeatureEngineering

class FeatureEngineeringModel:
    def __init__(self, data_path, embeddings_train_path, embeddings_test_path, threshold=10):
        self.data_path = data_path
        self.embeddings_train_path = embeddings_train_path
        self.embeddings_test_path = embeddings_test_path
        self.threshold = threshold
        self.feature_engineer = FeatureEngineering(threshold=threshold)
        self.normalize_data = self.feature_engineer.normalize_data

    def load_data(self):
        """Load the data and split it into training and testing datasets."""
        data = pd.read_json(self.data_path, lines=True)
        N = -10000
        data_train = data[:N]
        data_test = data[N:]

        return data_train, data_test

    def prepare_labels(self, data_train, data_test):
        """Prepare the labels for training and testing."""
        y_train = data_train['condition']
        y_test = data_test['condition']

        # Convert labels to binary (0 for 'new', 1 for 'not new')
        y_train = [0 if value == 'new' else 1 for value in y_train]
        y_test = [0 if value == 'new' else 1 for value in y_test]

        return y_train, y_test

    def prepare_features(self, data_train, data_test):
        """Prepare the features for training and testing."""
        X_train = data_train.drop(columns=['condition', 'title'])
        X_test = data_test.drop(columns=['condition', 'title'])

        # Apply feature engineering
        X_train = self.feature_engineer.feature_engineering(X_train)
        X_test = self.feature_engineer.feature_engineering(X_test)

        return X_train, X_test

    def normalize_features(self, X_train, X_test):
        """Normalize the feature data."""
        X_train, X_test, scaler = self.normalize_data(X_train, test_data=X_test, scaler=None)
        return X_train, X_test, scaler

    def preprocess_data(self):
        """Load, process, and prepare the data for training."""
        data_train, data_test = self.load_data()
        y_train, y_test = self.prepare_labels(data_train, data_test)
        X_train, X_test = self.prepare_features(data_train, data_test)
        X_train, X_test, scaler = self.normalize_features(X_train, X_test)
        X_train_embeddings = pd.read_parquet(self.embeddings_train_path).title
        X_test_embeddings = pd.read_parquet(self.embeddings_test_path).title

        return X_train, X_test, y_train, y_test, scaler, X_train_embeddings, X_test_embeddings

# Uso de la clase
if __name__ == "__main__":
    data_path = 'notebooks/MLA_100k_checked_v3.jsonlines'
    embeddings_train_path = 'notebooks/train_embeddings.parquet'
    embeddings_test_path = 'notebooks/test_embeddings.parquet'

    # Crear una instancia del modelo de ingeniería de características
    feature_engineering_model = FeatureEngineeringModel(
        data_path, embeddings_train_path, embeddings_test_path, threshold=10
    )

    # Preprocesar los datos
    X_train, X_test, y_train, y_test, scaler,  X_train_embeddings, X_test_embeddings = feature_engineering_model.preprocess_data()

    print(X_test.shape, X_train.shape)
