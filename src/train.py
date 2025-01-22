import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA


class XGBoostClassifierModel:
    def __init__(
        self,
        learning_rate=0.3,
        random_state=42,
        use_embeddings=False,
        use_pca=False,
        n_components=100
    ):
        """
        Initialize the XGBoost model with configuration options.

        Parameters:
        - learning_rate: Learning rate for the XGBoost model (default: 0.3).
        - random_state: Random seed for reproducibility (default: 42).
        - use_embeddings: Whether to include embeddings in the model (default: False).
        - use_pca: Whether to apply PCA to the combined features (default: False).
        - n_components: Number of principal components if PCA is used (default: 100).
        """
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.use_embeddings = use_embeddings
        self.use_pca = use_pca
        self.n_components = n_components
        self.model = None

    def train(self, X_train, y_train, X_test, y_test, X_train_embeddings=None, X_test_embeddings=None):
        """
        Train the XGBoost classifier and evaluate performance.

        Parameters:
        - X_train: DataFrame with the training features.
        - y_train: Series or array with the training labels.
        - X_test: DataFrame with the test features.
        - y_test: Series or array with the test labels.
        - X_train_embeddings: Training embeddings (optional if use_embeddings is True).
        - X_test_embeddings: Test embeddings (optional if use_embeddings is True).

        Returns:
        - model: Trained XGBoost model.
        - importance: DataFrame with features and their importance.
        """
        if self.use_embeddings:
            if X_train_embeddings is None or X_test_embeddings is None:
                raise ValueError("X_train_embeddings and X_test_embeddings are required if use_embeddings is True.")

            X_train_embeddings_array = np.vstack(X_train_embeddings)
            X_test_embeddings_array = np.vstack(X_test_embeddings)

            X_train_embeddings_df = pd.DataFrame(
                X_train_embeddings_array,
                columns=[f'title_embedding_{i}' for i in range(X_train_embeddings_array.shape[1])]
            )
            X_test_embeddings_df = pd.DataFrame(
                X_test_embeddings_array,
                columns=[f'title_embedding_{i}' for i in range(X_test_embeddings_array.shape[1])]
            )

            X_train = pd.concat([X_train.reset_index(drop=True), X_train_embeddings_df.reset_index(drop=True)], axis=1)
            X_test = pd.concat([X_test.reset_index(drop=True), X_test_embeddings_df.reset_index(drop=True)], axis=1)

        if self.use_pca:
            pca = PCA(n_components=self.n_components)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

        self.model = xgb.XGBClassifier(
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='logloss',
            learning_rate=self.learning_rate
        )
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        if self.use_pca:
            feature_names = [f'PC{i+1}' for i in range(X_train.shape[1])]
        else:
            feature_names = X_train.columns

        importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': self.model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        print("\nFeature Importances:")
        print(importance)

        return self.model, X_train  # return data after transformation to have a map of the final variables


# Guard clause to prevent execution when imported
if __name__ == "__main__":
    print("This module is meant to be imported into a main.py file.")
