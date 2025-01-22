import pickle
from src.train import XGBoostClassifierModel
from src.preprocessing import FeatureEngineeringModel

class MainPipeline:
    def __init__(self, 
                 data_path, 
                 embeddings_train_path, 
                 embeddings_test_path, 
                 model_save_path='model.pkl', 
                 threshold=10, 
                 learning_rate=0.3, 
                 random_state=42, 
                 use_embeddings=True, 
                 use_pca=False, 
                 n_components=100):
        """Initialize the pipeline with necessary configurations."""
        self.data_path = data_path
        self.embeddings_train_path = embeddings_train_path
        self.embeddings_test_path = embeddings_test_path
        self.model_save_path = model_save_path

        self.feature_engineering_model = FeatureEngineeringModel(
            data_path, embeddings_train_path, embeddings_test_path, threshold
        )

        self.model = XGBoostClassifierModel(
            learning_rate=learning_rate,
            random_state=random_state,
            use_embeddings=use_embeddings,
            use_pca=use_pca,
            n_components=n_components
        )

    def run_pipeline(self):
        """Run the entire pipeline: preprocessing, training, and saving the model."""
        # Step 1: Preprocess data
        print("Starting data preprocessing...")
        X_train, X_test, y_train, y_test, scaler, X_train_embeddings, X_test_embeddings = self.feature_engineering_model.preprocess_data()

        # Step 2: Train the model
        print("Training the model...")
        trained_model, transformed_data = self.model.train(
            X_train,
            y_train,
            X_test,
            y_test,
            X_train_embeddings=X_train_embeddings,
            X_test_embeddings=X_test_embeddings
        )

        # Step 3: Save the trained model
        print(f"Saving the model to {self.model_save_path}...")
        self.save_model(trained_model)

        print("Pipeline execution completed.")

    def save_model(self, model):
        """Save the trained model to a file."""
        with open(self.model_save_path, 'wb') as f:
            pickle.dump(model, f)

if __name__ == "__main__":
    # Define paths and parameters
    DATA_PATH = 'notebooks/MLA_100k_checked_v3.jsonlines'
    EMBEDDINGS_TRAIN_PATH = 'notebooks/train_embeddings.parquet'
    EMBEDDINGS_TEST_PATH = 'notebooks/test_embeddings.parquet'
    MODEL_SAVE_PATH = 'model.pkl'

    # Initialize and run the pipeline
    pipeline = MainPipeline(
        data_path=DATA_PATH,
        embeddings_train_path=EMBEDDINGS_TRAIN_PATH,
        embeddings_test_path=EMBEDDINGS_TEST_PATH,
        model_save_path=MODEL_SAVE_PATH
    )

    pipeline.run_pipeline()
