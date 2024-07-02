import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import read_yaml, create_directories,save_json
import json
import psutil

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model = None
        self.valid_generator = None
        self.score = None

        # Enable memory growth for GPUs
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("Enabled memory growth for GPUs")
            except RuntimeError as e:
                print(f"Error enabling memory growth for GPUs: {e}")

    def _valid_generator(self):
        try:
            datagenerator_kwargs = dict(
                rescale=1./255,
                validation_split=0.30
            )

            dataflow_kwargs = dict(
                target_size=self.config.params_image_size[:-1],
                batch_size=2,  # Further reduced batch size
                interpolation="bilinear"
            )

            valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                **datagenerator_kwargs
            )

            self.valid_generator = valid_datagenerator.flow_from_directory(
                directory=self.config.training_data,
                subset="validation",
                shuffle=False,
                **dataflow_kwargs
            )
            print("Validation generator set up successfully.")
        except Exception as e:
            print(f"An error occurred while setting up validation generator: {e}")
            raise e

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        try:
            return tf.keras.models.load_model(path)
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")
            raise e

    def evaluate(self):
        try:
            print("Loading model...")
            self.model = self.load_model(self.config.path_of_model)
            print("Model loaded successfully.")

            print("Setting up validation generator...")
            self._valid_generator()
            print("Validation generator set up successfully.")

            print("Evaluating model...")

            # Track memory usage before evaluation
            mem_before = psutil.virtual_memory().available
            print(f"Available memory before evaluation: {mem_before} bytes")

            # Evaluate model
            self.score = self.model.evaluate(self.valid_generator)

            # Track memory usage after evaluation
            mem_after = psutil.virtual_memory().available
            print(f"Available memory after evaluation: {mem_after} bytes")

            print(f"Model evaluated successfully. Score: {self.score}")

            self.save_score()
            print("Scores saved successfully.")
        except Exception as e:
            print(f"An error occurred during evaluation: {e}")
            raise e
        finally:
            # Clear session to free memory
            tf.keras.backend.clear_session()
            print("Session cleared.")

    def save_score(self):
        try:
            scores = {"loss": self.score[0], "accuracy": self.score[1]}
            with open("scores.json", "w") as f:
                json.dump(scores, f)
            print("Scores saved to scores.json.")
        except Exception as e:
            print(f"An error occurred while saving scores: {e}")
            raise e

    def log_into_mlflow(self):
        try:
            mlflow.set_registry_uri(self.config.mlflow_uri)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            with mlflow.start_run():
                mlflow.log_params(self.config.all_params)
                mlflow.log_metrics({"loss": self.score[0], "accuracy": self.score[1]})

                if tracking_url_type_store != "file":
                    mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
                else:
                    mlflow.keras.log_model(self.model, "model")
            print("Logged into MLflow successfully.")
        except Exception as e:
            print(f"An error occurred while logging into MLflow: {e}")
            raise e

