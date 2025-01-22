import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from chest_cancer.entity.config_entity import ModelEvaluationConfig
from chest_cancer.utils.commen import save_json

import dagshub
dagshub.init(repo_owner='ajuarjun528', repo_name='chest-cancer-classification', mlflow=True)


class Evaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    
    def _valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.trainig_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )


    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    

    def evaluation(self):
        self.model = self.load_model(self.config.model_path)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    
    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_url)
        tracking_url = urlparse(mlflow.get_tracking_uri()).scheme
    
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({"loss": self.score[0], "accuracy": self.score[1]})
    
            if tracking_url != "file":
                try:
                    # Ensure valid model registry parameters
                    mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
                except Exception as e:
                    print(f"Error registering model: {e}")
                    raise e
            else:
                mlflow.keras.log_model(self.model, "model")
