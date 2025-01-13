import os
import urllib.request as request
import tensorflow as tf
from zipfile import ZipFile
from pathlib import Path
from tensorflow.keras.applications import VGG16
from chest_cancer.entity.config_entity import BaseModelConfig


class BaseModelSelection:
    def __init__(self, config: BaseModelConfig):
        self.config = config

    def get_base_model(self):
        self.model = VGG16(
            input_shape=self.config.params_image_size,
            include_top=self.config.params_include_top,
            weights=self.config.params_weights
        )

        self.save_model(path=self.config.model_path, model=self.model)

    @staticmethod
    def prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        # Freeze layers
        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False

         # Add new layers on top of the base model
        flatten_in = tf.keras.layers.Flatten()(model.output)
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(flatten_in)

        # Create a new model
        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        # Compile the model
        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model
    

    def update_base_model(self):
        self.full_model = self.prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_model_path, model=self.full_model)
    


    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)