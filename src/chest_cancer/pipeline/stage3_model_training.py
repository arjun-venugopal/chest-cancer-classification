from chest_cancer.config.configuration import ConfigurationManager
from chest_cancer.components.model_training import Training
from chest_cancer import logger

STAGE_NAME = "Model Training Stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_model_config = config.get_training_model_config()
        training = Training(config=training_model_config)
        training.get_base_model()
        training.train_valid_generator()
        training.train()

if __name__ == '__main__':
    try:
        logger.info(f"----------> starting {STAGE_NAME}-------------->")
        Training_obj = ModelTrainingPipeline()
        Training_obj.main()
        logger.info(f"----------> {STAGE_NAME} completed-------------->\n\n")
    except Exception as e:
        logger.error(f"----------> {STAGE_NAME} failed <------------")
        logger.error(e)
        raise e