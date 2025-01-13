from chest_cancer.config.configuration import ConfigurationManager
from chest_cancer.components.base_model import BaseModelSelection
from chest_cancer import logger

STAGE_NAME = "Base Model Selection Stage"


class BaseModelTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        base_model_config = config.get_base_model_config()
        base_model = BaseModelSelection(config=base_model_config)
        base_model.get_base_model()
        base_model.update_base_model()


if __name__ == '__main__':
    try:
        logger.info(f"----------> starting {STAGE_NAME}-------------->")
        obj = BaseModelTrainingPipeline()
        obj.main()
        logger.info(f"----------> {STAGE_NAME} completed-------------->\n\n")
    except Exception as e:
        logger.error(f"----------> {STAGE_NAME} failed <------------")
        logger.error(e)
        raise e