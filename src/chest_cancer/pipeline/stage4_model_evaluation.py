from chest_cancer.config.configuration import ConfigurationManager
from chest_cancer.components.model_evaluation import Evaluation
from chest_cancer import logger

STAGE_NAME = "Model Evaluation Stage"

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        logger.info(f"----------> starting {STAGE_NAME}-------------->")
        try:
            config = ConfigurationManager()
            model_eval_config = config.get_model_evaluation_config()
            evaluation = Evaluation(config=model_eval_config)
            evaluation.evaluation()
            logger.info(f"----------> {STAGE_NAME} completed-------------->\n\n")
        except Exception as e:
            logger.error(f"----------> {STAGE_NAME} failed <------------")
            logger.error(e)
            raise e

if __name__ == '__main__':
    try:
        logger.info(f"----------> starting {STAGE_NAME}-------------->")
        evaluation_obj = ModelEvaluationPipeline()
        evaluation_obj.main()
        logger.info(f"----------> {STAGE_NAME} completed-------------->\n\n")
    except Exception as e:
        logger.error(f"----------> {STAGE_NAME} failed <------------")
        logger.error(e)
        raise e