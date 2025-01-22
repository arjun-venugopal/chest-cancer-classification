from chest_cancer.config.configuration import ConfigurationManager
from chest_cancer.components.model_evaluation import Evaluation
from chest_cancer import logger

STAGE_NAME = "Model Evaluation Stage"

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_model_evaluation_config()
        evaluation = Evaluation(eval_config)
        evaluation.evaluation()
        evaluation.log_into_mlflow()

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
