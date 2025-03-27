from src.Heart_Segmentation import logger
from src.Heart_Segmentation.config.configuration import ConfigurationManager
from src.Heart_Segmentation.components.Stage_05_Model_Evaluation import ModelEvaluation

STAGE_NAME = "Model Evaluation stage"

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_model_evaluation_config()
        model_eval = ModelEvaluation(eval_config)
        model_eval.evaluate()
    
if __name__ == "__main__":
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>> stage {STAGE_NAME} completed <<<<<<<<<\n\nx==============x")

    except Exception as e:
        logger.exception(e)
        raise e