from src.Heart_Segmentation import logger
from src.Heart_Segmentation.config.configuration import ConfigurationManager
from src.Heart_Segmentation.components.Stage_04_Model_Training import ModelTraining

STAGE_NAME = "Model Training stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        get_model_trainig_config = config.get_Model_training_config()
        model_training = ModelTraining(get_model_trainig_config)
        model_training.train()
        logger.info("Visualizing predictions...")
        model_training.visualize_predictions(num_samples=3)
    
if __name__ == "__main__":
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>> stage {STAGE_NAME} completed <<<<<<<<<\n\nx==============x")

    except Exception as e:
        logger.exception(e)
        raise e