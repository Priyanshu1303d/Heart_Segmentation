from src.Heart_Segmentation import logger
from src.Heart_Segmentation.config.configuration import ConfigurationManager
from src.Heart_Segmentation.components.Stage_03_Label_preprocessing import LabelsPreprocessing

STAGE_NAME = "Labels Preprocessing stage"

class LabelsPreprocessingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        labels_preprocessing_config = config.get_labels_preprocessing_config()
        labels_preprocessing = LabelsPreprocessing(labels_preprocessing_config)
        labels_preprocessing.preprocess_all_files()

if __name__ == "__main__":
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
        obj = LabelsPreprocessingPipeline()
        obj.main()
        logger.info(f">>>> stage {STAGE_NAME} completed <<<<<<<<<\n\nx==============x")

    except Exception as e:
        logger.exception(e)
        raise e