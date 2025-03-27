from Heart_Segmentation import logger
from Heart_Segmentation.config.configuration import ConfigurationManager
from Heart_Segmentation.components.Stage_02_Data_Preprocessing import DataPreprocessing

STAGE_NAME = "Data Preprocessing"

class DataPreprocessingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_preprocessing_config = config.get_data_preprocessing_config()
        data_preprocesing = DataPreprocessing(data_preprocessing_config)
        data_preprocesing.preprocess_all_files()

if __name__ == "__main__":
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
        obj = DataPreprocessingPipeline()
        obj.main()
        logger.info(f">>>> stage {STAGE_NAME} completed <<<<<<<<<\n\nx==============x")

    except Exception as e:
        logger.exception(e)
        raise e