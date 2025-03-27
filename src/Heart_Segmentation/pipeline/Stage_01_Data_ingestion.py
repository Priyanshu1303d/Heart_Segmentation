from Heart_Segmentation import logger
from Heart_Segmentation.config.configuration import ConfigurationManager
from Heart_Segmentation.components.Stage_01_Data_Ingestion import DataIngestion

STAGE_NAME = "Data Ingestion stage"

class DataIngestionPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingetsion_config()
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion.download_dataset()
        data_ingestion.zip_extract()

if __name__ == "__main__":
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
        obj = DataIngestionPipeline()
        obj.main()
        logger.info(f">>>> stage {STAGE_NAME} completed <<<<<<<<<\n\nx==============x")

    except Exception as e:
        logger.exception(e)
        raise e