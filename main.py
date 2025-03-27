from src.Heart_Segmentation import logger
from src.Heart_Segmentation.pipeline.Stage_01_Data_ingestion import DataIngestionPipeline
from src.Heart_Segmentation.pipeline.Stage_02_Data_Preprocessing import DataPreprocessingPipeline
from src.Heart_Segmentation.pipeline.Stage_03_Labels_Preprocessing import LabelsPreprocessingPipeline
from src.Heart_Segmentation.pipeline.Stage_04_Model_Training import ModelTrainingPipeline
from src.Heart_Segmentation.pipeline.Stage_05_Model_Evaluation import ModelEvaluationPipeline
from src.Heart_Segmentation.pipeline.Stage_06_Model_Optimization import ModelOptimizationPipeline


# STAGE_NAME1 = "Data Ingetsion"

# if __name__ == "__main__":
#     try:
#         logger.info(f"-------------------stage {STAGE_NAME1} started ---------------")
#         training = DataIngestionPipeline()
#         training.main()
#         logger.info(f"-----------stage {STAGE_NAME1} completed successfully ---------<<\n\n----")
#     except Exception as e:
#         logger.exception(e)
#         raise e
    

# STAGE_NAME2 = "Data PreProcessing"

# if __name__ == "__main__":
#     try:
#         logger.info(f"-------------------stage {STAGE_NAME2} started ---------------")
#         training = DataPreprocessingPipeline()
#         training.main()
#         logger.info(f"-----------stage {STAGE_NAME2} completed successfully ---------<<\n\n----")
#     except Exception as e:
#         logger.exception(e)
#         raise e
    

# STAGE_NAME3 = "Labels PreProcessing"

# if __name__ == "__main__":
#     try:
#         logger.info(f"-------------------stage {STAGE_NAME3} started ---------------")
#         training = LabelsPreprocessingPipeline()
#         training.main()
#         logger.info(f"-----------stage {STAGE_NAME3} completed successfully ---------<<\n\n----")
#     except Exception as e:
#         logger.exception(e)
#         raise e
    
# STAGE_NAME4 = "Model Training"

# if __name__ == "__main__":
#     try:
#         logger.info(f"-------------------stage {STAGE_NAME4} started ---------------")
#         training = ModelTrainingPipeline()
#         training.main()
#         logger.info(f"-----------stage {STAGE_NAME4} completed successfully ---------<<\n\n----")
#     except Exception as e:
#         logger.exception(e)
#         raise e
    
# STAGE_NAME5 = "Model Evaluation"

# if __name__ == "__main__":
#     try:
#         logger.info(f"-------------------stage {STAGE_NAME5} started ---------------")
#         training = ModelEvaluationPipeline()
#         training.main()
#         logger.info(f"-----------stage {STAGE_NAME5} completed successfully ---------<<\n\n----")
#     except Exception as e:
#         logger.exception(e)
#         raise e
    

STAGE_NAME6 = "Model Optimization"

if __name__ == "__main__":
    try:
        logger.info(f"-------------------stage {STAGE_NAME6} started ---------------")
        training = ModelOptimizationPipeline()
        training.main()
        logger.info(f"-----------stage {STAGE_NAME6} completed successfully ---------<<\n\n----")
    except Exception as e:
        logger.exception(e)
        raise e