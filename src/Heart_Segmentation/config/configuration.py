from Heart_Segmentation.constants import *
from Heart_Segmentation.utils.common import read_yaml ,create_directories
from Heart_Segmentation.entity.config_entity import (DataIngestionConfig , DataPreprocessingConfig , LabelsPreprocessingConfig
                                                         , ModelTrainingConfig, ModelEvaluationConfig,
                                                         ModelOptimizationConfig)


class ConfigurationManager:
    def __init__(self , config_filepath = CONFIG_FILE_PATH, params_filepath = PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    def get_data_ingetsion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir= config.root_dir,
            source_url= config.source_url,
            local_data_path= config.local_data_path,
            unzip_path= config.unzip_path
        )

        return data_ingestion_config
    
    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        config = self.config.data_preprocessing

        create_directories([config.root_dir , config.output_dir])


        data_preprocessing_config = DataPreprocessingConfig(
            root_dir= config.root_dir,
            input_dir= config.input_dir,
            output_dir = config.output_dir,
            target_size= config.target_size
        )

        return data_preprocessing_config
    
    def get_labels_preprocessing_config(self) -> LabelsPreprocessingConfig:
        config = self.config.labels_preprocessing
        create_directories([config.root_dir, config.output_dir])
        return LabelsPreprocessingConfig(
            root_dir=config.root_dir,
            input_dir=config.input_dir,
            output_dir=config.output_dir,
            target_size=config.target_size
        )
    

    def get_Model_training_config(self) -> ModelTrainingConfig:

        config = self.config.model_training 
        create_directories([config.root_dir ])

        model_training_config = ModelTrainingConfig(
            root_dir = Path(config.root_dir),
            images_dir= Path(config.images_dir),
            labels_dir= Path(config.labels_dir),
            model_save_path= Path(config.model_save_path),
            params_batch_size= self.params.BATCH_SIZE,
            params_dropout_rate= self.params.DROPOUT_RATE,
            params_epochs= self.params.EPOCHS,
            params_learning_rate= self.params.LEARNING_RATE,
            params_weight_decay= self.params.WEIGHT_DECAY
        )

        return model_training_config
    

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            images_dir=Path(config.images_dir),
            labels_dir=Path(config.labels_dir),
            model_metrics_json=Path(config.model_metrics_json),
            model_save_path = Path(config.model_save_path),
            params_batch_size=self.params.BATCH_SIZE
        )
        return model_evaluation_config
    
    def get_model_optimization_config(self, trial) -> ModelOptimizationConfig:
        config = self.config.model_optimization
        create_directories([config.root_dir])

        # Get hyperparameters from Optuna trial
        model_optimization_config = ModelOptimizationConfig(
            root_dir=Path(config.root_dir),
            images_dir=Path(config.images_dir),
            labels_dir=Path(config.labels_dir),
            model_save_path=Path(config.model_save_path),
            model_metrics_json=Path(config.model_metrics_json),
            best_model_path=Path(config.best_model_path),
            params_epochs=trial.suggest_int('epochs', 5, 10),
            params_learning_rate=trial.suggest_float('lr', 1e-5, 1e-3, log=True),
            params_batch_size=trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            params_weight_decay=trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True),
            params_dropout_rate=trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.1),
            params_optimizer_name=trial.suggest_categorical('Optimizer_name', ['Adam', 'SGD', 'RMSprop'])
        )
        return model_optimization_config