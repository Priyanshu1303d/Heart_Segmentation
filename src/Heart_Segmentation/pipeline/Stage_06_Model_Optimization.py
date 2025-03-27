from src.Heart_Segmentation import logger
from src.Heart_Segmentation.config.configuration import ConfigurationManager
from src.Heart_Segmentation.components.Stage_06_Model_Optimization import ModelOptimization , objective
import optuna



STAGE_NAME = "Model Optimization stage"

class ModelOptimizationPipeline:
    def __init__(self):
        pass

    def main(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=5)  # Adjust number of trials as needed
        logger.info(f"Best trial: {study.best_trial.params}, Best accuracy: {study.best_value}")

    def objective(trial):
        config = ConfigurationManager()
        config = config.get_model_optimization_config(trial)
        
        model_opt = ModelOptimization(config)
        model_opt.train()
        accuracy = model_opt.evaluate()
        
        # Save the best model
        model_opt.save_best_model(accuracy)
        
        return accuracy
    
if __name__ == "__main__":
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
        obj = ModelOptimizationPipeline()
        obj.main()
        logger.info(f">>>> stage {STAGE_NAME} completed <<<<<<<<<\n\nx==============x")

    except Exception as e:
        logger.exception(e)
        raise e