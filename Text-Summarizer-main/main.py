from src.textSummarizer.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.textSummarizer.logging import logger
from src.textSummarizer.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from src.textSummarizer.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from src.textSummarizer.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
from src.textSummarizer.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline


STAGE_NAME5 = "Model Evaluation stage"
try:
    logger.info(f">>>>>> {STAGE_NAME5} started <<<<<<")
    model_evaluation = ModelEvaluationTrainingPipeline()
    model_evaluation.main()
    logger.info(f">>>>> stage {STAGE_NAME5} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e