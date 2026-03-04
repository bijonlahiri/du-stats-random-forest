from dataclasses import dataclass

@dataclass
class DUStatsIngestionArtifact:
    ingestion_done:bool=False
    raw_data_filepath: str=None
    train_data_filepath: str=None
    test_data_filepath: str=None

@dataclass
class DUStatsValidationArtifact:
    validation_completed:bool=False
    validation_report_filepath: str=None
    valid_train_data_filepath: str=None
    valid_test_data_filepath: str=None
    invalid_train_data_filepath: str=None
    invalid_test_data_filepath: str=None

@dataclass
class DUStatsTransformationArtifact:
    transformation_done:bool=False
    transformed_train_data_filepath:str=None
    transformed_test_data_filepath:str=None
    preprocessor_filepath:str=None

@dataclass
class DUStatsModelTrainerArtifact:
    model_training_done:bool=False
    model_filepath:str=None
    model_report_filepath:str=None
    best_model_name:str=None
    best_model_params:dict=None

@dataclass
class DUStatsNeuralNetworkArtifact:
    nn_training_done:bool=False
    nn_training_report_filepath:str=None