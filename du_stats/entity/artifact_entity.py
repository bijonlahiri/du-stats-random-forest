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