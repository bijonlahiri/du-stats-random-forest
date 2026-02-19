from dataclasses import dataclass

@dataclass
class DUStatsIngestionArtifact:
    raw_data_filepath: str
    train_data_filepath: str
    test_data_filepath: str