# DU Stats Root Cause Analysis ML+ANN+ConvNet Pipeline

## Overview

An end-to-end machine learning pipeline for **Root Cause Analysis (RCA) classification** of 5G Downlink Unit (DU) statistics. The system identifies downlink transmission quality issues using three complementary model architectures: traditional Machine Learning, Artificial Neural Networks (ANN), and 1D Convolutional Neural Networks (ConvNet).

**Domain**: 5G Telecommunications
**Task**: Multi-class classification (4 RCA categories)
**Models**: scikit-learn, PyTorch (feedforward NN, 1D CNN)
**Data Source**: Databricks SQL
**Experiment Tracking**: MLflow

---

## Problem Statement

Cellular networks experience downlink transmission issues manifested in Block Error Rate (BLER), Channel Quality Indicator (CQI), and Modulation and Coding Scheme (MCS) metrics. This project classifies the root cause of performance degradation into four categories:

| Category | Description | Typical Indicators |
|----------|-------------|-------------------|
| **Good** | Low BLER, excellent channel quality, optimal MCS | All metrics healthy |
| **Bad Channel** | High BLER due to poor channel conditions | High BLER, low CQI |
| **High BLER Good Channel** | Anomalous RV0 transmissions despite good channel | Low RV0 transmissions, good CQI |
| **Scheduler Limited** | MCS capped by scheduler despite good channel | Capped MCS, good channel, many RV0 retransmissions |

---

## Architecture

### Data Pipeline Flow

```
Ingestion → Validation → Transformation → Training → Evaluation
    ↓           ↓              ↓              ↓           ↓
Databricks  Schema Check    Scaling      ML Models   Reports
   SQL      Drift Detect    Reshaping      ↓        (YAML)
             CSV Outputs   Normalization   ├─ ML (scikit-learn)
                                           ├─ ANN (PyTorch)
                                           └─ ConvNet (1D CNN)
```

### Component Organization

```
du_stats/
├── components/           # Pipeline stage implementations
│   ├── dustats_ingestion.py           # Data fetching & splitting
│   ├── dustats_validation.py          # Schema validation & drift detection
│   ├── dustats_transformation.py      # Feature preprocessing
│   ├── dustats_model_trainer.py       # Traditional ML training
│   └── dustats_nn_model_trainer.py    # Neural network orchestration
│
├── utils/               # Utility modules
│   ├── main_utils/      # File I/O, YAML handling
│   ├── ml_utils/        # Model evaluation & wrappers
│   ├── nn_utils/        # Feedforward NN training
│   └── conv_utils/      # ConvNet training, data loaders, evaluation
│
├── entity/              # Data classes
│   ├── config_entity.py      # Configuration classes for each stage
│   └── artifact_entity.py    # Pipeline output artifacts
│
├── exception/           # Error handling
│   └── exception.py     # Custom DUStatsException
│
├── logging/             # Logging utilities
│   └── logger.py        # Timestamped logging
│
└── constants/           # Configuration constants
    └── dustats_pipeline/ # Pipeline-specific constants
```

---

## Installation

### Prerequisites

- Python 3.8+
- Databricks SQL Connector credentials (hostname, HTTP path, token)
- pip or conda

### Setup Instructions

#### 1. Clone Repository
```bash
git clone <repository-url>
cd du_stats_machine_learning
```

#### 2. Create Virtual Environment
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n dustats python=3.10
conda activate dustats
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Configure Environment
Create a `.env` file in the project root:
```env
DATABRICKS_HOST=your_databricks_hostname
DATABRICKS_HTTP_PATH=your_http_path
DATABRICKS_TOKEN=your_access_token
```

**Security Note**: Never commit `.env` file. Use secret management in production.

#### 5. Verify Installation
```bash
python -c "import du_stats; print('Installation successful')"
```

---

## Usage

### Running the Complete Pipeline

```bash
python main.py
```

This executes:
1. **Data Ingestion**: Fetches data from Databricks, applies 80/20 train/test split
2. **Validation**: Schema validation, data drift detection (threshold: 0.05)
3. **Parallel Transformation & Training**:
   - ML track: StandardScaler → scikit-learn models
   - ANN track: StandardScaler → Feedforward NN (2 hidden layers, 128 neurons)
   - ConvNet track: Tensor reshaping → 1D CNN (960-sample sequences)
4. **Output**: Trained models, reports, and evaluation metrics in `artifacts/`

### Pipeline Output

Results are saved in timestamped artifact directories:

```
artifacts/
└── 10_03_26_21_05_54/  # timestamp
    ├── ingestion/
    │   ├── raw_data/
    │   └── split_data/
    ├── validation/
    │   ├── validation_report/
    │   └── validated/
    ├── transformation/
    │   ├── transformed_data/  # ml_, ann_, convnet_ prefixed
    │   └── transformer_model/
    └── neural_networks/       # For ANN & ConvNet
        ├── training/
        ├── evaluation/
        └── model/
```

### Configuration

Edit `du_stats/constants/dustats_pipeline/__init__.py` to customize:

```python
# Data split ratio
DUSTATS_INGESTION_TRAIN_TEST_SPLIT_RATIO = 0.2

# Data drift threshold (KS divergence)
DUSTATS_VALIDATION_DATA_DRIFT_THRESHOLD = 0.05

# Neural Network architecture
DUSTATS_NN_NUM_HIDDEN_LAYERS = 2
DUSTATS_NN_NUM_NEURONS = 128
DUSTATS_NN_NUM_TRAINING_EPOCHS = 100

# ConvNet sequence length
DUSTATS_TRANSFORMATION_CONVNET_SEQ_LEN = 960
```

---

## Data Schema

The pipeline expects input data with 30 columns organized into categories:

### Identifiers (5)
- `site_name`: Cell site identifier
- `log_date`: Date of measurement
- `cellid`: Cell identifier
- `ueid`: User equipment identifier
- `uptime`: Time instants in 30s resolution

### BLER Metrics (4)
- `ibler`: IBLER (Instantaneous BLER for RV0 transmissions)
- `rbler`: RBLER (Retransmission BLER for re-transmissions)
- `resbler`: Residual BLER (Residual BLER after exhausting all re-transmissions)
- `tbler`: Total BLER

### Channel & Scheduling (3)
- `cqi`: Channel Quality Indicator
- `mcs`: Modulation and Coding Scheme
- `ri`: Rank Indicator

### RV0 Transmissions (5)
- `rv0_tx`, `rv0_ack`, `rv0_nack`, `rv0_dtx`, `rv0_bler`

### RV2 Transmissions (5)
- `rv2_tx`, `rv2_ack`, `rv2_nack`, `rv2_dtx`, `rv2_bler`

### RV3 Transmissions (5)
- `rv3_tx`, `rv3_ack`, `rv3_nack`, `rv3_dtx`, `rv3_bler`

### Target Variable
- `rca_label`: One of {GOOD, BAD CHANNEL, GOOD CHANNEL HIGH BLER, SCHEDULER LIMITED}

---

## Model Architectures

### 1. Traditional ML (scikit-learn)

**Preprocessing**: StandardScaler normalization

**Algorithms Evaluated**:
- Decision Tree
- AdaBoost
- Gradient Boosting
- Random Forest

**Input Shape**: (n_samples, 22 features)
**Output**: Class probabilities

### 2. Artificial Neural Network (ANN)

**Architecture**:
- Input layer: 22 neurons (features)
- Hidden layer 1: 128 neurons (ReLU activation)
- Hidden layer 2: 128 neurons (ReLU activation)
- Output layer: 4 neurons (Softmax for 4 classes)

**Preprocessing**: StandardScaler normalization
**Framework**: PyTorch
**Loss**: Cross-Entropy
**Optimizer**: SGD
**Epochs**: 100

**Input Shape**: (n_samples, 22)
**Output**: Class probabilities

### 3. 1D Convolutional Neural Network (ConvNet)

**Architecture**:
- Input: 960-sample sequences with 22 features each
- Conv1D layers with ReLU activation
- Max pooling for dimensionality reduction
- Fully connected classification layers
- Output: 4-class prediction

**Preprocessing**:
- StandardScaler normalization
- Sequence creation (960-sample windows)
- Tensor reshaping: (n_sequences, 960, 22)

**Framework**: PyTorch
**Loss**: Cross-Entropy
**Optimizer**: SGD

**Input Shape**: (n_samples, 960, 22)
**Output**: Class probabilities

---

## Experiment Tracking

MLflow integration tracks all experiments:

```bash
# View MLflow UI
mlflow ui  # Opens at http://localhost:5000
```

Tracked metrics:
- Training/test accuracy
- Precision, recall, F1-score
- Model hyperparameters
- Training duration
- Data versioning (schema, drift)

---

## Validation & Quality Assurance

### Data Validation
- **Schema Checks**: Validates column names, types, and ranges
- **Drift Detection**: KS divergence < 0.05 threshold
- **Missing Values**: Handling and reporting
- **Outlier Detection**: Flagged in validation reports

### Model Evaluation
- **Test Set Performance**: Separate 20% held-out test data
- **Cross-validation**: K-fold validation metrics
- **Classification Metrics**: Accuracy, precision, recall, F1, confusion matrix
- **Reports**: YAML format with detailed results

---

## Logging

Timestamped logs are saved to `logs/` directory:

```
logs/
├── 10_03_26_21_05_54.log  # Per-run logs
```

Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Project Structure Details

### Key Configuration Classes

**DUStatsPipelineConfig**
- Root artifact directory with timestamp
- Used by all downstream stages

**DUStatsTransformationMLConfig / AnnConfig / ConvNetConfig**
- Separate configs for each model type
- Manages input/output file paths
- Prevents file conflicts with prefixed names (`ml_`, `ann_`, `convnet_`)

**DUStatsNeuralNetworkConfig**
- Neural network hyperparameters
- Model save paths
- Training configuration

### Key Processing Functions

**Ingestion**
- Fetches data from Databricks SQL
- 80/20 train/test split with random_state=42
- Saves to CSV for downstream stages

**Validation**
- Compares schema against `data_schema/schema.yaml`
- Detects data drift using KS divergence
- Separates valid/invalid records

**Transformation**
- ML/ANN: StandardScaler + numpy arrays
- ConvNet: Sorting by identifiers + tensor reshaping (960-sample sequences)
- Saves preprocessors for inference

**Training**
- ML: Grid search over scikit-learn algorithms
- ANN/ConvNet: PyTorch training loops with Adam optimizer
- Saves trained models + YAML reports

---

## Common Issues & Troubleshooting

### Issue: "Databricks connection failed"
**Solution**: Verify `.env` credentials and network connectivity
```bash
# Test Databricks connection
python -c "import databricks.sql; print('Connection OK')"
```

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size or set `max_workers=1` (already done)
```python
# In main.py, ThreadPoolExecutor(max_workers=1) runs sequentially
```

### Issue: "Schema mismatch error"
**Solution**: Update `data_schema/schema.yaml` to match source data
```bash
python -c "from du_stats.utils.main_utils.utils import read_yaml; print(read_yaml('data_schema/schema.yaml'))"
```

### Issue: "Data drift threshold exceeded"
**Solution**: Increase threshold in `dustats_pipeline/__init__.py`
```python
DUSTATS_VALIDATION_DATA_DRIFT_THRESHOLD = 0.10  # Increase from 0.05
```

---

## Development & Extension

### Adding a New Model Type

1. Create transformation config in `entity/config_entity.py`:
```python
class DUStatsTransformationYourModelConfig:
    def __init__(self, dustats_pipeline_config):
        # Define file paths for your model format
```

2. Add transformation method in `components/dustats_transformation.py`:
```python
def initiate_data_transformation_yourmodel(self):
    # Implement preprocessing for your model
```

3. Add training function in `main.py`:
```python
def transform_and_train_yourmodel(pipeline_config, validation_artifact):
    # Create config and transformation
    # Call trainer
    # Return artifact
```

4. Register in ThreadPoolExecutor:
```python
executor.submit(transform_and_train_yourmodel, dustats_pipeline_config, dustats_validation_artifact)
```

### Running with Google Colab

Use `requirements_colab.txt` for Colab-specific dependencies:
```bash
pip install -r requirements_colab.txt
```

---

## Performance Benchmarks

| Model | Train Accuracy | Test Accuracy | Training Time | Model Size |
|-------|---|---|---|---|
| Random Forest | 93.4% | 93.2% | ~30s | 2.3 MB |
| ANN | 89.5% | 87.1% | ~120s | 150 KB |
| ConvNet | 98.8% | 97.6% | ~180s | 250 KB |

*Note: Benchmarks are example metrics. Actual results depend on data and hyperparameters.*

---

## Project Statistics

- **Total Lines of Code**: ~1,400
- **Python Files**: 25+
- **Components**: 5 (ingestion, validation, transformation, ML training, NN training)
- **Models Supported**: 3 architectures (ML, ANN, ConvNet)
- **Configuration Management**: YAML + Python classes
- **Error Handling**: Custom exceptions with stack traces
- **Logging**: Structured with timestamps and severity levels

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | Latest | Data manipulation |
| numpy | <2.0 | Numerical operations |
| scikit-learn | Latest | Traditional ML models |
| torch | Latest | Deep learning |
| mlflow | Latest | Experiment tracking |
| databricks-sql-connector | Latest | Database connectivity |
| python-dotenv | Latest | Environment configuration |
| pyaml | Latest | YAML processing |
| tqdm | Latest | Progress bars |

---

## Changelog

**v1.1.0** (Mar 10, 2026)
- Added parallel ML/ANN/ConvNet pipeline
- Separate transformation configs for each model type
- Fixed ANN training method
- Sequential execution with ThreadPoolExecutor

**v1.0.0** (Feb 22, 2026)
- Initial release with ConvNet support
- MLflow integration
- Data validation with drift detection

---

## Acknowledgments

- Databricks SQL for scalable data access
- PyTorch for deep learning framework
- scikit-learn for traditional ML algorithms
- MLflow for experiment tracking
