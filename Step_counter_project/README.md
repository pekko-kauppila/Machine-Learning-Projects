# Step Detection with Neural Networks from Hip-Based Accelerometer Data

Step detection and counting from hip-based accelerometer data using 1D Convolutional Neural Networks.

## Dataset

This project uses the OxWalk Annotated Step Count Dataset containing triaxial accelerometer data from 39 healthy adults with video-annotated ground truth step counts.

Download: https://doi.org/10.5287/bodleian:KJe7VgMNV

After downloading, extract the data to `data/OxWalk_Dec2022/`:
```
Step_counter_project/
└── data/
    └── OxWalk_Dec2022/
        ├── Hip_100Hz/
        ├── Hip_25Hz/
        ├── Wrist_100Hz/
        ├── Wrist_25Hz/
        └── metadata.csv
```

### Explanatory Data Analysis

Run eda.ipynb to get a general understanding of the data

### Generate Windowed Data

Run the data_preparation_cnn.ipynb notebook to create input data for the CNN

Run all cells. This will generate processed data files in `data/processed/`:
- `cnn_train_data.npz`
- `cnn_val_data.npz`
- `cnn_test_data.npz`
- `cnn_normalization.npz`
- Metadata CSV files

### Training Models

#### Local Training (CPU/GPU in IDE)

Open and run the local training notebook model_development.ipynb.
The notebook uses models defined in src/models folder.
Run all cells to train and save the model to `models/saved/[experiment_name]/`.

#### Google Colab Training (GPU) with hyperparam looping

For GPU training if otherwise unavailable.

1. **Upload processed data to Google Drive**:
   ```
   /MyDrive/path_to_project_folder/data/processed/
   ├── cnn_train_data.npz
   ├── cnn_val_data.npz
   └── cnn_test_data.npz
   ```

2. **Upload notebook to Colab**:
   - Upload `/MyDrive/path_to_project_folder/notebooks/hyperparameter_search_colab.ipynb` for automated grid search

3. **Configure runtime**:
   - Runtime → Change runtime type → GPU (T4)

4. **Update paths** in the notebook:
   ```python
   DRIVE_DATA_DIR = Path('/content/drive/MyDrive/path_to_project_folder/data/processed')
   DRIVE_SAVE_DIR = Path('/content/drive/MyDrive/path_to_project_folder/models/saved')
   ```

### Model Evaluation

Run evaluation.ipynb to evaluate a single model.

In the notebook, specify the model path:
```python
model_path = 'models/saved/deep_cnn_20240315_143022/final_model.pth'
```

Run all cells to see:
- Test set metrics (MAE, RMSE, R²)
- Prediction vs actual plots
- Error distribution
- Per-participant performance

### Compare Multiple Models

Run compare_experiments.ipynb

This notebook reads `models/saved/experiments.csv` and visualizes performance across models.

## Available Models

- **ShallowCNN**: Single convolutional layer baseline
- **DeepCNN**: 5-layer CNN with batch normalization and global pooling

Each model supports both:
- **Binary classification**: Step detection (is stepping / not stepping)
- **Regression**: Step count prediction

