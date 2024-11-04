# E-commerce Product Attribute Prediction

This project aims to predict multiple attributes of e-commerce products from images using a hybrid CNN-RNN model. The system enhances product listings by automatically generating attributes for each item, optimizing for varying categories and attribute numbers. This model is particularly useful for e-commerce platforms to improve product categorization and search experience.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Training and Evaluation](#training-and-evaluation)
- [Submission File Generation](#submission-file-generation)
- [Customization](#customization)
- [Results and Analysis](#results-and-analysis)
- [Acknowledgments](#acknowledgments)

## Project Overview
The model predicts 10 attributes for each product, even for categories that require fewer than 10 attributes. For categories with fewer attributes, the model fills unused attributes with a placeholder (`"dummy_value"`), based on the number of attributes needed for each category.

## Dataset
The dataset includes:
- **Product Images**: Images of products organized in `train_images` and `test_images` directories.
- **Attributes CSV**: `train.csv` and `test.csv` files with attribute labels for each product.
- **Category Attributes Parquet**: `category_attributes.parquet` defines the number of required attributes per category.

## Model Architecture
This project utilizes a hybrid **CNN-RNN** approach:
1. **CNN Backbone (ResNet101)**: Extracts visual features from product images.
2. **LSTM Layer**: Sequentially processes the features for predicting each attribute.
3. **Fully Connected Layers**: Independent layers for each attribute to handle multi-output prediction.

## Installation
Clone the repository and install the required packages.
```bash
git clone https://github.com/your-username/ecommerce-attribute-prediction.git
cd ecommerce-attribute-prediction
pip install -r requirements.txt
```

## Usage

### 1. Prepare Data
Ensure `train.csv`, `test.csv`, `train_images`, `test_images`, and `category_attributes.parquet` are in the project directory.

### 2. Training the Model
Run the script to train the model.
```bash
python train.py
```

### 3. Predict and Generate Submission
After training, generate predictions and save the submission file.
```bash
python predict.py
```

## Training and Evaluation
The training script implements:
- **Cross-Entropy Loss**: Computed separately for each attribute, accommodating multi-label classification.
- **Learning Rate Scheduler**: To improve optimization, decreasing the learning rate every 3 epochs.
- **Data Augmentation**: Random cropping, flipping, and color jitter to enhance generalization.

### Key Parameters
- `num_epochs`: Number of epochs (default: 1)
- `batch_size`: Size of each batch (default: 32)
- `learning_rate`: Learning rate for optimizer (default: 0.001)

### Model Checkpointing
The model can be checkpointed to save intermediate weights for further tuning and evaluation.

## Submission File Generation
1. **Predictions**: After inference, predictions for each attribute are saved in `submission5.csv`.
2. **Handling Missing Attributes**: Attributes that are not required by a category are set to `"dummy_value"`.
3. **Category-Specific Requirements**: The `category_attributes.parquet` file is used to dynamically fill only the required number of attributes for each category.

## Customization
You can modify the following for enhanced performance:
- **CNN Backbone**: Experiment with deeper or lighter CNNs (e.g., ResNet50, EfficientNet) by replacing `resnet101` in the code.
- **Data Augmentation**: Adjust transformations in `data_transforms` for better generalization on different datasets.
- **Loss Function**: Adjust the weighting if certain classes are underrepresented.
- **Learning Rate Scheduler**: Fine-tune the scheduling to improve model convergence.

## Results and Analysis
Once training is complete, examine the logs for model performance metrics:
- **Loss per Epoch**: Evaluate the overall and attribute-specific loss for each epoch.
- **Test Set Evaluation**: Run `predict.py` to generate predictions, analyze model accuracy, and check the quality of the submission file.

## Acknowledgments
This project leverages [PyTorch](https://pytorch.org/) and [TorchVision](https://pytorch.org/vision/) for model building and [Pandas](https://pandas.pydata.org/) for data handling. Thanks to the [Meesho Data Challenge](https://www.kaggle.com/competitions/visual-taxonomy) for providing the dataset and evaluation criteria.
