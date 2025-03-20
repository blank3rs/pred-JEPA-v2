# Healthcare Prediction System with JEPA

This system uses a Joint Embedding Predictive Architecture (JEPA) to learn from healthcare data and make predictions for new patients.

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Ensure your healthcare_dataset.csv file is in the root directory.

## Training the Model

To train the model on your healthcare data:

```
python healthcare_predictor.py
```

This will:
- Load and preprocess the healthcare data
- Convert tabular data to text form for the JEPA model
- Train the model for 3 epochs
- Save the best model as 'healthcare_jepa_model.pt'

## Making Predictions

To make predictions for new patients:

```
python predict_patient.py
```

Follow the interactive prompts to:
1. Enter basic patient information (name, age, gender, etc.)
2. Skip fields you want the model to predict
3. View prediction results based on similar patients

## How It Works

This system:
1. Uses a JEPA model to learn healthcare patterns through self-supervised learning
2. Converts structured healthcare data to text for natural language processing
3. Uses masked token prediction to understand relationships between fields
4. Finds similar patients to make predictions for missing information

## Architecture

- `jepa_encoder.py`: Core JEPA model implementation
- `healthcare_predictor.py`: Adapter for healthcare data and training
- `predict_patient.py`: Interactive prediction interface
- `latent_space.py`: Visualization utilities

## Note

For best results, train with a large and diverse healthcare dataset. The model performance improves with more training data. 