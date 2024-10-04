# GRU Model for Air Passenger Sequence Prediction

This project implements a Gated Recurrent Unit (GRU) model to predict the number of air passengers based on the historical data from the **AirPassengers** dataset. The project is structured to walk through the steps of loading and preprocessing the dataset, building and training the GRU model, and visualizing the prediction results.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Model Architecture](#model-architecture)
6. [Results](#results)
7. [Future Improvements](#future-improvements)

## Project Overview

Time series forecasting plays a crucial role in various industries where predicting future values is necessary for decision making. In this project, we use a GRU-based deep learning model to predict air passenger counts using historical data. GRU is an advanced version of RNN (Recurrent Neural Networks) designed to handle long-term dependencies more efficiently by mitigating the vanishing gradient problem.

## Dataset

The dataset used in this project is the **AirPassengers** dataset, which contains monthly totals of international airline passengers from 1949 to 1960. It has the following structure:

- **Month**: The month of the record.
- **Passengers**: The total number of passengers for the given month.

The dataset can be found in the file `AirPassengers.csv`.

## Installation

To get started, clone this repository and set up the required dependencies.

### 1. Clone the repository:
```bash
git clone https://github.com/ahmdmohamedd/GRU-sequence-prediction.git
cd GRU-sequence-prediction
```

### 2. Set up a virtual environment (optional but recommended):
```bash
# Install virtualenv if not already installed
pip install virtualenv

# Create a virtual environment
virtualenv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On MacOS/Linux:
source venv/bin/activate
```

### Dependencies:

- Python 3.x
- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## Usage

Once you've set up the environment and installed the dependencies, you can run the project using the following steps:

### 1. Data Preprocessing
The `AirPassengers.csv` file is loaded, and the following steps are performed:
- The **Month** column is parsed as a `datetime` object.
- The **Passengers** column is normalized using `MinMaxScaler` to bring values into the range `[0, 1]`.

### 2. Model Training
The GRU model is constructed and trained on 80% of the dataset. The remaining 20% is used for testing and validating the model's predictive performance.


### 3. Visualization
The predicted values are compared to the actual passenger counts and plotted using Matplotlib.

## Model Architecture

The GRU model consists of the following layers:
- **GRU Layers**: Two GRU layers with 100 units each.
- **Dropout Layers**: Dropout layers with a rate of 0.2 to prevent overfitting.
- **Dense Layer**: A Dense layer with 1 unit to output the final prediction.

The model is trained using the `Adam` optimizer and the `mean_squared_error` loss function.

```python
model = Sequential([
    GRU(100, return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
    Dropout(0.2),
    GRU(100, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])
```

## Results

After training, the model's performance is visualized by plotting the predicted passenger counts against the actual values. While the model captures the general trend of the data, some improvements can still be made in predicting sharp increases and decreases in passenger counts.

## Future Improvements

- **Hyperparameter Tuning**: Experiment with the number of GRU units, dropout rates, and sequence length to improve model performance.
- **Additional Features**: Incorporate additional time-based features like year, month, or seasonal effects.
- **Different Architectures**: Test with more advanced architectures such as LSTM or Transformer-based models.
- **Handling Seasonality**: Apply techniques like seasonal decomposition to capture periodic trends more effectively.
