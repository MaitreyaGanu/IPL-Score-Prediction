# IPL Score Prediction Using Neural Networks

This project demonstrates the use of a neural network model to predict the total score of the batting team in an Indian Premier League (IPL) match. The model uses encoded and scaled features such as venue, batting team, bowling team, batsman, and bowler. A deep neural network (DNN) architecture was employed to learn complex patterns from the dataset and predict the final score of the batting team.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Model Architecture](#model-architecture)



## Introduction

The goal of this project is to predict the total score of the batting team in an IPL match based on various features such as the venue, the batting team, the bowling team, the batsman, and the bowler. This task is framed as a regression problem, where the model learns from historical IPL match data to forecast the score of a team.

We use a **neural network** to model the relationship between the input features and the target variable (total score). The neural network architecture is designed to capture complex interactions between the features and predict the batting team's score with high accuracy. The code provided in the notebook provides an interactive interface to the user to dynamically predict the total score of any match at any venue.

## Features

- **Venue**: The location where the match is being played.
- **Batting Team**: The team that is batting in the match.
- **Bowling Team**: The team that is bowling.
- **Batsman**: Information about the batsman playing at a given time.
- **Bowler**: Information about the bowler currently bowling.
- **Total Score (Target)**: The total score of the batting team, which is the target variable for the model.

The features are encoded and scaled to ensure they are suitable for model input.

## Technologies Used

- **Python**: Programming language used for the project.
- **TensorFlow/Keras**: Deep learning framework used to build, train, and evaluate the neural network.
- **NumPy**: For numerical computations and data handling.
- **Pandas**: For data manipulation and preprocessing.
- **Matplotlib**: For plotting training results and visualizing the model's performance.
- **Scikit-learn**: For encoding categorical features and splitting the dataset into training and testing sets.
## Model Architecture

### 4.1 Model Overview

The neural network model is designed to learn complex relationships in the data and predict the total score of the batting team. The architecture comprises the following components:

1. **Input Layer**: 
   - Accepts encoded and scaled features such as venue, batting team, bowling team, batsman, and bowler.
   - Ensures that all numerical inputs have similar ranges, enabling efficient model training.

2. **Two Hidden Layers**:
   - **First Hidden Layer**: Contains **512 neurons** with a **ReLU activation function** to introduce nonlinearity and capture complex patterns in the data.
   - **Second Hidden Layer**: Contains **216 neurons**, also using the **ReLU activation function**.

3. **Output Layer**:
   - Consists of **1 neuron** with a **linear activation function**.
   - Designed for regression tasks, the output is a continuous value representing the predicted total score of the batting team.

### 4.2 Loss Function and Optimization

1. **Huber Loss Function**:
   - Combines the advantages of **Mean Squared Error (MSE)** and **Mean Absolute Error (MAE)**.
   - Provides robustness to outliers in the dataset by penalizing large errors less aggressively than MSE.

2. **Adam Optimizer**:
   - Chosen for its adaptability and efficiency in training deep learning models.
   - Ensures faster convergence and improved performance.

