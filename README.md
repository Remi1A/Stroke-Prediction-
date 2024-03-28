# Stroke-Prediction-
This Python script is a comprehensive tool designed to predict stroke risk using health data. Initially, it imports essential libraries for data manipulation (NumPy, pandas), visualization (Matplotlib, seaborn), statistical analysis (SciPy), machine learning (Scikit-learn), and creating graphical user interfaces (Tkinter). The core functionality is encapsulated within several classes:

Dataset: Handles data loading from a CSV file, cleaning (removing unspecified gender entries, handling missing BMI values, and filtering unknown smoking statuses), and visualization of correlation matrices. It also preprocesses the data by encoding categorical features and splitting the dataset into training and testing sets.

Model: Utilizes the preprocessed data from the Dataset class to train a K-Neighbors Classifier. It finds the optimal number of neighbors (k) that yields the highest accuracy in predicting stroke risk. The class can calibrate the model based on the best k value and make predictions on new data.

Visual: Provides visualization of the model's performance, particularly how the accuracy varies with different values of k.

StrokePredictionGUI: Creates a user-friendly graphical interface allowing users to input health-related parameters and receive a stroke risk prediction. It interacts with the Model class to process user inputs and display the prediction results.

The script follows a procedural approach to utilize these classes. It starts by creating a Dataset object to load and clean the data, then visualizes the data's correlation matrix and checks the normality of certain features. It proceeds to preprocess the data and initializes a Model object to train and evaluate the K-Neighbors Classifier. Visualization of the model's accuracy across different k values is done before finally initializing the StrokePredictionGUI to interact with the user.

This structured approach demonstrates the integration of data preprocessing, machine learning modeling, performance visualization, and user interface development within a single Python script. It showcases the script's capability to provide end-to-end functionality from raw data handling to actionable stroke risk predictions through a user-friendly graphical interface.
