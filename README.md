Wine Quality Prediction App
This Wine Quality Prediction App is a web-based application built using Streamlit to classify the quality of wine based on key chemical features. The app uses a Random Forest Classifier model, trained on a wine quality dataset, to predict the quality of wine. Users can input various chemical characteristics such as acidity, pH level, and alcohol content, and the app will predict the wine's quality score on a scale from 3 to 8.
Features
-User-friendly Interface: Allows users to input important chemical features of the wine such as acidity, residual sugar, chlorides, and alcohol content.
-Random Forest Classifier Model: The app uses a Random Forest Classifier model, which is known for its robustness and high performance in classification tasks.
-Real-time Predictions: After inputting the chemical features, the app instantly predicts the wine's quality score using the trained Random Forest model.
-Default Values for Inputs: If no values are provided by the user, the app uses default values based on typical chemical characteristics for wine.
How It Works
>Fixed Acidity
>Volatile Acidity
>Citric Acid
>Residual Sugar
>Chlorides
>Free Sulfur Dioxide
>Total Sulfur Dioxide
>Density
>pH
>Sulphates
>Alcohol
Upon clicking the "Predict Wine Quality" button, the app uses the trained Random Forest Classifier model to predict the wine's quality score, which ranges from 3 to 8.
The app then displays:
-Predicted Wine Quality: A number representing the predicted wine quality score based on the input chemical features.
-Model Accuracy: The accuracy of the Random Forest Classifier, calculated based on the training dataset.

How to Run the App
1.Clone this repository:
git clone https://github.com/your-username/wine-quality-prediction-app.git

2.Navigate into the project directory:
cd wine-quality-prediction-app

3.Install the required packages:
pip install -r requirements.txt

4.Run the Streamlit app:
streamlit run app.py
Open your browser at the displayed local URL to interact with the app.

5.Open your browser at the displayed local URL to interact with the app.

-->Requirements
>Python 3.x
>Streamlit
>Scikit-learn
>Numpy
>Pandas


-->Model
The Random Forest Classifier used in this app has been trained on a wine quality dataset, using the following parameters:
>n_estimators: 100 (number of decision trees)
>random_state: 42 (for reproducibility)

The model is designed to classify the quality of wine based on chemical features, which include various acidity levels, sugar content, and alcohol concentration. This app makes it easy to predict wine quality in real-time based on chemical characteristics. 
