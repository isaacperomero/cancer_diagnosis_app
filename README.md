# Breast Cancer Prediction App

![screenshot](https://github.com/isaacperomero/cancer_diagnosis_app/blob/main/screenshot_app.png)

### Dataset
Dataset can be found on (https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data).
- Refer to the file `data.csv`

#### Attribute Information:
1.	ID number
2.	Diagnosis (M = malignant, B = benign)

* (3 – 32)
Ten real-valued features are computed for each cell nucleus:
1. radius (mean of distances from center to points on the perimeter)
2. texture (standard deviation of gray-scale values)
3. perimeter
4. area
5. smoothness (local variation in radius lengths)
6. compactness (perimeter^2 / area - 1.0)
7. concavity (severity of concave portions of the contour)
8. concave points (number of concave portions of the contour)
9. symmetry
10. fractal dimension ("coastline approximation" - 1)

* The mean, standard error and "worst" or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features. For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.
* All feature values are recoded with four significant digits.

#### Missing attribute values: none
#### Class distribution: 357 benign, 212 malignant

## Model Building 

Refer to the file `model.py`

It imports the necessary dependencies, including pandas for data manipulation, scikit-learn for data preprocessing and modeling, and pickle for serializing the trained model and scaler.
1. The `create_model` function is defined, which takes the input data and performs the following steps:
- Separates the independent variables (features) and the dependent variable (diagnosis).
- Applies feature scaling using StandardScaler to standardize the feature values.
- Splits the data into training and testing sets using the train_test_split function.
- Creates a logistic regression model using LogisticRegression from scikit-learn.
- Trains the model using the training data.
- Makes predictions on the test data and evaluates the model's performance by calculating the accuracy and generating a classification report.
2. The `clean_data` function is defined, which performs some basic data cleaning steps:
- Reads the breast cancer dataset from a CSV file.
- Removes unnecessary columns (Unnamed: 32 and id) from the dataset.
- Maps the diagnosis column values (M for malignant, B for benign) to binary labels (1 for malignant, 0 for benign).
3. The `main` function is defined, which serves as the entry point of the script:
- Calls the `clean_data` function to obtain the cleaned dataset.
- Calls the `create_model` function, passing the cleaned dataset.
- Serializes the trained model and scaler objects using pickle and saves them as files ('model.pkl' and 'scaler.pkl').
- Finally, the main function is called when the script is executed.

## Streamlit App 

Refer to the file `app.py`

It imports the necessary dependencies, including Streamlit for creating the web application, pickle for loading the trained model and scaler, pandas for data manipulation, NumPy for numerical operations, and Plotly for creating radar charts.

- The `get_clean_data` function reads the breast cancer dataset from a CSV file, performs data cleaning by dropping unnecessary columns, and maps the diagnosis column values to binary labels (the same function in the modelling step).
- The `add_sidebar` function creates a sidebar in the Streamlit application, which displays sliders for user input of cell nuclei measurements. The function retrieves the cleaned data and defines a list of slider labels and keys for each measurement.
- The `get_scaled_values` function scales the user input values based on the range of the corresponding measurement in the original dataset. It uses the cleaned dataset and the input dictionary to calculate scaled values for each measurement.
- The `get_radar_chart` function creates a radar chart using Plotly. It takes the scaled input data, defines categories for the chart, and plots three radar traces representing different types of measurements: mean, standard error, and worst.
- The `add_prediction` function loads the trained model and scaler using pickle. It transforms the scaled user input into the appropriate format and makes a prediction using the logistic regression model. The function displays the predicted diagnosis (benign or malignant) and the probabilities associated with each class.
- The `main` function sets the configuration of the Streamlit application, including the page title, icon, layout, and initial sidebar state. It also includes custom CSS styling for the application. Inside the main function, the user input data is collected using the `add_sidebar` function, and a container is created to display the application title and description.
- The application layout is divided into two columns. In the first column, the radar chart is displayed based on the user input data, using the `get_radar_chart` function. In the second column, the diagnosis prediction and associated probabilities are shown using the `add_prediction` function.
- Finally, the `main` function is called to run the Streamlit application.

Note: The block of CSS code in `style.css` defines some custom styles that are used in the Streamlit application for breast cancer prediction.

### [App Link](https://cancerdiagnosisapp-isaacpero.streamlit.app/)
