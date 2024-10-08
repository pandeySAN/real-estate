# Real Estate Property Price Prediction

This project uses machine learning to predict the price of real estate properties based on a dataset provided by a real estate company. The model analyzes various features of the property such as size, location, number of rooms, etc., to make accurate price predictions.

## Project Structure

- `data.csv`: Contains the real estate data with property features and prices.
- `Dragon.joblib`: The saved machine learning model for price prediction, which can be loaded and used for predictions.
- `housing.data`: The raw data file in a specific format used for training the machine learning models.
- `housing.names`: Describes the column names and structure of the `housing.data` file.
- `model usage.ipynb`: Jupyter notebook that demonstrates how to load the trained model (`Dragon.joblib`) and use it for predicting property prices.
- `Outputs from different models`: A directory containing the outputs from various machine learning models tested during the project.
- `real estate.ipynb`: Jupyter notebook used for training the machine learning models, feature selection, data visualization, and evaluation of model performance.

## Dataset

The dataset contains the following features for each property:
- **CRIM**: Crime rate by town
- **ZN**: Proportion of residential land zoned for lots over 25,000 sq. ft.
- **INDUS**: Proportion of non-retail business acres per town
- **CHAS**: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
- **NOX**: Nitric oxide concentration (parts per 10 million)
- **RM**: Average number of rooms per dwelling
- **AGE**: Proportion of owner-occupied units built prior to 1940
- **DIS**: Weighted distances to five Boston employment centers
- **RAD**: Index of accessibility to radial highways
- **TAX**: Full-value property tax rate per $10,000
- **PTRATIO**: Pupil-teacher ratio by town
- **B**: 1000(Bk - 0.63)^2 where Bk is the proportion of Black people by town
- **LSTAT**: Percentage of lower status of the population
- **MEDV**: Median value of owner-occupied homes in $1000s (target variable)

## Model and Approach

1. **Data Preprocessing**: 
   - Data cleaning and handling of missing values.
   - Feature scaling and normalization.
   - Splitting the dataset into training and test sets.

2. **Model Training**:
   - Several models were trained and evaluated, including:
     - Linear Regression
     - Decision Tree Regression
     - Random Forest Regression
     - Gradient Boosting
   - The final model saved as `Dragon.joblib` provided the best accuracy and performance.

3. **Evaluation Metrics**:
   - Mean Absolute Error (MAE)
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)

4. **Usage**:
   - The notebook `model usage.ipynb` demonstrates how to load the saved model (`Dragon.joblib`) and use it for making predictions on new data.

## Running the Project

1. Install the required libraries:
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn
2. Open the Jupyter notebooks (real estate.ipynb and model usage.ipynb) to view the data analysis, training, and model prediction steps.

3. Load the saved model:

import joblib
model = joblib.load('Dragon.joblib')

4. Use the model to make predictions:
predicted_price = model.predict(new_data)

Results
The project successfully predicts real estate prices with reasonable accuracy. Various models were tested, and the Gradient Boosting model performed the best in terms of accuracy and error metrics.

Conclusion
This machine learning model provides a useful tool for real estate companies to predict property prices based on multiple features. It can be improved further by adding more features or tuning the model hyperparameters.

Future Improvements
Add more property features such as proximity to public transportation, neighborhood amenities, etc.
Tune hyperparameters using Grid Search or Randomized Search to improve model performance.
Implement cross-validation to ensure the robustness of the model.

Author
Sanchit Pandey
