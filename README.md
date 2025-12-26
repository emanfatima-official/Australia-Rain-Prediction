# Rain Tomorrow Prediction (Australia Weather)

This project builds a machine learning classification model to predict whether it will rain **tomorrow** using historical Australian weather observations. The model is trained on **10 years of observed meteorological data** collected from multiple weather stations across Australia.
## Dataset Information:
* **Source:** Bureau of Meteorology (BOM), Australia
* **Type:** Observed (real measured) weather data
* **Time Span:** ~10 years of daily records
* **Target Variable:** `RainTomorrow`
  * `1` → Rain tomorrow
  * `0` → No rain tomorrow

Key features include humidity, pressure, cloud cover, wind direction/speed, temperature, and rainfall indicators.
## 🧠 Model Description:
* **Model Type:** Machine Learning Classification
* **Algorithm:** XGBoost Classifier
* **Problem Type:** Binary Classification
* **Output:**
  * RainTomorrow (Yes / No)
  * Probability of rain

The model learns patterns from weather conditions such as **Humidity3pm, Pressure, Cloud Cover, Wind Direction, and Max Temperature** to make predictions.
## Workflow:
1. Load dataset using `kagglehub`
2. Data cleaning & preprocessing
3. Encode categorical variables
4. Train-test split
5. Model training (XGBoost)
6. Model evaluation
7. Prediction on unseen data
8. Save trained model
## Model Saving:
The trained model is saved using `joblib`:

```python
import joblib
joblib.dump(model, "rain_tomorrow_model.joblib")
```

Load the model later using:

```python
model = joblib.load("rain_tomorrow_model.joblib")
```

## Example Prediction:

```python
prediction = model.predict(sample)[0]
probability = model.predict_proba(sample)[0, 1]
```

**Output:**

* `RainTomorrow = No`
* Probability of rain = XX%

---

## Use Cases:

* Educational ML projects
* Weather trend analysis
* Decision support systems
* Real-time weather prediction (with API integration)

## Technologies Used:

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* KaggleHub
* Jupyter Notebook

##  Disclaimer:

This model predicts rain based on historical patterns and **does not represent official weather forecasts**. Real-time accuracy depends on the quality of input data.

Bas bolo 😊
