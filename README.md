# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
Date:
### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
Import necessary libraries (NumPy, Matplotlib)

Load the dataset

Calculate the linear trend values using least square method

Calculate the polynomial trend values using least square method

End the program
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load the data from the CSV file
file_path = '/content/NFLX.csv'
data = pd.read_csv(file_path)

# Convert the 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Set the 'Date' column as the index
data.set_index('Date', inplace=True)

# Use only numerical index for fitting models
data.reset_index(inplace=True)

# Extract the features and target
X = np.arange(len(data)).reshape(-1, 1)  # Days as feature
y = data['Open'].values  # Open prices as the target

# Linear Trend
linear_model = LinearRegression()
linear_model.fit(X, y)
y_linear_pred = linear_model.predict(X)

# Polynomial Trend (degree 2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_poly_pred = poly_model.predict(X_poly)

# Plotting the actual data and trends
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], y, label='Actual Data')
plt.plot(data['Date'], y_linear_pred, label='Linear Trend', linestyle='--')
plt.plot(data['Date'], y_poly_pred, label='Polynomial Trend (Degree 2)', linestyle='-.')
plt.title('Netflix Stock Price with Linear and Polynomial Trends')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.legend()
plt.show()

# Print the linear and polynomial trend equations
print(f"Linear Trend Equation: y = {linear_model.coef_[0]:.2f} * x + {linear_model.intercept_:.2f}")
print("Polynomial Trend Equation (Degree 2): y = {:.2f} * x^2 + {:.2f} * x + {:.2f}".format(
    poly_model.coef_[2], poly_model.coef_[1], poly_model.intercept_))
```

### OUTPUT
![image](https://github.com/user-attachments/assets/2713d0ff-488d-4a5f-9c51-5656cb032c2d)

A - LINEAR TREND ESTIMATION : y = 0.09 * x + -121.90

B- POLYNOMIAL TREND ESTIMATION : y = 0.00 * x^2 + -0.12 * x + 59.29

### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
