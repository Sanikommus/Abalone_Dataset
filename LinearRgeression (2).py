

import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#Reading csv file
df = pd.read_csv("C:/Users/manta/OneDrive/Desktop/Semester/Semester 3/DATA SCIENCE 3/B21015_Lab_5_Ds3/abalone.csv")

#splitting the data in 70 and 30
[X_train, X_test] =train_test_split(df, test_size=0.3, random_state=42,shuffle=True)

#getting the x_train and x_label_train 
X_label_train = X_train["Rings"]
X_train = X_train.drop(["Rings"] , axis = "columns" )

#getting the testing data into x_test and x_label_test
X_label_test = X_test["Rings"]
X_test = X_test.drop(["Rings"] , axis = "columns" )

#defining the linear regression model
reg = LinearRegression().fit(X_train , X_label_train)

#prediciting the data for both training and testing
Y_predict_training = reg.predict(X_train)
Y_predict_testing = reg.predict(X_test)

rmse_train = math.sqrt(mean_squared_error(X_label_train , Y_predict_training))
rmse_test = math.sqrt(mean_squared_error(X_label_test , Y_predict_testing))
print("The prediction accuracy on the training data using root mean squared error is : ",rmse_train)
print("The prediction accuracy on the training data using root mean squared error is : ",rmse_test)

plt.scatter(X_label_test , Y_predict_testing)
plt.xlabel("Actual Rings")
plt.ylabel("Predicted Rings")
plt.title("Actual Rings Vs Predicted Rings on the test data")
plt.show()