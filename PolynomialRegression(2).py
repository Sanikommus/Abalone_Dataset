
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

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

rmse_train = []
rmse_test = []
p = [2,3,4,5]

#calculating the rmse value for both train and test for various value of p
for i in p:
    #defining a non linear regression model
    poly_features = PolynomialFeatures(i)
    
    #training our model
    x_poly = poly_features.fit_transform(X_train)
    y_poly = poly_features.fit_transform(X_test)
    regressor = LinearRegression().fit(x_poly, X_label_train)
    
    #predicting the result
    Y_predict_training = regressor.predict(x_poly)
    Y_predict_testing = regressor.predict(y_poly)
    
    #calculate rmse
    rmse_train.append(math.sqrt(mean_squared_error(X_label_train , Y_predict_training)))
    rmse_test.append(math.sqrt(mean_squared_error(X_label_test , Y_predict_testing)))
  
#plotting the bar graph
plt.bar(p,rmse_train)
plt.show()
plt.bar(p,rmse_test)
plt.show()

#plotting the first bit polynomial predicted data
min_rmse_test_index = p[rmse_test.index((min(rmse_test)))]
poly_features = PolynomialFeatures(min_rmse_test_index)
x_poly = poly_features.fit_transform(X_train)
y_poly = poly_features.fit_transform(X_test)
regressor = LinearRegression().fit(x_poly, X_label_train)
Y_predict_testing = regressor.predict(y_poly)
plt.scatter(X_label_test,Y_predict_testing)
plt.xlabel("Actual Number of rings")
plt.ylabel("Predicted Number of Rings")
plt.title("Actual no.of Rings Vs Predicted no.of Rings")
plt.show()