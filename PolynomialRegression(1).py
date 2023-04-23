
import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
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

col = X_train.columns

m = -1

#calculating the column with mac correlation coffecient 
for i in col:
    a = pearsonr(df[str(i)],df["Rings"])
    if(m < a[0]):
        m = a[0]
        maxcorr = str(i)
        
#reshaping the data for linear regression model
X = (X_train[maxcorr].to_numpy()).reshape(-1,1)
Y = (X_label_train.to_numpy()).reshape(-1,1)
T = (X_test[maxcorr].to_numpy()).reshape(-1,1)
S = (X_label_test.to_numpy()).reshape(-1,1)

rmse_train = []
rmse_test = []
p = [2,3,4,5]

#calculating the rmse value for both train and test for various value of p
for i in p:
    #defining a nonlinear regression model
    poly_features = PolynomialFeatures(i)
    
    #training our model
    x_poly = poly_features.fit_transform(X)
    y_poly = poly_features.fit_transform(T)
    regressor = LinearRegression().fit(x_poly, Y)
    
    #prediciting the result
    Y_predict_training = regressor.predict(x_poly)
    Y_predict_testing = regressor.predict(y_poly)
    
    #calculating rmse
    rmse_train.append(math.sqrt(mean_squared_error(Y , Y_predict_training)))
    rmse_test.append(math.sqrt(mean_squared_error(S , Y_predict_testing)))
 
#plotting the bar plot
plt.bar(p,rmse_train)
plt.show()
plt.bar(p,rmse_test)
plt.show()


#plotting the best fit polynomial
min_rmse_train_index = p[rmse_train.index(min(rmse_train))]
poly_features = PolynomialFeatures(min_rmse_train_index)
x_poly = poly_features.fit_transform(X)
#y_poly = poly_features.fit_transform(T)
regressor = LinearRegression().fit(x_poly, Y)
Y_predict_training = regressor.predict(x_poly)

plt.scatter(X,Y,color = "Red")
plt.scatter(X,Y_predict_training,color = "Green")
plt.xlabel("Shell weight")
plt.ylabel("Rings")
plt.title("Best fit curve on the training data")
plt.show()


min_rmse_test_index = p[rmse_test.index((min(rmse_test)))]
poly_features = PolynomialFeatures(min_rmse_test_index)
x_poly = poly_features.fit_transform(X)
y_poly = poly_features.fit_transform(T)
regressor = LinearRegression().fit(x_poly, Y)
Y_predict_testing = regressor.predict(y_poly)
plt.scatter(S,Y_predict_testing)
plt.xlabel("Actual Number of rings")
plt.ylabel("Predicted Number of Rings")
plt.title("Actual no.of Rings Vs Predicted no.of Rings")
plt.show()
