# Abalone_Dataset


Given a data file abalone.csv. Abalones are marine snails. The dataset has been prepared
with the aim of making age predictions easier. Customarily, the age of abalone is determined by
cutting the shell through the cone, staining it, and counting the number of rings through a microscope.
But it is a tedious and time-consuming task. Therefore, other measurements, which are easier to
obtain, are used to predict age.

The code:

* Loads the dataset into the Spyder Enviornment.
* Applys Linear Regression between the traget attribute and the attribute wihich has the highest pearson cofficient with the target attribute(Fitting a Straight line Between two attributes).
* Applys Linear Regression between the target attribute and all the other attributes(Fitting a straight line in higher Dimensions.)
* Applys Polynomial Regression between the target attribute and the attribute which has the highest pearson cofficient with the target attribute(Fitting a polynomial curve between two attibutes).
* Applys a Polynomial Regression between the target attribute and all the other attributes(Fitting a polynomial curve in the higher dimension).


# Input Dataset

https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset

![image](https://user-images.githubusercontent.com/119813195/228891338-29c5ce1b-09db-4c18-a20f-ef319679d7e5.png)

# Output

Linear Regression between target attribute and the attribute with highest correlation :

![image](https://user-images.githubusercontent.com/119813195/228892170-803c8d83-e5e2-46e8-86dc-84dfaf082ce5.png)

Linear Regression between target attribute and all other attributes :

The best fit line is in higher dimensions so can't be plotted,

![image](https://user-images.githubusercontent.com/119813195/228892504-dbd6f595-ba4f-4cbb-accf-476fd28b3c08.png)

Polynomial Regression between target attribute and the attribute with highest correlation :

![image](https://user-images.githubusercontent.com/119813195/228893242-5ec73c08-961b-4514-b7ff-093921128c6a.png)

Polynomial Regression between target attribute and all other attributes :

The best fit curve can't be plotted as they are in higher dimensions.

![image](https://user-images.githubusercontent.com/119813195/228893774-3ab81e0c-19c3-4cae-9468-4562023f1461.png)



