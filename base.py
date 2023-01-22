#rm -rf .git remove git from local

import pandas as pd #for data manipulation
import numpy as np # numerical computations
import seaborn as sns #visualizations
data = pd.read_csv("Book1.csv") #getting data 

x =data[["GRE Score","TOEFL Score","University Rating","CGPA"]] #independent
y = data[["chance"]] #dependent


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.30,random_state=1)
#building the model
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
ypred=lr.predict(x_test)

import pickle 
pickle.dump(lr,open('linear.pkl','wb'))