# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 20:58:55 2023

@author: Rahul
"""

import pandas as pd
import numpy as np

df = pd.read_excel("D:\\DS\\books\\ASSIGNMENTS\\Forecasting\\Airlines+Data.xlsx")
df
df.shape
df.describe()
df.info()

df.set_index('Month', inplace=True)#making the month column as index
df.head()

df.index.year

df.isnull().sum()#no null values

df[df.duplicated()].shape 
#found the 16 duplicated rows

df[df.duplicated()]

df.drop_duplicates(inplace=True)
df.shape

df1 = df.copy()

# Visulization
#lineplot
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf

plt.figure(figsize=(12,3))
sns.lineplot(x="Month",y="Passengers",data=df1)

plt.figure(figsize=(8,6))
ax = plt.axes()
ax.set_facecolor("white")
plt.plot(df1['Passengers'], color = 'black', linewidth=3)
plt.xlabel('Year')
plt.ylabel("number of passengers")
plt.show()


df1["Passengers"].hist()

#density plot
ax =plt.axes()
ax.set_facecolor('black')
df1["Passengers"].plot(kind='kde',figsize=(8,5),color='blue')

# Lagplot
from pandas.plotting import lag_plot
plt.figure(figsize=(8,5))
ax = plt.axes()
ax.set_facecolor("black")
lag_plot(df1['Passengers'])
plt.show()

plt.figure(figsize=(12,4))
df1.Passengers.plot(label="org")
for i in range(2,8,2):
    df1["Passengers"].rolling(i).mean().plot(label=str(i))
plt.legend(loc='best')

#Timeseries decomposition plot
from statsmodels.tsa.seasonal import seasonal_decompose
decompose_ts_add = seasonal_decompose(df1.Passengers,period=12)
decompose_ts_add.plot()
plt.show()

sns.boxplot(x = 'Passengers',data=df1)

#upsampling
upsampled = df1.resample('M').mean()
print(upsampled.head(32))

interpolated = upsampled.interpolate(method='linear') ## interplation was done for nan values which we get after doing upsampling by month
print(interpolated.head(15))
interpolated.plot()
plt.show()

Train = interpolated.head(81)
Test = interpolated.tail(14)

plt.figure(figsize=(12,4))
interpolated.Passengers.plot(label="org")
for i in range(2,24,6):
    interpolated["Passengers"].rolling(i).mean().plot(label=str(i))
plt.legend(loc='best')

import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(interpolated.Passengers,lags=14)
tsa_plots.plot_pacf(interpolated.Passengers,lags=14)
plt.show()

def MAPE(pred,org):
    temp = np.abs((pred-org)/org)*100
    return np.mean(temp)

#forecasting models
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from numpy import sqrt
from sklearn.metrics import mean_squared_error

#SimpleExp0nentialMethod
ses_model = SimpleExpSmoothing(Train["Passengers"]).fit(smoothing_level=0.2)
pred_ses = ses_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_ses,Test.Passengers)

#Holt method 
hw_model = Holt(Train["Passengers"]).fit(smoothing_level=0.1, smoothing_slope=0.2)
pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hw,Test.Passengers)

#Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(Train["Passengers"],seasonal="add",trend="add",seasonal_periods=12).fit(smoothing_level=0.1, smoothing_slope=0.2) #add the trend to the model
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_add_add,Test.Passengers)

#Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(Train["Passengers"],seasonal="mul",trend="add",seasonal_periods=12).fit(smoothing_level=0.1, smoothing_slope=0.2) 
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_mul_add,Test.Passengers)

rmse_hwe_mul_add = sqrt(mean_squared_error(pred_hwe_mul_add,Test.Passengers))
rmse_hwe_mul_add


#Final Model by combining train and test
hwe_model_add_add = ExponentialSmoothing(interpolated["Passengers"],seasonal="add",trend="add",seasonal_periods=10).fit()


#Forecasting for next 10 time periods
hwe_model_add_add.forecast(10)


interpolated.reset_index(inplace=True)
interpolated['t'] = 1

for i,row in interpolated.iterrows():
  interpolated['t'].iloc[i] = i+1

interpolated
interpolated['t_sq'] = (interpolated['t'])**2 # inserted t_sq column with values


interpolated["month"] = interpolated.Month.dt.strftime("%b") # month extraction
interpolated["year"] = interpolated.Month.dt.strftime("%Y") # year extraction
interpolated

months = pd.get_dummies(interpolated['month']) ## converting the dummy variables for month column
months

months = months[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']] # storing the months as serial wise again in months variable

airdata = pd.concat([interpolated,months],axis=1)
airdata.head()

airdata['log_passengers'] = np.log(airdata['Passengers'])
airdata


plt.figure(figsize=(12,8))
heatmap_y_month = pd.pivot_table(data=airdata,values="Passengers",index="year",columns="month",aggfunc="mean",fill_value=0)
sns.heatmap(heatmap_y_month,annot=True,fmt="g")


sns.boxplot(x="month",y="Passengers",data= airdata)

sns.boxplot(x="year",y="Passengers",data= airdata)

plt.figure(figsize=(12,3))
sns.lineplot(x="year",y="Passengers",data=airdata)

Train = airdata.head(81) 
Test = airdata.tail(14)

def RMSE(org, pred):
    rmse=np.sqrt(np.mean((np.array(org)-np.array(pred))**2))
    return rmse

#Linear Model
import statsmodels.formula.api as smf 

linear_model = smf.ols('Passengers~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear =RMSE(Test['Passengers'], pred_linear)
rmse_linear

#Exponential
Exp = smf.ols('log_passengers~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = RMSE(Test['Passengers'], np.exp(pred_Exp))
rmse_Exp


#Quadratic 
Quad = smf.ols('Passengers~t+t_sq',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_sq"]]))
rmse_Quad = RMSE(Test['Passengers'], pred_Quad)
rmse_Quad


#Additive seasonality 
add_sea = smf.ols('Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = RMSE(Test['Passengers'], pred_add_sea)
rmse_add_sea


#Additive Seasonality Quadratic 
add_sea_Quad = smf.ols('Passengers~t+t_sq+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_sq']]))
rmse_add_sea_quad = RMSE(Test['Passengers'], pred_add_sea_quad)
rmse_add_sea_quad


##Multiplicative Seasonality
Mul_sea = smf.ols('log_passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = RMSE(Test['Passengers'], np.exp(pred_Mult_sea))
rmse_Mult_sea


#Multiplicative Additive Seasonality 
Mul_Add_sea = smf.ols('log_passengers~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = RMSE(Test['Passengers'], np.exp(pred_Mult_add_sea))
rmse_Mult_add_sea

#Compareing the results 

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse.sort_values(['RMSE_Values'])


#rmse_mult_add_sea is prefered than any other models in this analysis


# Building the model on entire data set
#final data
df.rename(columns={'Month':'Date'},inplace = True)
df

df['year'] = df.Date.dt.strftime('%Y')
df['month'] = df.Date.dt.strftime('%b')
df['date'] = df.Date.dt.strftime('%d')
df

df['t']= np.arange(0,96)
df['t_square'] = df['t']*df['t']
df['log_passanger'] = np.log(df['Passengers'])

month_dummies = pd.get_dummies(df['month'])
month_dummies

airline = pd.concat([df,month_dummies] ,axis = 1)
airline

# choosing "Multiplicative Additive seosanality" (mult_add_sea)

t = np.arange(97,108)
t
t_square = t*t
t_square


month = pd.date_range(start='1/1/2003',end='11/1/2003',freq='MS')
month

Month = pd.DataFrame(month,columns=['Date'])
Month


df_1 = {'t':t,'t_square': t_square}
value = pd.DataFrame(df_1)
value

data = pd.concat([Month,value],axis = 1)
data


data['year'] = data.Date.dt.strftime('%Y')
data['month'] = data.Date.dt.strftime('%b')
data['day'] = data.Date.dt.strftime('%d')
data

month_dummy = pd.get_dummies(data['month'])
month_dummy


final_data = pd.concat([data,month_dummy],axis = 1)
final_data.head()


pred_final = Mul_Add_sea.predict(final_data)
pred_final

x = np.exp(pred_final)

pred_final = pd.DataFrame(x,columns=['pred_final'])
pred_final

pred_data = pd.concat([final_data,round(pred_final)],axis=1)
pred_data

#total dummies create 12 .and RMSE score of Multiplicative Additive Seasonality very good








































