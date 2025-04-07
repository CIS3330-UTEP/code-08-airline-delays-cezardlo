import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

filename = 'Flight_Delays_2018.csv'
df=pd.read_csv(filename)
df.describe()
print(df.head())

top_airlines = df['OP_CARRIER_NAME'].value_counts().head(5)
print(top_airlines)

Y = df['ARR_DELAY']
X = df[['DEP_DELAY', 'CARRIER_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY']]
X = sm.add_constant(X)  # Adds the intercept to the regression
model = sm.OLS(Y, X).fit()  # Fit the OLS regression model
print(model.params)  # Show coefficients for each variable including intercept
fig, ax = plt.subplots()
fig = sm.graphics.plot_fit(model, 1, ax=ax)  
plt.title("OLS Fit: DEP_DELAY vs ARR_DELAY")
plt.xlabel("Departure Delay")
plt.ylabel("Arrival Delay")
plt.show()