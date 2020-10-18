import pandas as pd
import numpy as np
from fbprophet import Prophet
import matplotlib.pyplot as plt

df = pd.read_csv('D:\\code\\pyproject\\MLlearing\\pytorchDemo\\time_sequence_prediction\\examples\\example_air_passengers.csv')
df['y'] = np.log(df['y'])
df.head()
m = Prophet(growth="linear", n_changepoints=0,
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False)
m.fit(df)
future = m.make_future_dataframe(periods=4)
future.tail()
forecast = m.predict(future)

m.plot(forecast)

# x1 = forecast['ds']
# y1 = forecast['yhat']


plt.plot(x1,y1)
plt.show()