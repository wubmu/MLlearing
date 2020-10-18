# Python
import pandas as pd
import numpy as np
from fbprophet import Prophet
import matplotlib.pyplot as plt
# 读入数据集
df = pd.read_csv('D:\\code\\pyproject\\MLlearing\\pytorchDemo\\time_sequence_prediction\\examples\\example_wp_log_peyton_manning.csv')


playoffs = pd.DataFrame({
  'holiday': 'playoff',
  'ds': pd.to_datetime(['2008-01-13', '2009-01-03', '2010-01-16',
                        '2010-01-24', '2010-02-07', '2011-01-08',
                        '2013-01-12', '2014-01-12', '2014-01-19',
                        '2014-02-02', '2015-01-11', '2016-01-17',
                        '2016-01-24', '2016-02-07']),
  'lower_window': 0,
  'upper_window': 1,
})

superbowls = pd.DataFrame({
  'holiday': 'superbowl',
  'ds': pd.to_datetime(['2010-02-07', '2014-02-02', '2016-02-07']),
  'lower_window': 0,
  'upper_window': 1,
})
holidays = pd.concat((playoffs, superbowls))#季后赛和超级碗比赛特别日期
# 拟合模型
m = Prophet(holidays=holidays)
m.fit(df)

# 构建待预测日期数据框，periods = 365 代表除历史数据的日期外再往后推 365 天
future = m.make_future_dataframe(periods=365)

# 预测数据集
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

# 展示预测结果
m.plot(forecast)

# 预测的成分分析绘图，展示预测中的趋势、周效应和年度效应
m.plot_components(forecast)
plt.show()