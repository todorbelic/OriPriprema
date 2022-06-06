# TODO 3: implementirati primenu jednostavne linearne regresije
# nad podacima iz datoteke "data/skincancer.csv".
import pandas as pd
data = pd.read_csv('../../customer_churn.csv')

import linreg_simple as ls
import matplotlib.pyplot as plt

data['churn'] = data['churn'].astype(dtype=float)
X  = data['total intl minutes'].values
Y  = data['churn'].values

slope, intercept = ls.linear_regression(X, Y)

line_y = ls.create_line(X, slope, intercept)

plt.plot(X, Y, '.')
plt.plot(X, line_y, 'b')
plt.title('Slope: {0}, intercept: {1}'.format(slope, intercept))
print('verovatnoca da napusti sa pet minuta {}'.format(ls.predict(5, slope, intercept) * 100))
print('verovatnoca da napusti sa 60 minuta {}'.format(ls.predict(60, slope, intercept) * 100))
plt.show()