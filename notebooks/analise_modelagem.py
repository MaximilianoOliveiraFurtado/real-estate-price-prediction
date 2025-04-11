# regressao_imoveis.ipynb (convertido para script Python)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

df = pd.read_csv("./data/train.csv")

df = df.drop(columns=["Alley", "PoolQC", "Fence", "MiscFeature"], errors='ignore')
df = df.dropna()

features = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea',
            'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
df_model = df[features + ['SalePrice']]

X = df_model.drop('SalePrice', axis=1)
y = df_model['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("Regressão Linear")
print("R²:", r2_score(y_test, y_pred_lr))
print("RMSE:", mean_squared_error(y_test, y_pred_lr, squared=False))

X_train_sm = sm.add_constant(X_train)
ols_model = sm.OLS(y_train, X_train_sm).fit()
print(ols_model.summary())

dt = DecisionTreeRegressor(max_depth=5)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print("\nÁrvore de Regressão")
print("R²:", r2_score(y_test, y_pred_dt))
print("RMSE:", mean_squared_error(y_test, y_pred_dt, squared=False))

plt.figure(figsize=(8, 5))
sns.kdeplot(y_test, label='Real', linewidth=2)
sns.kdeplot(y_pred_lr, label='Regressão Linear', linestyle="--")
sns.kdeplot(y_pred_dt, label='Árvore de Regressão', linestyle=":")
plt.title("Comparação de Distribuições: Real vs Predito")
plt.legend()
plt.show()
