# %% [markdown]
# # Price prediction
# 
# Análise exploratória, modelagem e avaliação com dados do Kaggle.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

# %% [markdown]
# ## Carregamento e visualização inicial dos dados

# %%
df = pd.read_csv("../data/train.csv")
df.head()

# %%
df.describe()

# %%
df = df.drop(columns=["Alley", "PoolQC", "Fence", "MiscFeature"], errors='ignore')
df = df.dropna()
df = pd.get_dummies(df, columns=["Neighborhood"], drop_first=True)

# %% [markdown]
# ## Visualização da distribuição dos preços

# %%
plt.figure(figsize=(8, 5))
sns.histplot(df['SalePrice'], kde=True)
plt.title("Distribuição dos Preços")
plt.xlabel("Preço")
plt.ylabel("Frequência")
plt.show()

# %% [markdown]
# ## Correlação com a variável alvo

# %%
corr = df.corr(numeric_only=True)
plt.figure(figsize=(12, 8))
sns.heatmap(corr[['SalePrice']].sort_values(by='SalePrice', ascending=False), annot=True)
plt.title("Correlação com SalePrice")
plt.show()

# %% [markdown]
# ## Seleção de variáveis e preparação para modelagem

# %%
bairros = [col for col in df.columns if col.startswith("Neighborhood_")]
features = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea',
            'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt'] + bairros
df_model = df[features + ['SalePrice']]
X = df_model.drop('SalePrice', axis=1)
y = df_model['SalePrice']

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [markdown]
# ## Modelo 1: Regressão Linear

# %%
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("R²:", r2_score(y_test, y_pred_lr))
print("RMSE:", mean_squared_error(y_test, y_pred_lr, squared=False))

# %%
X_train_sm = sm.add_constant(X_train)
X_train_sm = X_train_sm.astype(float)
ols_model = sm.OLS(y_train, X_train_sm).fit()
ols_model.summary()

# %% [markdown]
# ## Modelo 2: Árvore de Regressão

# %%
dt = DecisionTreeRegressor(max_depth=5)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print("R²:", r2_score(y_test, y_pred_dt))
print("RMSE:", mean_squared_error(y_test, y_pred_dt, squared=False))

# %% [markdown]
# ## Comparação visual

# %%
plt.figure(figsize=(8, 5))
sns.kdeplot(y_test, label='Real', linewidth=2)
sns.kdeplot(y_pred_lr, label='Regressão Linear', linestyle="--")
sns.kdeplot(y_pred_dt, label='Árvore de Regressão', linestyle=":")
plt.title("Comparação de Distribuições: Real vs Predito")
plt.legend()
plt.show()


