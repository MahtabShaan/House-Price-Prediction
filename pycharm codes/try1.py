import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import warnings
warnings.simplefilter('ignore')
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm, skew
import sys
sys.path.append('C:\\Users\\Mahtab Noor Shaan\\Anaconda3\\Lib\\site-packages')

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

train_ID = train_df['Id']
test_ID = test_df['Id']

train_df.drop("Id", axis = 1, inplace = True)
test_df.drop("Id", axis = 1, inplace = True)

print("The train data shape after dropping Id feature is : {} ".format(train_df.shape))
print("The test data shape after dropping Id feature is : {} ".format(test_df.shape))

fig, ax = plt.subplots()
ax.scatter(x = train_df['GrLivArea'], y = train_df['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()

train_df = train_df.drop(train_df[(train_df['GrLivArea']>4000) & (train_df['SalePrice']<300000)].index)

sns.distplot(train_df['SalePrice'])
plt.show()

train_df["SalePrice"] = np.log1p(train_df["SalePrice"])
sns.distplot(train_df['SalePrice'])
plt.show()

corrmat = train_df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.9, square=True)
plt.show()

ntrain = train_df.shape[0]
ntest = test_df.shape[0]
y_train = train_df.SalePrice.values
total_df = pd.concat((train_df, test_df)).reset_index(drop=True)
total_df.drop(['SalePrice'], axis=1, inplace=True)
print("total data shape is : {}".format(total_df.shape))
total_df.info()

total_df_na = (total_df.isnull().sum() / len(total_df)) * 100
total_df_na = total_df_na.drop(total_df_na[total_df_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :total_df_na})
missing_data

for col in ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'):
    total_df[col] = total_df[col].fillna('None')

total_df["LotFrontage"] = total_df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    total_df[col] = total_df[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    total_df[col] = total_df[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    total_df[col] = total_df[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    total_df[col] = total_df[col].fillna('None')
total_df["MasVnrType"] = total_df["MasVnrType"].fillna("None")
total_df["MasVnrArea"] = total_df["MasVnrArea"].fillna(0)

total_df['MSZoning'] = total_df['MSZoning'].fillna(total_df['MSZoning'].mode()[0])
total_df = total_df.drop(['Utilities'], axis=1)
total_df["Functional"] = total_df["Functional"].fillna("Typ")
total_df['Electrical'] = total_df['Electrical'].fillna(total_df['Electrical'].mode()[0])
total_df['KitchenQual'] = total_df['KitchenQual'].fillna(total_df['KitchenQual'].mode()[0])
total_df['Exterior1st'] = total_df['Exterior1st'].fillna(total_df['Exterior1st'].mode()[0])
total_df['Exterior2nd'] = total_df['Exterior2nd'].fillna(total_df['Exterior2nd'].mode()[0])
total_df['SaleType'] = total_df['SaleType'].fillna(total_df['SaleType'].mode()[0])

for col in ('MSSubClass', 'OverallCond', 'YrSold', 'MoSold'):
    total_df[col] = total_df[col].astype(str)

from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
        'YrSold', 'MoSold')

for c in cols:
    lbl = LabelEncoder()
    total_df[c] = lbl.fit_transform(total_df[c])

print('Shape of total_df: {}'.format(total_df.shape))

numeric_feats = total_df.dtypes[total_df.dtypes != "object"].index

skewed_feats = total_df[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)

skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p

skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    total_df[feat] = boxcox1p(total_df[feat], lam)

# total_df[skewed_features] = np.log1p(total_df[skewed_features])
total_df = pd.get_dummies(total_df)
print(total_df.shape)

new_train_df = total_df[:ntrain]
new_test_df = total_df[ntrain:]

NN = MLPRegressor()
activation_options = ['identity', 'logistic', 'tanh', 'relu']
solver_options = ['lbfgs', 'sgd', 'adam']
learning_rate_options = ['constant', 'invscaling', 'adaptive']
alpha = [0.001, 0.01, 0.1, 0.5]
#hidden_layer = [(i, j, k) for (i, j, k) in zip((1, 100, 20), (1, 100, 20), (1, 100, 20))]
param_gridNN = dict(activation=activation_options,
                    solver=solver_options,
                    learning_rate=learning_rate_options,
                    alpha = alpha)
gridNN = GridSearchCV(NN, param_gridNN, cv=5,
                      scoring='neg_mean_squared_error')
gridNN.fit(new_train_df.values, y_train)

Best = gridNN
BestScore = gridNN.best_score_

print("NN Score " + str(gridNN.best_score_))
print("NN  best Params " + str(gridNN.best_params_))