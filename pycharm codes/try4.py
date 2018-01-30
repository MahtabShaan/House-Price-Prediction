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

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import SGD
from sklearn.preprocessing import StandardScaler

np.random.seed(10)

#create Model
#define base model
def base_model(init_mode='uniform', activation='sigmoid', dropout_rate=0.1):
    model = Sequential()
    model.add(Dense(150, input_dim=219, init=init_mode, activation=activation))
    model.add(Dropout(dropout_rate))
    #model.add(Dense(50, init=init_mode, activation=activation))
    #model.add(Dropout(dropout_rate))
    model.add(Dense(1, init=init_mode, activation='relu'))

    optimizer = SGD(lr=0.001, momentum=0.6)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model


seed = 7
np.random.seed(seed)

scale = StandardScaler()
X_train = scale.fit_transform(new_train_df)
X_test = scale.fit_transform(new_test_df)

clf = KerasRegressor(build_fn=base_model, epochs=100, batch_size=60, verbose=2)
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42)
    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

score = rmsle_cv(clf)
print("\nMLPRegressor score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))