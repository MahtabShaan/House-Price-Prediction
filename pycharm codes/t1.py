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

total_df['TotalSF'] = total_df['TotalBsmtSF'] + total_df['1stFlrSF'] + total_df['2ndFlrSF']

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
from sklearn.metrics import r2_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll.base import scope
from hyperopt.pyll.stochastic import sample

scale = StandardScaler()
X_train = scale.fit_transform(new_train_df)
X_test = scale.fit_transform(new_test_df)

from sklearn.model_selection import train_test_split

X_train1, X_val, y_train1, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

n_folds = 2

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42)
    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

space = {'choice': hp.choice('num_layers',
                    [ {'layers':'two', },
                    {'layers':'three',
                    'units3': sample(scope.int(hp.uniform('units3', 64,1024))),
                    'dropout3': hp.uniform('dropout3', .25,.75),
                     'activation3': hp.choice('activation3',['relu', 'softmax', 'sigmoid'])}
                    ]),

            'units1': sample(scope.int(hp.uniform('units1', 64,1024))),
            'units2': sample(scope.int(hp.uniform('units2', 64,1024))),

            'dropout1': hp.uniform('dropout1', .25,.75),
            'dropout2': hp.uniform('dropout2',  .25,.75),

            'batch_size' : sample(scope.int(hp.uniform('batch_size', 28,128))),

            'nb_epochs' :  100,
            'optimizer': hp.choice('optimizer',['Adadelta','Adam','RMSprop','SGD']),
            'activation1': hp.choice('activation1',['relu','softmax','sigmoid']),
            'activation2': hp.choice('activation2',['relu','softmax','sigmoid'])
        }

#np.random.seed(10)

#create Model
#define base model
def f_nn(params):
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.optimizers import Adadelta, Adam, rmsprop

    print ('Params testing: ', params)
    model = Sequential()
    model.add(Dense(output_dim=params['units1'], input_dim = 220))
    model.add(Activation(params['activation1']))
    model.add(Dropout(params['dropout1']))

    model.add(Dense(output_dim=params['units2'], init = "uniform"))
    model.add(Activation(params['activation2']))
    model.add(Dropout(params['dropout2']))

    if params['choice']['layers']== 'three':
        model.add(Dense(output_dim=params['choice']['units3'], init = "uniform"))
        model.add(Activation(params['choice']['activation3']))
        model.add(Dropout(params['choice']['dropout3']))

    model.add(Dense(1))
    model.add(Activation('relu'))
    model.compile(loss='mean_squared_error', optimizer=params['optimizer'])

    clf = KerasRegressor(build_fn=model, nb_epoch=100, batch_size=128, verbose=0)

    score = np.sqrt(-cross_val_score(clf, X_train, y_train, scoring="neg_mean_squared_error", cv = 5))


    return {'loss': score, 'status': STATUS_OK}


trials = Trials()
best = fmin(f_nn, space, algo=tpe.suggest, max_evals=50, trials=trials)
print('best: ')
print(best)