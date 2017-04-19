# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 15:59:18 2017

@author: padma
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import preprocessing
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostRegressor




df = pd.read_csv('train.csv')


print(df.dtypes)

d_type = df.dtypes



cols = list(df)

cols = df.columns

print(cols)


def fill_string_nans(df) :

         
    df.MSZoning = df.MSZoning.map({'C (all)' : 1, 'FV' : 2 , 'RL':3, 'RM' : 4, 'RH' : 5})

    df.Street = df.Street.map({'Grvl' : 1, 'Pave' : 2})

    df.Alley = df.Street.map({'Grvl' : 1, 'Pave' : 2, 'NA' : 3})
    
    df.LotShape = df.LotShape.map({'IR1' : 1, 'IR2' : 2, 'IR3' : 3, 'Reg' : 4})
    
    df.LandContour = df.LandContour.map({'Lvl' : 1, 'Bnk' : 2, 'HLS' : 3, 'Low' : 4})
    
    df.Utilities = df.Utilities.map({'AllPub' : 1, 'NoSeWa' : 2})

    df.LotConfig = df.LotConfig.map({'Corner' : 1, 'CulDSac' : 2, 'FR2' : 3, 'FR3' : 4, 'Inside' : 5})

    df.LandSlope = df.LandSlope.map({'Gtl' : 1, 'Mod' : 2, 'Sev' : 3})

##    df.Neighborhood = df.Neighborhood.map({'Blmngtn' : 1, 
##                                            'Blueste' : 2, 
##                                            'BrDale' : 3, 
##                                            'BrkSide' : 4,
##                                            'ClearCr' : 5,
##                                            'CollgCr' : 6,
## #                                           'Crawfor' : 7,
##                                            'Edwards' : 8,
## #                                           'Gilbert' : 9,
##                                            'IDOTRR' : 10,
##                                            'MeadowV' : 11,
##                                            'Mitchel' : 12,                                            'NAmes' : 13,
##                                            'NoRidge' : 14,
##                                            'NPkVill' : 15,
##                                            'NridgHt' : 16,
##                                            'NWAmes' : 17,
##                                            'OldTown' : 18,
##                                           'Sawyer' : 19,
##                                            'SawyerW' : 20,
##                                            'Somerst' : 21,
##                                            'StoneBr' : 22,
##                                            'SWISU'  :23,
##                                            'Timber' : 24,
##                                            'Veenker' : 25})

    df.Condition1 = df.Condition1.map({'Artery' : 1, 
                                       'Feedr' : 2, 
                                       'Norm' : 3, 
                                       'PosA' : 4,
                                       'PosN' : 5,
                                       'RRAe' : 6,
                                       'RRAn' : 7,
                                       'RRNe' : 8,
                                       'RRNn' : 9})
    
    
    df.Condition2 = df.Condition2.map({'Artery' : 1, 
                                       'Feedr' : 2, 
                                       'Norm' : 3, 
                                      'PosA' : 4,
                                       'PosN' : 5,
                                       'RRAe' : 6,
                                       'RRAn' : 7,
                                       'RRNe' : 8,
                                       'RRNn' : 9})
    
    df.BldgType = df.BldgType.map({'1Fam':1,
                                   '2fmCon' : 2,
                                   'Duplex' : 3,
                                   'Twnhs' : 4,
                                   'TwnhsE' : 5})
    

    df.HouseStyle = df.HouseStyle.map({'1.5Fin' : 1, 
                                       '1.5Unf' : 2, 
                                       '1Story' : 3,
                                       '2.5Fin' : 4,
                                       '2.5Unf' : 5,
                                       '2Story' : 6,
                                       'SFoyer': 7,
                                       'SLvl' : 8})   
    
    

    df.RoofStyle = df.RoofStyle.map({'Flat' : 1,
                                'Gable' : 2,
                                'Gambrel' : 3,
                                'Hip' : 4,
                                'Mansard' : 5,
                                'Shed' : 6})

    df.RoofMatl = df.RoofMatl.map({'ClyTile' : 1,
                              'CompShg' : 2,
                              'Membran' : 3,
                              'Metal' : 4,
                              'Roll' : 5,
                              'Tar&Grv' : 6,
                              'WdShake' : 7,
                              'WdShngl' : 8})

    df.Exterior1st = df.Exterior1st.map({'AsbShng' : 1,
                                    'AsphShn' : 2,
                                    'BrkComm' : 3,
                                    'BrkFace' : 4,
                                    'CBlock' : 5,
                                    'CemntBd' : 6,
                                    'HdBoard' : 7,
                                    'ImStucc' : 8,
                                    'MetalSd' : 9,
                                    'Plywood' : 10,
                                    'Stone' : 11,
                                    'Stucco' : 12,
                                    'VinylSd' : 13,
                                    'Wd Sdng' : 14,
                                    'WdShing' : 15})

       
    
    df.Exterior2nd = df.Exterior2nd.map({'AsbShng' : 1,
                                    'AsphShn' : 2,
                                    'BrkCmn' : 3,
                                    'Brk Cmn' : 3,
                                    'BrkFace' : 4,
                                    'CBlock' : 5,
                                    'CmentBd' : 6,
                                    'HdBoard' : 7,
                                    'ImStucc' : 8,
                                    'MetalSd' : 9,
                                    'Plywood' : 10,
                                    'Stone' : 11,
                                    'Stucco' : 12,
                                    'VinylSd' : 13,
                                    'Wd Sdng' : 14,
                                    'Wd Shng' : 15,
                                    'Other' : 16})
    
    
    
    df.MasVnrType = df.MasVnrType.map({'BrkCmn' : 1,
                                  'BrkFace' : 2,
                                  'CBlock' : 3,
                                  'None' : 4,
                                  'Stone' : 5})
    
##    df.ExterQual = df.ExterQual.map({'Ex': 1,
##                                 'Fa' :2,
##                                 'Gd' : 3,
##                                 'TA' : 4})
    
    df.ExterCond = df.ExterCond.map({'Ex': 1,
                                 'Fa' :2,
                                 'Gd' : 3,
                                 'TA' : 4,
                                 'Po' : 5})
    
    df.Foundation = df.Foundation.map({'BrkTil' : 1,
                                   'CBlock' : 2,
                                   'PConc' : 3,
                                   'Slab' : 4,
                                   'Stone' : 5,
                                   'Wood' : 6})
    
    df.BsmtQual = df.BsmtQual.map({'Ex': 1,
                                 'Fa' :2,
                                 'Gd' : 3,
                                 'TA' : 4,
                                 'Po' : 5})
    
    df.BsmtCond = df.BsmtCond.map({'Ex': 1,
                                 'Fa' :2,
                                 'Gd' : 3,
                                 'TA' : 4,
                                 'Po' : 5})
    
    df.BsmtExposure = df.BsmtExposure.map({'Av' : 1,
                                       'Gd' : 2,
                                       'Mn' : 3})
                                      
    
    df.BsmtFinType1 = df.BsmtFinType1.map({'ALQ' : 1,
                                       'BLQ' : 2,
                                       'GLQ' : 3,
                                       'LwQ' : 4,
                                       'NA' : 5,
                                       'Rec' : 6,
                                       'Unf' : 7})
    
    
    df.BsmtFinType2 = df.BsmtFinType2.map({'ALQ' : 1,
                                       'BLQ' : 2,
                                       'GLQ' : 3,
                                       'LwQ' : 4,
                                       'NA' : 5,
                                       'Rec' : 6,
                                       'Unf' : 7})
    
    df.Heating = df.Heating.map({'Floor' : 1,
                             'GasA' : 2,
                             'GasW' : 3,
                             'Grav' : 4,
                             'OthW' : 5,
                             'Wall' : 6})
    
    df.HeatingQC = df.HeatingQC.map({'Ex': 1,
                                 'Fa' :2,
                                 'Gd' : 3,
                                 'TA' : 4,
                                 'Po' : 5})
    
    df.CentralAir = df.CentralAir.map({'Y' : 0,
                                   'N' : 1})
    
    df.Electrical = df.Electrical.map({'FuseA' : 1,
                                   'FuseF' : 2,
                                   'FuseP' :3,
                                   'Mix' : 4,
                                   'SBrKr' : 5})
    
    
    df.KitchenQual = df.KitchenQual.map({'Ex': 1,
                                 'Fa' :2,
                                 'Gd' : 3,
                                 'TA' : 4,
                                 'Po' : 5})
    
    
    df.Functional = df.Functional.map({'Maj1' : 1,
                                   'Maj2' : 2,
                                   'Min1' : 3,
                                   'Min2' : 4,
                                   'Mod' : 5,
                                   'Sev' : 6,
                                   'Typ' : 7,
                                   'Sal' : 8})
    
    
    df.FireplaceQu = df.FireplaceQu.map({'Ex': 1,
                                 'Fa' :2,
                                 'Gd' : 3,
                                 'TA' : 4,
                                 'Po' : 5})
                           
    
    df.GarageType = df.GarageType.map({'2Types' : 1,
                                   'Attchd' : 2,
                                   'Basment' : 3,
                                   'Builtln' : 4,
                                   'CarPort' : 5,
                                   'Detchd' : 6})
                            
    
    df.GarageFinish = df.GarageFinish.map({'Fin' : 1,
                                       'RFn' :2,
                                       'Unf' : 3})
    
    
    df.GarageQual = df.GarageQual.map({'Ex': 1,
                                 'Gd' : 2,
                                 'TA' : 3,
                                 'Fa' : 4,
                                 'Po' : 5})
                          
    
    df.GarageCond = df.GarageCond.map({'Ex': 1,
                                 'Gd' : 2,
                                 'TA' : 3,
                                 'Fa' : 4,
                                 'Po' : 5})
    
    
    df.PavedDrive = df.PavedDrive.map({'Y' : 1,
                                   'N' : 2,
                                   'P' :3})
    
    
    df.PoolQC = df.PoolQC.map({'Ex': 1,
                                 'Fa' :2,
                                 'Gd' : 3})
                               
    
    df.Fence = df.Fence.map({'GdPrv' : 1,
                         'GdWo' : 2,
                         'MnPrv' : 3,
                         'MnWw' : 4})
                  
    
    df.MiscFeature = df.MiscFeature.map({'Elev' : 1,
                                         'Gar2' : 2,
                                     'Othr' : 3,
                                     'Shed' : 4,
                                     'TenC' : 5})
    
    df.SaleType = df.SaleType.map({'COD' : 1,
                               'Con' : 2,
                               'ConLD' : 3,
                               'ConLI' : 4,
                               'ConLw' : 5,
                               'CWD' : 6,
                               'New' : 7,
                               'Oth' : 8,
                               'WD' : 9,
                               'VWD' : 10})
    
    
    df.SaleCondition = df.SaleCondition.map({'Abnorml' : 1,
                                         'AdjLand' : 2,
                                         'Alloca' : 3,
                                         'Family' : 4,
                                         'Partial' : 5,
                                         'Normal' : 6})
    
     
    return df
  

def fill_string_getdummies(df) :
 
    df_new = pd.get_dummies(df['Neighborhood','ExterQual','BsmtQual',
                                'GarageFinish','FireplaceQu','Foundation',
                                'GarageType','BsmtFinType1','HeatingQC',
                               'Exterior1st','Exterior2nd'])
    
    df = pd.concat([df,df_new],axis=1)
    
    print(df.columns)
    
    return df
    
def converttoint(df) :

    df['MSZoning'] = df['MSZoning'].fillna(0)   
   
    df.MSZoning = df.MSZoning.round(0).astype(int)

    df.Street = df.Street.round(0).astype(int)

    df.Alley = df.Street.round(0).astype(int)
    
    df.LotShape = df.LotShape.round(0).astype(int)
    
    df.LandContour = df.LandContour.round(0).astype(int)

    df.Utilities = df.Utilities.fillna(0)
    
    df.Utilities = df.Utilities.round(0).astype(int)
    
    df['LotFrontage'][df['LotFrontage'].isnull()] = df.LotFrontage.mean()

    df.LotConfig = df.LotConfig.round(0).astype(int)

    df.LandSlope = df.LandSlope.round(0).astype(int)
    
##    df.Neighborhood = df.Neighborhood.round(0).astype(int)

    df.Condition1 = df.Condition1.round(0).astype(int)
        
    df.Condition2 = df.Condition2.round(0).astype(int)
    
    df.BldgType = df.BldgType.round(0).astype(int)

    df.HouseStyle = df.HouseStyle.round(0).astype(int)
    
    

    df.RoofStyle = df.RoofStyle.round(0).astype(int)

    df.RoofMatl = df.RoofMatl.round(0).astype(int)

    df.Exterior1st = df.Exterior1st.fillna(0)
    
    df.Exterior1st = df.Exterior1st.round(0).astype(int)
       
    df.Exterior2nd = df.Exterior2nd.fillna(0)
    
    df.Exterior2nd = df.Exterior2nd.round(0).astype(int)
    
    df.MasVnrType = df.MasVnrType.fillna(0)
    
    df.MasVnrType = df.MasVnrType.round(0).astype(int)
    
    df.MasVnrArea = df.MasVnrArea.fillna(0)
    
##    df.ExterQual = df.ExterQual.fillna(0)
    
##    df.ExterQual = df.ExterQual.round(0).astype(int)
    
    df.ExterCond = df.ExterCond.fillna(0)
    
    df.ExterCond = df.ExterCond.round(0).astype(int)
    
    df.Foundation = df.Foundation.round(0).astype(int)
    
    df.BsmtQual = df.BsmtQual.fillna(0)
    
    df.BsmtQual = df.BsmtQual.round(0).astype(int)
       
    df.BsmtCond = df.BsmtCond.fillna(0)   
    
    df.BsmtCond = df.BsmtCond.round(0).astype(int)
        
    df.BsmtExposure = df.BsmtExposure.fillna(0)
    
    df.BsmtExposure = df.BsmtExposure.round(0).astype(int)
    
    df.BsmtFinType1 = df.BsmtFinType1.fillna(0)
      
    df.BsmtFinType1 = df.BsmtFinType1.round(0).astype(int)
    
    df.BsmtFinType2 = df.BsmtFinType2.fillna(0)
    
    df.BsmtFinType2 = df.BsmtFinType2.round(0).astype(int)
    
    df.Heating = df.Heating.round(0).astype(int)
    
    df.HeatingQC = df.HeatingQC.round(0).astype(int)
    
    df.CentralAir = df.CentralAir.round(0).astype(int)
    
    df.Electrical = df.Electrical.fillna(0)

    df.Electrical = df.Electrical.round(0).astype(int)
    
    df.KitchenQual = df.KitchenQual.fillna(0)
    
    df.KitchenQual = df.KitchenQual.round(0).astype(int)
    
    df.Functional = df.Functional.fillna(0)
    
    df.Functional = df.Functional.round(0).astype(int)
    
    df.FireplaceQu = df.FireplaceQu.fillna(0)
    
    df.FireplaceQu = df.FireplaceQu.round(0).astype(int)
    
    df.GarageType = df.GarageType.fillna(0)
    
    df.GarageType = df.GarageType.round(0).astype(int)
    
    df['GarageYrBlt'][df['GarageYrBlt'].isnull()] = df.YearBuilt
       
    df.GarageFinish = df.GarageFinish.fillna(0)
    
    df.GarageFinish = df.GarageFinish.round(0).astype(int)
    
    df.GarageCars = df.GarageCars.fillna(0)
    
    df.GarageArea = df.GarageArea.fillna(0)
    
    df.GarageQual = df.GarageQual.fillna(0)
    
    df.GarageQual = df.GarageQual.round(0).astype(int)
    
    df.GarageCond = df.GarageCond.fillna(0)
    
    df.GarageCond = df.GarageCond.round(0).astype(int)
    
    
    df.PavedDrive = df.PavedDrive.round(0).astype(int)
    
    df.PoolQC = df.PoolQC.fillna(0)
    
    df.PoolQC = df.PoolQC.round(0).astype(int)
    
    df.Fence = df.Fence.round(0).fillna(0)
    
    df.Fence = df.Fence.round(0).astype(int)
    
    df.MiscFeature = df.MiscFeature.fillna(0)
    
    df.MiscFeature = df.MiscFeature.round(0).astype(int)
    
    df.SaleType = df.SaleType.fillna(0)

    df.SaleType = df.SaleType.round(0).astype(int)
      
    df.SaleCondition = df.SaleCondition.round(0).astype(int)

    df = df.fillna(0)
    
    df = df.round(0).astype(int)
 
    return df

    
def find_derived_features(df):

    df['TotalBuiltupArea'] = df['BsmtFinSF1']+ df['BsmtFinSF2']+df['1stFlrSF']+df['2ndFlrSF']+df['GrLivArea'] 

    return df

##    df['Age'] = 0

##    df['Age'] = df.apply(lambda x: abs(df['YearRemodAdd'] - df['YrSold']))

##    return df
    
df = pd.get_dummies(data = df, columns = ['Neighborhood','ExterQual'], drop_first = True)


##X = fill_select_string_nans(X)
   
df = fill_string_nans(df)

df = find_derived_features(df)


##df = fill_string_getdummies(df)

df = converttoint(df)

##df = df.drop(df['GrLivArea'] > 4000)


##df = df.drop(df['YearBuilt'] < 1900)

##print(df.YearBuilt)


##print(df)

##print(df.describe())

##print(df.corr())

y = df['SalePrice']

X = df[['OverallQual','YrSold','SaleType','SaleCondition',
        'BsmtQual','KitchenQual','GarageCars','GarageYrBlt','GarageArea',
        'TotalBsmtSF',
##        'BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',
        'LotFrontage','LotArea',
        '1stFlrSF','2ndFlrSF','GrLivArea',
        'FullBath','GarageFinish','MasVnrArea','MasVnrType',
        'TotRmsAbvGrd','Foundation','YearBuilt','YearRemodAdd',
        'GarageType','HeatingQC',
        'Exterior2nd','Exterior1st',
##        'Neighborhood','ExterQual']]
        'Neighborhood_Blueste',
        'Neighborhood_BrDale','Neighborhood_BrkSide','Neighborhood_ClearCr',
        'Neighborhood_CollgCr','Neighborhood_Crawfor','Neighborhood_Edwards',
        'Neighborhood_Gilbert','Neighborhood_IDOTRR','Neighborhood_MeadowV',
        'Neighborhood_Mitchel','Neighborhood_NAmes','Neighborhood_NoRidge',
        'Neighborhood_NPkVill','Neighborhood_NridgHt','Neighborhood_NWAmes',
        'Neighborhood_OldTown','Neighborhood_Sawyer','Neighborhood_SawyerW',
         'Neighborhood_Somerst','Neighborhood_StoneBr','Neighborhood_SWISU',
         'Neighborhood_Timber','Neighborhood_Veenker','ExterQual_Fa',
         'ExterQual_Gd','ExterQual_TA']]
         
X.hist()

plt.show()

   

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.33, random_state=1)

T = preprocessing.Normalizer().fit(X_train[['1stFlrSF','2ndFlrSF','GrLivArea','GarageYrBlt','GarageArea','YrSold','MasVnrArea',
       'TotalBsmtSF','LotFrontage','YearBuilt','YearRemodAdd','LotArea']])

X_train[['1stFlrSF','2ndFlrSF','GrLivArea','GarageYrBlt','GarageArea','YrSold','MasVnrArea',
        'TotalBsmtSF','LotFrontage','YearBuilt','YearRemodAdd','LotArea']] = T.transform(X_train[['1stFlrSF','2ndFlrSF','GrLivArea','GarageYrBlt','GarageArea','YrSold','MasVnrArea',
        'TotalBsmtSF','LotFrontage','YearBuilt','YearRemodAdd','LotArea']])

X_test[['1stFlrSF','2ndFlrSF','GrLivArea','GarageYrBlt','GarageArea','YrSold','MasVnrArea',
       'TotalBsmtSF','LotFrontage','YearBuilt','YearRemodAdd','LotArea']] = T.transform(X_test[['1stFlrSF','2ndFlrSF','GrLivArea','GarageYrBlt','GarageArea','YrSold','MasVnrArea',
        'TotalBsmtSF','LotFrontage','YearBuilt','YearRemodAdd','LotArea']])

model = tree.DecisionTreeRegressor(max_depth=5)

model.fit(X_train, y_train)

score = model.score(X_test, y_test)

print("Decision Tree score: ", score*100)

predict = model.predict(X_test).round(0).astype(int)

print(predict)

print(y_test)

##model = RandomForestClassifier(n_estimators = 10, oob_score = True, max_depth = 10, random_state = 0)


##model.fit(X_train, y_train)

##score = model.score(X_test, y_test)

##print("Random Forest score: ",score*100)

##print(model.predict(X_test))

##print("y test", y_test)


df = pd.read_csv('test.csv')

df = pd.get_dummies(data = df, columns = ['Neighborhood','ExterQual'], drop_first = True)


df = fill_string_nans(df)

df = find_derived_features(df)

df = converttoint(df)

y = df[['Id']]

X = df[['OverallQual','GrLivArea','YrSold','SaleType','SaleCondition',
        'BsmtQual','KitchenQual','GarageCars','GarageYrBlt','GarageArea',
        'TotalBsmtSF',
##        'BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',
        '1stFlrSF','2ndFlrSF','LotFrontage','LotArea',
        'FullBath','GarageFinish','MasVnrArea','MasVnrType',
        'TotRmsAbvGrd','Foundation','YearBuilt','YearRemodAdd',
        'GarageType','HeatingQC',
        'Exterior2nd','Exterior1st',
##        ,'Neighborhood','ExterQual']]
        'Neighborhood_Blueste', 
        'Neighborhood_BrDale','Neighborhood_BrkSide','Neighborhood_ClearCr',
        'Neighborhood_CollgCr','Neighborhood_Crawfor','Neighborhood_Edwards',
        'Neighborhood_Gilbert','Neighborhood_IDOTRR','Neighborhood_MeadowV',
        'Neighborhood_Mitchel','Neighborhood_NAmes','Neighborhood_NoRidge',
        'Neighborhood_NPkVill','Neighborhood_NridgHt','Neighborhood_NWAmes',
        'Neighborhood_OldTown','Neighborhood_Sawyer','Neighborhood_SawyerW',
         'Neighborhood_Somerst','Neighborhood_StoneBr','Neighborhood_SWISU',
         'Neighborhood_Timber','Neighborhood_Veenker','ExterQual_Fa',
         'ExterQual_Gd','ExterQual_TA']]
         
         
T = preprocessing.Normalizer().fit(X[['GrLivArea','GarageYrBlt','GarageArea','YrSold','MasVnrArea',
       'TotalBsmtSF','1stFlrSF','2ndFlrSF','LotFrontage','YearBuilt','YearRemodAdd','LotArea']])

X[['GrLivArea','GarageYrBlt','GarageArea','YrSold','MasVnrArea',
        'TotalBsmtSF','1stFlrSF','2ndFlrSF','LotFrontage','YearBuilt','YearRemodAdd','LotArea']] = T.transform(X[['GrLivArea','GarageYrBlt','GarageArea','YrSold','MasVnrArea',
        'TotalBsmtSF','1stFlrSF','2ndFlrSF','LotFrontage','YearBuilt','YearRemodAdd','LotArea']])


predict = model.predict(X).round(0).astype(int)

print(predict)

predict = pd.DataFrame(predict, columns = ['SalePrice'])

predict = pd.concat([y, predict], axis=1)

print("Predicted output test result", predict)

predict.to_csv('Sample_Submission.csv', index = False)











