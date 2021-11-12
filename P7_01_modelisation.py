#!/usr/bin/env python
# coding: utf-8

# # PROJECT #7 : SCORING MODEL

# # Librairies

# In[1]:


import timeit
from   os                              import listdir
import warnings

import numpy                           as np
import pandas                          as pd
import matplotlib.pyplot               as plt
import seaborn                         as sns

import matplotlib                      as mpl
from   matplotlib                      import cm

import shap
import plotly.graph_objects            as go


from   sklearn.experimental            import enable_iterative_imputer
from   sklearn.impute                  import SimpleImputer, IterativeImputer

from   sklearn.preprocessing           import LabelEncoder, OneHotEncoder, OrdinalEncoder
from   sklearn.preprocessing           import PolynomialFeatures

from   sklearn.model_selection         import train_test_split

from   sklearn                         import preprocessing

from   sklearn                         import model_selection
from   sklearn                         import linear_model
from   sklearn                         import metrics, dummy

from   sklearn                         import neighbors, kernel_ridge

from   sklearn.metrics                 import roc_curve, auc

from   sklearn.linear_model            import Ridge, LogisticRegression
from   sklearn.model_selection         import StratifiedKFold
from   sklearn.feature_selection       import SelectFromModel


from   sklearn.ensemble                import BaggingClassifier
from   sklearn.ensemble                import RandomForestClassifier
from   sklearn.ensemble                import GradientBoostingClassifier

from   imblearn.under_sampling         import RandomUnderSampler

import lightgbm                        as lgb
from   catboost                        import CatBoostClassifier


from   quilt.data.ResidentMario        import missingno_data
import missingno                       as msno


import collections
from   collections                     import Counter
import pickle

import requests
import json

import ast
import time

import datetime
import re


# # Parameters

# In[2]:


pd.set_option('display.max_columns', 500)

sns.set()

get_ipython().run_line_magic('matplotlib', 'inline')

warnings.filterwarnings('ignore')


# In[3]:


path       = "C:/Users/fredj/Desktop/data_scientist/project_7_scoring/data/"


# In[4]:


cols_names = {"SK_ID_CURR"                   : "id", 
              
              "TARGET"                       : "target",
              
              "DAYS_BIRTH"                   : "age", 
              "CODE_GENDER"                  : "gender", 
              "NAME_FAMILY_STATUS"           : "family_status",
              "CNT_FAM_MEMBERS"              : "family_nb",
              "CNT_CHILDREN"                 : "children_nb",
              "NAME_TYPE_SUITE"              : "type_suite", 
              
              "NAME_EDUCATION_TYPE"          : "education_type", 
              
              "OCCUPATION_TYPE"              : "work_type",
              "ORGANIZATION_TYPE"            : "organization_type",
              "DAYS_EMPLOYED"                : "work_exp",
              "AMT_INCOME_TOTAL"             : "income_amount", 
              "NAME_INCOME_TYPE"             : "income_type",
              
              "NAME_CONTRACT_TYPE"           : "contract", 
              "AMT_CREDIT"                   : "credit_amount", 
              "AMT_ANNUITY"                  : "annuity", 
              "AMT_GOODS_PRICE"              : "goods_price", 
              "DAYS_REGISTRATION"            : "credit_time_elapsed",             
              "DAYS_ID_PUBLISH"              : "id_published_time_elapsed",
              "WEEKDAY_APPR_PROCESS_START"   : "appr_process_weekday",
              "HOUR_APPR_PROCESS_START"      : "appr_process_hour",         
              
              "FLAG_OWN_CAR"                 : "own_car", 
              "OWN_CAR_AGE"                  : "own_car_age",
              
              "FLAG_OWN_REALTY"              : "own_realty", 
              "NAME_HOUSING_TYPE"            : "housing_type",
              
              
              "APARTMENTS_AVG"               : "apartments_avg",
              "ELEVATORS_AVG"                : "elevators_avg", 
              "ENTRANCES_AVG"                : "entrances_avg", 
              "FLOORSMAX_AVG"                : "floors_max_avg", 
              "FLOORSMIN_AVG"                : "floors_min_avg", 
              
              "APARTMENTS_MODE"              : "apartments_mode",
              'ELEVATORS_MODE'               : "elevators_mode", 
              'ENTRANCES_MODE'               : "entrances_mode", 
              'FLOORSMAX_MODE'               : "floors_max_mode", 
              'FLOORSMIN_MODE'               : "floors_min_mode", 
              
              'APARTMENTS_MEDI'              : "apartments_medi",
              'ELEVATORS_MEDI'               : "elevators_medi", 
              'ENTRANCES_MEDI'               : "entrances_medi",
              'FLOORSMAX_MEDI'               : "floors_max_medi", 
              'FLOORSMIN_MEDI'               : "floors_min_medi",
              
              "BASEMENTAREA_AVG"             : "basement_area_avg",
              "COMMONAREA_AVG"               : "common_area_avg",
              "LANDAREA_AVG"                 : "land_area_avg",             
              "LIVINGAREA_AVG"               : "living_area_avg",
              "NONLIVINGAREA_AVG"            : "non_living_area_avg",             
              "LIVINGAPARTMENTS_AVG"         : "living_apartments_avg",              
              "NONLIVINGAPARTMENTS_AVG"      : "non_living_apartments_avg",              
              
              "BASEMENTAREA_MODE"            : "basement_area_mode",              
              'COMMONAREA_MODE'              : "common_area_mode",              
              'LANDAREA_MODE'                : "land_area_mode",    
              'LIVINGAPARTMENTS_MODE'        : "living_apartment_mode",
              'NONLIVINGAPARTMENTS_MODE'     : "non_living_apartments_mode",
              'LIVINGAREA_MODE'              : "living_area_mode",             
              'NONLIVINGAREA_MODE'           : "non_living_area_mode",
              
              'BASEMENTAREA_MEDI'            : "basement_area_medi",              
              'COMMONAREA_MEDI'              : "common_area_medi",               
              'LANDAREA_MEDI'                : "land_area_medi", 
              'LIVINGAPARTMENTS_MEDI'        : "living_apartments_medi",
              'LIVINGAREA_MEDI'              : "living_area_medi", 
              'NONLIVINGAPARTMENTS_MEDI'     : "non_living_apartments_medi",
              'NONLIVINGAREA_MEDI'           : "non_living_area_medi",
              
              "YEARS_BEGINEXPLUATATION_AVG"  : "years_eval_avg",
              "YEARS_BUILD_AVG"              : "years_build_avg",
              
              'YEARS_BEGINEXPLUATATION_MODE' : "years_eval_mode",
              'YEARS_BUILD_MODE'             : "years_build_mode",
                            
              'YEARS_BEGINEXPLUATATION_MEDI' : "years_eval_medi",
              'YEARS_BUILD_MEDI'             : "years_build_medi",
              
              'FONDKAPREMONT_MODE'           : "fondkapremont_mode",
              'HOUSETYPE_MODE'               : "house_type_mode",
              'TOTALAREA_MODE'               : "total_area_mode",
              'WALLSMATERIAL_MODE'           : "walls_material_mode",
              'EMERGENCYSTATE_MODE'          : "emergency_state_mode",
              
              
              "REGION_POPULATION_RELATIVE"   : "region_pop_rate", 
              
              "REGION_RATING_CLIENT"         : "region_rating",               
              "REG_REGION_NOT_LIVE_REGION"   : "region_not_live", 
              "REG_REGION_NOT_WORK_REGION"   : "region_not_work", 
              "LIVE_REGION_NOT_WORK_REGION"  : "live_region_not_work", 
              
              "REGION_RATING_CLIENT_W_CITY"  : "city_rating",
              "REG_CITY_NOT_LIVE_CITY"       : "city_not_live", 
              "REG_CITY_NOT_WORK_CITY"       : "city_not_work", 
              "LIVE_CITY_NOT_WORK_CITY"      : "live_city_not_work",              
              
              'OBS_30_CNT_SOCIAL_CIRCLE'     : "social_circle_30_obs_nb",
              'DEF_30_CNT_SOCIAL_CIRCLE'     : "social_circle_30_def_nb",
              'OBS_60_CNT_SOCIAL_CIRCLE'     : "social_circle_60_obs_nb",
              'DEF_60_CNT_SOCIAL_CIRCLE'     : "social_circle_60_def_nb",
              
              
              'DAYS_LAST_PHONE_CHANGE'       : "last_call_days",
              
              "FLAG_MOBIL"                   : "mobile",
              "FLAG_PHONE"                   : "phone", 
              "FLAG_EMP_PHONE"               : "emp_phone", 
              "FLAG_WORK_PHONE"              : "work_phone",
              "FLAG_CONT_MOBILE"             : "cont_mobile",
              "FLAG_EMAIL"                   : "email", 
              
              'FLAG_DOCUMENT_2'              : "doc_2",
              'FLAG_DOCUMENT_3'              : "doc_3",
              'FLAG_DOCUMENT_4'              : "doc_4",
              'FLAG_DOCUMENT_5'              : "doc_5",
              'FLAG_DOCUMENT_6'              : "doc_6",
              'FLAG_DOCUMENT_7'              : "doc_7",
              'FLAG_DOCUMENT_8'              : "doc_8",
              'FLAG_DOCUMENT_9'              : "doc_9",
              'FLAG_DOCUMENT_10'             : "doc_10",
              'FLAG_DOCUMENT_11'             : "doc_11",
              'FLAG_DOCUMENT_12'             : "doc_12",
              'FLAG_DOCUMENT_13'             : "doc_13",
              'FLAG_DOCUMENT_14'             : "doc_14",
              'FLAG_DOCUMENT_15'             : "doc_15",
              'FLAG_DOCUMENT_16'             : "doc_16",
              'FLAG_DOCUMENT_17'             : "doc_17",
              'FLAG_DOCUMENT_18'             : "doc_18",
              'FLAG_DOCUMENT_19'             : "doc_19",
              'FLAG_DOCUMENT_20'             : "doc_20",
              'FLAG_DOCUMENT_21'             : "doc_21",
              
              
              'OBS_30_CNT_SOCIAL_CIRCLE'     : "social_circle_30_obs_nb",
              'DEF_30_CNT_SOCIAL_CIRCLE'     : "social_circle_30_def_nb",
              'OBS_60_CNT_SOCIAL_CIRCLE'     : "social_circle_60_obs_nb",
              'DEF_60_CNT_SOCIAL_CIRCLE'     : "social_circle_60_def_nb",
              
              
              'AMT_REQ_CREDIT_BUREAU_HOUR'   : "req_credit_hour",
              'AMT_REQ_CREDIT_BUREAU_DAY'    : "req_credit_day",
              'AMT_REQ_CREDIT_BUREAU_WEEK'   : "req_credit_week",
              'AMT_REQ_CREDIT_BUREAU_MON'    : "req_credit_mon",
              'AMT_REQ_CREDIT_BUREAU_QRT'    : "req_credit_qrt",
              'AMT_REQ_CREDIT_BUREAU_YEAR'   : "req_credit_year",
                  
              
              "EXT_SOURCE_1"                 : "ext_source_1",
              "EXT_SOURCE_2"                 : "ext_source_2",
              "EXT_SOURCE_3"                 : "ext_source_3",                  
             }


# In[5]:


cols_sort = ['target', 
             'age',
             'gender',
             'family_status',
             'family_nb',
             'children_nb',
             'type_suite',
             'education_type',
             'work_type',
             'organization_type',
             'work_exp',
             'work_exp_outliers',
             'work_exp_rate',
             'income_amount',
             'income_type',
             'contract',
             'credit_amount',
             'debt_rate',
             'debt_load',
             'credit_term',
             'annuity',
             'goods_price',
             'cover_rate',       
             'credit_time_elapsed',
             'id_published_time_elapsed',
             'appr_process_weekday',
             'appr_process_hour',
             'own_car',
             'own_car_age',
             'own_realty',
             'housing_type',
             'apartments_avg',
             'elevators_avg',
             'entrances_avg',
             'floors_max_avg',
             'floors_min_avg',
             'apartments_mode',
             'elevators_mode',
             'entrances_mode',
             'floors_max_mode',
             'floors_min_mode',
             'apartments_medi',
             'elevators_medi',
             'entrances_medi',
             'floors_max_medi',
             'floors_min_medi',
             'basement_area_avg',
             'common_area_avg',
             'land_area_avg',
             'living_area_avg',
             'non_living_area_avg',
             'living_apartments_avg',
             'non_living_apartments_avg',
             'basement_area_mode',
             'common_area_mode',
             'land_area_mode',
             'living_apartment_mode',
             'non_living_apartments_mode',
             'living_area_mode',
             'non_living_area_mode',
             'basement_area_medi',
             'common_area_medi',
             'land_area_medi',
             'living_apartments_medi',
             'living_area_medi',
             'non_living_apartments_medi',
             'non_living_area_medi',
             'years_eval_avg',
             'years_build_avg',
             'years_eval_mode',
             'years_build_mode',
             'years_eval_medi',
             'years_build_medi',
             'fondkapremont_mode',
             'house_type_mode',
             'total_area_mode',
             'walls_material_mode',
             'emergency_state_mode',
             'region_pop_rate',
             'region_rating',
             'region_not_live',
             'region_not_work',
             'live_region_not_work',
             'city_rating',
             'city_not_live',
             'city_not_work',
             'live_city_not_work',
             'social_circle_30_obs_nb',
             'social_circle_30_def_nb',
             'social_circle_60_obs_nb',
             'social_circle_60_def_nb',
             'last_call_days',
             'mobile',
             'phone',
             'emp_phone',
             'work_phone',
             'cont_mobile',
             'email',
             'doc_2',
             'doc_3',
             'doc_4',
             'doc_5',
             'doc_6',
             'doc_7',
             'doc_8',
             'doc_9',
             'doc_10',
             'doc_11',
             'doc_12',
             'doc_13',
             'doc_14',
             'doc_15',
             'doc_16',
             'doc_17',
             'doc_18',
             'doc_19',
             'doc_20',
             'doc_21',
             'req_credit_hour',
             'req_credit_day',
             'req_credit_week',
             'req_credit_mon',
             'req_credit_qrt',
             'req_credit_year',
             'ext_source_1',
             'ext_source_2',
             'ext_source_3',
             'ext_source_1^2',
             'ext_source_1 ext_source_1',
             'ext_source_1 ext_source_2',
             'ext_source_1 age',
             'ext_source_1 work_exp',
             'ext_source_2^2',
             'ext_source_2 age',
             'ext_source_2 work_exp',
             'age^2',
             'age work_exp',
             'work_exp^2',
             'ext_source_1^3',
             'ext_source_1^2 ext_source_1',
             'ext_source_1^2 ext_source_2',
             'ext_source_1^2 age',
             'ext_source_1^2 work_exp',
             'ext_source_1 ext_source_1^2',
             'ext_source_1 ext_source_1 ext_source_2',
             'ext_source_1 ext_source_1 age',
             'ext_source_1 ext_source_1 work_exp',
             'ext_source_1 ext_source_2^2',
             'ext_source_1 ext_source_2 age',
             'ext_source_1 ext_source_2 work_exp',
             'ext_source_1 age^2',
             'ext_source_1 age work_exp',
             'ext_source_1 work_exp^2',
             'ext_source_2^3',
             'ext_source_2^2 age',
             'ext_source_2^2 work_exp',
             'ext_source_2 age^2',
             'ext_source_2 age work_exp',
             'ext_source_2 work_exp^2',
             'age^3',
             'age^2 work_exp',
             'age work_exp^2',
             'work_exp^3',
            ]


# In[6]:


ordinal_feats = ['contract', 
                 'own_car', 
                 'own_realty'
                ]
      
one_hot_feats = ["gender",
                 "family_status",
                 "type_suite",
                 "education_type",
                 "work_type",
                 "organization_type",
                 "income_type",
                 "appr_process_weekday",
                 "housing_type",
                 "fondkapremont_mode",
                 "house_type_mode",
                 "walls_material_mode",
                 "emergency_state_mode",
                ]   

cat_feats     = ordinal_feats + one_hot_feats


# In[7]:


models_dict = {"Dummy"                  : {"strategy"     : "stratified"},
               "KNN"                    : {"model_name"   : "KNN",
                                           "model"        : neighbors.KNeighborsClassifier(), 
                                           "params_grid"  : {'n_neighbors' : [3, 5, 7, 9, 10, 15, 20, 50, 75, 100], 
                                                            }, 
                                          }    ,
               "Logistic Regression"    : {"model_name"   : "Logistic Regression",
                                           "model"        : linear_model.LogisticRegression(random_state=20), 
                                           "params_grid"  : {
#                                                              'solver'   : ['lbfgs', 'newton-cg', 'liblinear'],             
                                                             'penalty'  : ['l2', 'elasticnet'],                 
                                                             'l1_ratio' : np.arange(0, 1.25, 0.25),
                                                             'C'        : [100, 75, 50, 25, 10, 1.0, 0.1, 0.01, 0.001, 0.0001]
                                                            }, 
                                          }       ,
               "Bagging"                : {"model_name"   : "Bagging",
                                           "model"        : BaggingClassifier(random_state=20), 
                                           "params_grid"  : {'n_estimators' : [10, 50], 
                                                             'max_samples'  : np.arange (0.2, 1.2, 0.2), 
                                                            }, 
                                          }      , 
               "Random Forest"          : {"model_name"   : "Random Forest",
                                           "model"        : RandomForestClassifier(oob_score    = True, 
                                                                                   n_estimators = 200, 
                                                                                   random_state = 20
                                                                                  ), 
                                           "params_grid"  : {"max_depth" : [3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 50],  
                                                            }, 
                                          }, 
               "Gradient Boosting"      : {"model_name"   : "XG Boost",
                                           "model"        : GradientBoostingClassifier(n_estimators = 200,
                                                                                       random_state = 20
                                                                                     ), 
                                           "params_grid"  : {"max_depth" : [3, 5, 7],  
                                                            }, 
                                          }, 
               "LightGBM"               : {"model_name"   : "Light GBM",
                                           "model"        : lgb.LGBMClassifier(random_state=20), 
                                           "params_grid"  : {"max_depth" : [3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 50],  
                                                            }, 
                                          }, 
#                "CatBoost"               : {"model_name"   : "Cat Boost",
#                                            "model"        : CatBoostClassifier(random_state=20), 
#                                            "params_grid"  : {"max_depth" : [3, 7, 15],  
#                                                             }, 
#                                           }    
              }


# # FUNCTIONS

# # 0. General

# In[8]:


def clean_cmp(style, start_rate=0.15, end_rate=0.85):
    
    """ get color map without extremes values """
    
    interval = np.hstack([np.linspace(0, start_rate), 
                          np.linspace(end_rate, 1)
                         ]
                        )
    
    interval = np.linspace(start_rate, end_rate)
    
    colors   = plt.cm.get_cmap(style)(interval)
    cmap     = mpl.colors.LinearSegmentedColormap.from_list('my_cmap', colors)
    
    return cmap


# # I. Dataset

# In[9]:


def color_under_threshold(value, threshold):
    
    """ color value in red in dateset if value is under a threshold value """
    
    color = 'red' if value < threshold else 'green'
    
    return 'color : {}'.format(color)

def get_data_completion(dataset, comp_threshold=30, display=False):  
    
    """ get dataset rate completion for each column """
    
    data_comp      = dataset.notna().mean(axis = 0).sort_values(ascending = False)
    data_comp      = (data_comp * 100).round(2)
    
    data_comp.name = "completion (%)"
    
    data_comp      = data_comp.to_frame()
    
    if display:
        
        ax         = data_comp.plot(legend  = False, 
                                    rot     = 90, 
                                    figsize = (18, 7), 
                                   )

        plt.title("Variables Completion", 
                  fontsize   = 15, 
                  fontweight = "bold"
                 )
    
    if comp_threshold:
        
        if display:
            
            fig   = ax.axhline(y     = comp_threshold, 
                               color = "red"
                              )
    
        f         = {'completion (%)':'{:.2f}'} 
        
        data_comp = data_comp.style.format(f).applymap(lambda value : color_under_threshold(value, comp_threshold))
        
    return data_comp

def plot_na(data, sample=200): 
    
    " plot missing value matrix"

    if data.shape[1] > 50: 
        
        loop_nb = int(round(data.shape[1] / 50, 0))

        for i in range(loop_nb) : 

            fig = msno.matrix(data.reindex (columns = data.columns[i*50 : (i+1)*50]).sample(sample), 
                              color = (0.066, 0.145, 0.411)
                             )
    else :
        
        fig     = msno.matrix(data.sample(sample), 
                              color = (0.066, 0.145, 0.411)
                             )


# # II. Preprocessing

# In[10]:


def clean_time_data(data):
    
    """ clean time variables """
    
    work_exp_outliers                = data.work_exp > 0
    data.work_exp                    = data.work_exp.replace({365243: np.nan}) / -365.25
    
    data.age                         = data.age / -365.25
    
    data.credit_time_elapsed         = data.credit_time_elapsed / -365.25
    data.id_published_time_elapsed   = data.id_published_time_elapsed / -365.25

    data.last_call_days              = data.last_call_days / -1
    
    return data, work_exp_outliers


# In[11]:


def encode_df(df, var, kind): 

    """ encode categorical variable var in dataset """
    
    if kind == "OneHot":   
        
        df      = pd.get_dummies(df)
          
        return df
    
    elif kind == "Label":
        
        encoder = LabelEncoder()
        results = encoder.fit_transform(df[[var]])
        df[var] = list(results.reshape(1, -1)[0].astype(int))
    
        return df, encoder
    
def encode_data(data, cat_feats, kind): 

    """ encode categorical variables and add them to the dataset """
    
    if kind == "Dummy":
        
        data          = pd.get_dummies(data)
        encoder       = None
        
    
    elif kind == "OneHot":   
        
        encoder       = OneHotEncoder()
        
        one_hot_data  = data[cat_feats]
        
        for var in cat_feats:
            
            one_hot_data[var] = one_hot_data[var].str.replace(', ', '/')
            one_hot_data[var] = one_hot_data[var].str.replace(' / ', '/')
            one_hot_data[var] = one_hot_data[var].str.replace(' ', '_')
            one_hot_data[var] = one_hot_data[var].str.upper()
            
        encoded_array = encoder.fit_transform(one_hot_data).toarray()
                
        encoded_data  = pd.DataFrame(encoded_array, 
                                     columns = encoder.get_feature_names_out(), 
                                     index   = data.index
                                    )
        
        data          = data.drop(cat_feats, axis=1)
        data          = data.merge(encoded_data, left_index=True, right_index=True)
        
        
    elif kind == "Label":
              
        if len(cat_feats) == 1:
            
            encoder   = LabelEncoder()
        
            results   = encoder.fit_transform(data[[var]])        
            data[var] = list(results.reshape(1, -1)[0].astype(int))
            
        else:
            
            encoders  = {}
                
            for var in cat_feats:
    
                encoder       = LabelEncoder()

                results       = encoder.fit_transform(data[[var]])
                data[var]     = list(results.reshape(1, -1)[0].astype(int))

                encoders[var] = encoder
                
            encoder   = encoders.copy()
            
        
    elif kind == "Ordinal":        

        encoder       = OrdinalEncoder()
        
        ordinal_data  = data[cat_feats]
        encoded_array = encoder.fit_transform(ordinal_data)

        encoded_data  = pd.DataFrame(encoded_array, 
                                     columns = cat_feats, 
                                     index   = data.index
                                    )

        data          = data.drop(cat_feats, axis=1)
        data          = data.merge(encoded_data, left_index=True, right_index=True)
    
    
    return data, encoder


# In[12]:


def impute_missing(dataset, method): 

    """ impute missing value with simple or iterative imputer """
    
    if method == "iterative":   
        imputer     = IterativeImputer()
        
    elif method == "simple":
        imputer     = SimpleImputer(strategy="mean")

    target          = dataset[['target']]
    X               = dataset.drop("target", axis=1)
    
    cleaned_dataset = imputer.fit_transform(X)
    
    cleaned_dataset = pd.DataFrame(cleaned_dataset, index=X.index, columns=X.columns)   
    cleaned_dataset = cleaned_dataset.merge(target, left_index=True, right_index=True)
    
    return cleaned_dataset, imputer


# In[13]:


def reverse_encoding(data, cat_feats, encoder):
    
    """ reverse encoding and add decoded categorical variables to the dataset """
    
    categorical_data = data[cat_feats]
    cat_array        = encoder.inverse_transform(categorical_data)
    
    cat_data         = pd.DataFrame(cat_array, 
                                    columns = cat_feats, 
                                    index   = data.index
                                   )
    
    data             = data.drop(cat_feats, axis=1)
    data             = data.merge(cat_data, left_index=True, right_index=True)

    return data


# # III. Exploration

# In[14]:


def plot_pie(data, var, title=None, others_nb=25, others_name='Others', pre_labels=[], counter=False,
             colors=None, colormap='Blues_r', kind='pie'):
    
    """ plot pie chart """
    
    subset  = data.groupby(var)[var].count().sort_values(ascending=False)   
    
    if others_nb: 
        
        others  = subset.iloc[others_nb:].sum()
        subset  = subset.iloc[:others_nb]
        
        if others_name:
            subset[others_name] = others
            
    subset.name = 'nb'
    subset      = subset.to_frame().reset_index()
    subset      = subset.sort_values(by='nb', ascending=False)

    labels      = list(subset[var].values)

    if pre_labels != []:
        labels  = ["{} - {}".format(pre_labels[i], labels[i]) for i in range(len(labels))]

    if counter:
        labels  = ["{} ({})".format(labels[i], 
                                    subset.nb.apply(lambda x: "{:,}".format(x)).to_frame().nb.iloc[i]
                                   ) for i in range(len(labels))]        
    
    if colors: 

        fig = subset.plot(figsize    = (6 + 0.3 * subset.shape[0], 
                                        6 + 0.3 * subset.shape[0]), 
                          legend     = False, 
                          autopct    = '%1.0f%%',
                          kind       = kind, 
#                           explode    = (0.05, 0, 0, 0, 0),
                          wedgeprops = {'linewidth' : 2.0, 
                                        'edgecolor' : 'white'
                                       },
                          textprops  = {"size"  : 13, 
                                        "color" : "black", 
#                                         "fontweight": "bold"
                                       },
                          y          = 'nb',
                          labels     = labels,
                          colors     = [colors[c] for c in list(subset[var].values)],
                         )

    elif colormap: 

        fig = subset.plot(figsize    = (6 + 0.3 * subset.shape[0], 
                                        6 + 0.3 * subset.shape[0]), 
                          legend     = False, 
                          autopct    = '%1.0f%%',
                          kind       = kind, 
#                           explode    = (0.05, 0, 0, 0, 0),
                          wedgeprops = {'linewidth' : 2.0, 
                                        'edgecolor' : 'white'
                                       },
                          textprops  = {"size"  : 13, 
                                        "color" : "black", 
#                                         "fontweight": "bold"
                                       },
                          y          = 'nb',
                          labels     = labels,
                          colormap   = clean_cmp(colormap, 0.2, 0.95)
                     )

    ax  = plt.ylabel ("")
    
    if title is None : 
        title = ' '.join(var.split('_')).title() + ' Mix'
        
    plt.title (title, 
               fontsize   = 15, 
               fontweight = "bold"
              )
    
def plot_kde (dataset, var, title_var):
    
    """ plot kde chart """
    
    plt.figure(figsize = (12, 6))

    fig = sns.kdeplot(data      = dataset[dataset.target == 0], 
                      x         = var, 
                      shade     = True, 
                      alpha     = 0.3, 
                      label     = "repaid",
                      color     = ['forestgreen'], 
                     )

    fig = sns.kdeplot(data      = dataset[dataset.target == 1], 
                      x         = var, 
                      shade     = True, 
                      alpha     = 0.3, 
                      label     = "default",
                      color     = ['firebrick'], 
                     )

    if title_var is None:
        title_var = ' '.join(var.split('_')).title()
        
    plt.title('{} Density by {}'.format(title_var, "Target"),
              fontsize   = 15, 
              fontweight = "bold", 
              pad        = 20
             )

    ax  = plt.xlabel("")
    ax  = plt.ylabel("")

    plt.legend()

    plt.show()
    
def plot_corr_heatmap(corr_matrix, title): 

    """ plot heatmap chart """
    
    fig, ax = plt.subplots(figsize=(10, 8)) 

    sns.heatmap(corr_matrix, 
                annot   = True, 
                cmap    = clean_cmp("RdBu_r", 0.05, 0.9)
                )

    plt.title('Correlation Heatmap : {}'.format(title),
              fontsize   = 15, 
              fontweight = "bold", 
              pad        = 20 
             )
    
    plt.show()


# In[15]:


def get_target_mix(data, y_train=None, y_test=None):
    
    """ get target distribution """
    
    target                      = data.target
    target.name                 = "dataset_nb"

    target                      = target.value_counts().to_frame()
    target.index                = ["repaid (0)", "default (1)"]
    
    if y_train is not None :
        
        y_train_0_nb            = y_train[y_train==0].count()
        y_train_1_nb            = y_train[y_train==1].count()
    
        target['trainset_nb']   = [y_train_0_nb, y_train_1_nb]
        target['trainset_nb']   = target['trainset_nb'].apply(lambda x: "{:,}".format(x))
        
#         y_train_0_freq          = round(y_train_0_nb / y_train.shape[0] * 100, 2)
#         y_train_1_freq          = round(y_train_1_nb / y_train.shape[0] * 100, 2)
        
#         target['trainset_freq'] = [y_train_0_freq, y_train_1_freq]
#         target['trainset_freq'] = target['trainset_freq'].apply(lambda x: "{}%".format(x))
    
    if y_test is not None :
        
        y_test_0_nb             = y_test[y_test==0].count()
        y_test_1_nb             = y_test[y_test==1].count()
    
        target['testset_nb']    = [y_test_0_nb, y_test_1_nb]
        target['testset_nb']    = target['testset_nb'].apply(lambda x: "{:,}".format(x))
         
#         y_test_0_freq           = round(y_test_0_nb / y_test.shape[0] * 100, 2)
#         y_test_1_freq           = round(y_test_1_nb / y_test.shape[0] * 100, 2)
        
#         target['testset_freq']  = [y_test_0_freq, y_test_1_freq]
#         target['testset_freq']  = target['testset_freq'].apply(lambda x: "{}%".format(x))

    target['freq']              = (target['dataset_nb'] / target['dataset_nb'].sum() * 100).round(2)
    
    target.dataset_nb           = target.dataset_nb.apply(lambda x: "{:,}".format(x))
    target.freq                 = target.freq.apply(lambda x: "{}%".format(x))
    
    return target


# # IV. Modelisation : General

# ## 1. Dataset

# In[16]:


def split_dataset(dataset, target_name="target", test_size=0.3): 
    
    """ split dataset into trainin and testing, and standardize them """
    # Split Data & Target
    
    X              = dataset.drop(target_name, axis=1) 
    y              = dataset[target_name]
#     X[target_name] = y                                   # target = last column
    
    
    # Split Training Set & Testing Set
    
    (X_train, 
     X_test, 
     y_train, 
     y_test)      = train_test_split(X, y, 
                                     stratify     = y, 
                                     test_size    = test_size, 
                                     random_state = 20
                                    )

    
    # Get Stratified K Folds for Cross Validation
        
    X_train_ids   = X_train.index                
    X_test_ids    = X_test.index
        
#     X             = X.iloc       [:, :-1]
    
#     X_train       = X_train.iloc [:, :-1]
#     X_test        = X_test.iloc  [:, :-1]
    
#     skf_train   = StratifiedKFold (n_splits = 10).split (X_train, X_train.loc [:, stratify_var])
    
#     X           = X.drop       (columns = [stratify_var])
    
#     X_train     = X_train.drop (columns = [stratify_var])    
#     X_test      = X_test.drop  (columns = [stratify_var])

    
    # Standardize Training Set & Testing Set
    
    std_scaler  = preprocessing.StandardScaler().fit(X_train)

    X_train_std = std_scaler.transform(X_train)
    X_test_std  = std_scaler.transform(X_test)
    
       
    return X, y, X_train_std, y_train, X_test_std, y_test, X_train_ids, X_test_ids, std_scaler


# In[17]:


def sample_train_data(data, train_data, test_data, train_size=0.10):
    
    """ get sampled training set """
    
    N            = data.shape[0] * train_size
    
    train_data   = train_data.groupby('target', group_keys=False
                                     ).apply(lambda x: x.sample(int(np.rint(N * len(x) / len(data))))
                                            ).sample(frac=1)
    
    X_train      = train_data.drop(['target'], axis=1).values
    y_train      = train_data['target']
    
    sampled_data = pd.concat([train_data, test_data], ignore_index=False)
    
    return sampled_data, X_train, y_train


# In[18]:


def resample_data(X_train, y_train, test_data, vars_names):
    
    """ get resampled dataset with balanced target """
    
    rus                  = RandomUnderSampler(sampling_strategy = 1, 
                                              random_state      = 20
                                             )

    (X_train, 
     y_train)            = rus.fit_resample(X_train, y_train)
    
    train_data           = pd.DataFrame(X_train, index=rus.sample_indices_, columns=vars_names)
    train_data['target'] = y_train.values
    
    resampled_data       = pd.concat([train_data, test_data], ignore_index=False)
    
    return resampled_data, X_train, y_train


# ## 2. Results

# In[19]:


def get_scores(model_name, results_dict, score_name, scores_df, update=True):
       
    """ get model scores : auc, recall, precision, f1_score and fbeta_score (if score_name is Febta) """
        
    auc         = metrics.roc_auc_score   (results_dict[model_name]["y_test"], results_dict[model_name]["y_proba"]) 
    recall      = metrics.recall_score    (results_dict[model_name]["y_test"], results_dict[model_name]["y_pred"])
    precision   = metrics.precision_score (results_dict[model_name]["y_test"], results_dict[model_name]["y_pred"])
    f1_score    = metrics.f1_score        (results_dict[model_name]["y_test"], results_dict[model_name]["y_pred"])
    
    if score_name == "Auroc" :
        
        new_row     = pd.Series({"Recall"   : recall,
                                 "F1_score" : f1_score,
                                 "AUROC"    : auc,
                                 "Time"     : round(results_dict[model_name]["time"], 3),
                                }, 
                                name = model_name
                               )
        
    elif score_name == "Fbeta" :
        
        fbeta_score = metrics.fbeta_score     (results_dict[model_name]["y_test"], results_dict[model_name]["y_pred"], beta=2)
 
        new_row     = pd.Series({"Recall"      : recall,
                                 "F1_score"    : f1_score,
                                 "AUROC"       : auc,
                                 "Fbeta_score" : fbeta_score,
                                 "Time"        : round(results_dict[model_name]["time"], 3),
                                }, 
                                name = model_name
                               )

#     if score : 
#         new_row[score_name] = metrics.get_scorer(score)._score_func(results_dict[model_name]["y_test"], 
#                                                                     results_dict[model_name]["y_pred"]
#                                                                    )
        
    if update :
        scores_df = scores_df.append(new_row)
        
    return scores_df


# In[20]:


def plot_roc(y_test, y_pred, model_name):
        
    """ plot roc curve """
        
    plt.figure(figsize=(12, 6))
        
    plt.title('ROC : {}'.format(model_name),
              fontsize   = 15, 
              fontweight = "bold", 
              pad        = 20 
             )

    (fpr, tpr, thr) = metrics.roc_curve(y_test, y_pred)
    
    plt.plot(fpr, tpr)
    
    plt.xlabel('1-Specificity', fontsize=13)
    plt.ylabel('Recall', fontsize=13)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    plt.show()
    
def plot_multi_roc(model_name, results_dict):
        
    """ plot multiple roc curves on the same chart """
        
    fig = plt.figure(figsize = (8, 8))
        
    plt.title('ROC',
              fontsize   = 15, 
              fontweight = "bold", 
              pad        = 20 
             )

    for model_name in results_dict.keys():
        
        (fpr, tpr, thr) = metrics.roc_curve(results_dict[model_name]["y_test"], 
                                            results_dict[model_name]["y_proba"]
                                           )
    
        plt.plot(fpr, tpr, label=model_name)
    
    plt.xlabel('1-Specificity', fontsize=13)
    plt.ylabel('Recall', fontsize=13)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    plt.legend()
    
    plt.show()
        
def get_conf_matrix(model_name, results_dict):
    
    """ get and plot confusion matrix """
    
    labels      = results_dict[model_name]["y_test"].unique().tolist()
    
    conf_mat    = metrics.confusion_matrix(results_dict[model_name]["y_test"], 
                                           results_dict[model_name]["y_pred"], 
                                           labels=labels
                                          )
    
    conf_mat_ds = pd.DataFrame(conf_mat, index=labels, columns=labels)

    conf_mat_ds = conf_mat_ds.rename(index   = {0 : "repaid (0)", 
                                                1 : "default (1)"
                                               },
                                     columns = {0 : "repaid (0)", 
                                                1 : "default (1)"
                                               },
                                    )     

    fig, ax     = plt.subplots(figsize = (8, 6))
    
    sns.heatmap(conf_mat_ds, annot=True, cmap="RdBu_r")
    
    plt.title("Confusion Matrix : {}".format(model_name), 
              fontsize   = 15, 
              fontweight = "bold", 
              pad        = 20
             )   
    
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top') 

    ax.set_xlabel('Predicted Classes', fontsize=14, labelpad=10)
    ax.set_ylabel('True Classes',      fontsize=14, labelpad=10)
    
    plt.tick_params(axis   = 'x',        # changes apply to the x-axis
                    bottom = False,      # ticks along the bottom edge are off
                    top    = False,      # ticks along the top edge are off
                   )
    
    print(metrics.classification_report(results_dict[model_name]["y_test"], 
                                        results_dict[model_name]["y_pred"]
                                       )
         )
    
    return conf_mat_ds


# In[21]:


def get_model_perfs(model_name, results_dict, score_name, scores_df):
    
    """ get model performances """
    
    scores_df = get_scores(model_name, results_dict, score_name, scores_df)
    
    plot_multi_roc(model_name, results_dict)
    get_conf_matrix(model_name, results_dict)
    
    return results_dict, scores_df    


# ## 3. Predict

# In[22]:


def predict_with_dummy(X_train, y_train, X_test, y_test, strategy, results_dict, score_name, scores_df): 

    """ predict target with dummy classifier """
    
    start_time   = timeit.default_timer()
    
    model        = dummy.DummyClassifier(strategy     = strategy, 
                                         random_state = 20
                                        )
    model.fit(X_train, y_train)

    y_pred       = model.predict(X_test)
    y_proba      = model.predict_proba(X_test)[:, 1]
    
    duration     = timeit.default_timer() - start_time
        
    results_dict["Dummy Classifier"] = {"y_test"  : y_test, 
                                        "y_pred"  : y_pred, 
                                        "y_proba" : y_proba, 
                                        "time"    : duration,
                                        "model"   : model
                                       }
    
    (results_dict, 
     scores_df)  = get_model_perfs("Dummy Classifier", results_dict, score_name, scores_df)
    
    return results_dict, scores_df


# In[23]:


def get_best_params_str(model):

    """ get best parameters from model tests """
    
    best_params      = model.best_params_
    best_params_list = []
    
    for para in best_params: 
        
        para_value = model.best_params_[para]
        
        if type(para_value) == int:
            para_value = round(para_value, 4)
            
        best_params_list.append(para.replace('_', ' ') + ' : ' + str(para_value))
        
    return ' - '.join(best_params_list)

def display_cv_perfs (model, model_name, score_name) : 
    
    """ display cross validation performances """
    
    print("\n{} CV Perfs (training set):\n".format(model_name))

    for mean, std, params in zip(model.cv_results_['mean_test_score'], # score moyen
                                 model.cv_results_['std_test_score'],  # écart-type du score
                                 model.cv_results_['params']           # valeur de l'hyperparamètre
                                ):
        
        first_para = list(params.keys())[0]
        
        print("    {} = {:.03f} (+/-{:.03f}) for {} = {:.05f}".format(score_name, mean, std * 2, 
                                                                      first_para, params[first_para],
                                                                     )
             )

    print("")


# In[24]:


def predict_with_cv(X_train, y_train, X_test, y_test, model_dict, results_dict, score_name, scores_df, display_cv=False): 
    
    """ predict with cross validation and stratifier k folds """
    
    start_time    = timeit.default_timer()
    
    if score_name == "Auroc":
        score     = 'roc_auc'
        
    elif score_name == "Fbeta":
        score     = metrics.make_scorer(metrics.fbeta_score, beta=2)
        
    skf           = StratifiedKFold(n_splits     = 10, 
                                    shuffle      = True,
                                    random_state = 20
                                   ).split(X_train, y_train)   
        
    model         = model_selection.GridSearchCV(model_dict["model"],    
                                                 model_dict["params_grid"],              
                                                 cv      = skf,   
                                                 scoring = score,
                                                 n_jobs  = -1
                                                )
        
    model.fit(X_train, y_train)

    y_pred        = model.predict(X_test)
    y_proba       = model.predict_proba(X_test)[:, 1]
    
    duration      = timeit.default_timer() - start_time
    
    best_params   = get_best_params_str(model)
    model_name    = "{} ({})".format (model_dict["model_name"], best_params)
     
    results_dict[model_name] = {"y_test"  : y_test, 
                                "y_pred"  : y_pred, 
                                "y_proba" : y_proba, 
                                "time"    : duration, 
                                "model"   : model
                               }
    
    (results_dict, 
     scores_df)   = get_model_perfs(model_name, results_dict, score_name, scores_df)
      
    print("\n{} Perfs :\n".format(model_name))
    
    print("    => training_set : {} = {:.03f}".format (score_name, model.best_score_))    
    print("    => testing_set  : {} = {:.03f}".format (score_name, scores_df.iloc[-1, -2]))    
    print("")
    
    if display_cv:         
        display_cv_perfs(model, model_dict["model_name"], score_name)
           
    if model_dict['model_name'] == "Random Forest": 
        return results_dict, scores_df, model.best_params_, model
    
    else:
        return results_dict, scores_df, model.best_params_


# In[25]:


def rf_select_vars(X_train, X_test, rf, rf_params, vars_names):

    """ get and plot most importantes variables from Random Forest """
    
    # Get

    imps          = rf.best_estimator_.feature_importances_

    feat_imp      = pd.Series(imps, index=vars_names).sort_values(ascending=False)

    threshold     = min (feat_imp[feat_imp > 1 / X_train.shape[1]][-1], 
                         feat_imp.iloc [:int (X_train.shape [1] * 0.33)][-1]
                        )

    most_imp_vars = feat_imp[feat_imp >= threshold].index.tolist()


    # Plot

    fig, ax       = plt.subplots(figsize=(18, 7))

    feat_imp.plot.bar(ax=ax)

    ax.set_title("Feature importances", 
                 fontsize=15
                )

    ax.axhline(y         = threshold, 
               color     = "red",
               linestyle = "dashed", 
              )

    
    # Train

    X_train_rf = X_train[:, np.where (imps > threshold)[0]]
    X_test_rf  = X_test[:, np.where (imps > threshold)[0]]

    return X_train_rf, X_test_rf, most_imp_vars


# In[26]:


def predict_with_models(X_train, y_train, X_test, y_test, models_dict, score_name, display_cv=False):
        
    """ predict target with grid search, cross validation and stratified k folds on classifiers """
        
    if score_name == "Auroc":
        scores_df   = pd.DataFrame(columns=['Recall', 'F1_score', 'AUROC', 'Time'])
        
    elif score_name == "Fbeta":   
        scores_df   = pd.DataFrame(columns=['Recall', 'F1_score', 'AUROC', 'Fbeta_score', 'Time'])
        
    results_dict    = collections.OrderedDict()
    
        
    for model in models_dict.keys():
        
        model_dict  = models_dict[model]
        
        print("\n> {}\n".format(model))
        
        if model == "Dummy": 
            
            (results_dict,
             scores_df)     = predict_with_dummy(X_train, y_train, X_test, y_test, model_dict["strategy"], 
                                                 results_dict, score_name, scores_df
                                                )
            
        elif model == "Random Forest":
            
            (results_dict, 
             scores_df, 
             rf_params, 
             rf)            = predict_with_cv(X_train, y_train, X_test, y_test, 
                                              model_dict, results_dict, score_name, scores_df, display_cv
                                             )

            (X_train_rf, 
             X_test_rf, 
             rf_vars)       = rf_select_vars(X_train, X_test, rf, params, vars_names) 
            
            select_rf_dict  = {"model_name"   : "Selective Random Forest ({} feats)".format(X_train_rf.shape[1]),
                               "model"        : model_dict["model"],
                               "params_grid"  : {"max_depth" : [rf_params['max_depth']]}, 
                              }      
            
            (results_dict, 
             scores_df, 
             select_rf_params) = predict_with_cv(X_train, y_train, X_test, y_test, 
                                                 select_rf_dict, results_dict, score_name, scores_df, display_cv
                                                )
            
        else:
            
            (results_dict, 
             scores_df, 
             params)    = predict_with_cv(X_train, y_train, X_test, y_test, 
                                          model_dict, results_dict, score_name, scores_df, display_cv
                                         )
            
    return results_dict, scores_df, rf_vars


# # V. Modelisation : Scorer (Auroc & Fbeta)

# In[27]:


def get_scorer_df(scorer_df, scores_df, scorer_name, data_name):
    
    """ get scores dataframe """
    
    if scorer_name == "Fbeta":
        scorer_name += "_score"
    
    if "Initial" in data_name:
        
        scorer_df       = scores_df.reindex(columns=[scorer_name])
        scorer_df       = scorer_df.rename({scorer_name : data_name}, axis=1)
        scorer_df.index = [name.split('(')[0] for name in scorer_df.index.tolist()]

    else:
        
        new_df          = scores_df.reindex(columns=[scorer_name])
        new_df          = new_df.rename({scorer_name : data_name}, axis=1)
        new_df.index    = [name.split('(')[0] for name in new_df.index.tolist()]
        
        scorer_df       = scorer_df.merge(new_df, left_index=True, right_index=True, how='left')
               
    return scorer_df  


# In[28]:


def manual_split(data, train_ids, test_ids):
    
    """ split manually dataset into training and testing set with ids """
    
    train_data = data[data.index.isin(train_ids)]
    test_data  = data[data.index.isin(test_ids)]

    X_train    = train_data.drop(['target'], axis=1).values
    y_train    = train_data['target']

    X_test     = test_data.drop(['target'], axis=1).values
    y_test     = test_data['target']

    std_scaler = preprocessing.StandardScaler().fit(X_train)

    X_train    = std_scaler.transform(X_train)
    X_test     = std_scaler.transform(X_test)
    
    return X_train, y_train, X_test, y_test, std_scaler


# In[29]:


def get_poly_features(data):
    
    """ get polynomiale features """
    
    ext_data_df       = data[['target', 
                              'ext_source_1', 
                              'ext_source_2', 
                              'ext_source_3', 
                              'age', 
                              'work_exp', 
                              'credit_amount'
                             ]
                            ]
    
    ext_target        = ext_data_df.target

    ext_feat_df       = ext_data_df.drop(columns=['credit_amount', 'target'])

    poly_transformer  = PolynomialFeatures(degree=3)
    ext_feat          = poly_transformer.fit_transform(ext_feat_df)
    
    ext_data_df       = pd.DataFrame(ext_feat, 
                                     index   = ext_data_df.index,
                                     columns = poly_transformer.get_feature_names(['ext_source_1', 
                                                                                   'ext_source_2', 
                                                                                   'ext_source_3',  
                                                                                   'age', 
                                                                                   'work_exp',
                                                                                  ]
                                                                                  )
                                    )

    ext_data_df['target'] = ext_target
    
    
    # Correlations
    
    ext_corrs         = ext_data_df.corr()['target'].sort_values(ascending=False)
    ext_corrs.name    = "target_correlation"
    ext_corrs         = ext_corrs.to_frame()
    
    most_corr_var     = ext_corrs.head(15)
    
    most_anticorr_var = ext_corrs.sort_values(by="target_correlation").head(15)
    
    
    # Poly Data
    
    ext_data_df       = ext_data_df.drop(['ext_source_1', 
                                          'ext_source_2', 
                                          'ext_source_3',  
                                          'age', 
                                          'work_exp', 
                                          'target', 
                                          '1'
                                         ], axis=1
                                        )

    ext_data_df       = ext_data_df.loc[:, ~ext_data_df.columns.duplicated()]
    
    return ext_data_df, most_corr_var, most_anticorr_var, poly_transformer

def get_domain_features(data, work_experience_outliers):
    
    """ get domain knowledge features """
    
    domain_data_df                      = pd.DataFrame()

    domain_data_df['work_exp_outiers']  = work_experience_outliers

    domain_data_df['debt_rate']         = data['credit_amount'] / data['income_amount']
    domain_data_df['debt_load']         = data['annuity'] / data['income_amount']
    domain_data_df['credit_term']       = data['credit_amount'] / data['annuity']

    domain_data_df['cover_rate']        = data['credit_amount'] / data['goods_price']

    domain_data_df['work_exp_rate']     = data['work_exp'] / data['age']

    return domain_data_df

def select_features(dataset, model):

    """ get most important features from given model with SelectFromModel"""
    
    X        = dataset.drop("target", axis=1)
    y        = dataset.target

    selector = SelectFromModel(estimator=model).fit(X, y)

    feats    = np.array(X.columns)[selector.get_support()]

    X        = selector.transform(X) 

    dataset  = pd.DataFrame(X, columns=feats, index=dataset.index)

    dataset['target'] = y

    return dataset


# In[30]:


def save_models(results_dict, data_name):
    
    """ save models dictionnary with pickle """
    
    keys         = list(results_dict.keys())[1:]
    models_names = ['_'.join(model_name.lower().split(' (')[0].split(' ')) for model_name in keys]
    
    for name, k in zip(models_names, keys):

        file_name = "results/{}_{}".format(data_name, name)
        
        joblib.dump(results_dict[k]['model'].best_estimator_, file_name)


# # VI. Thresholding

# In[31]:


def get_threshold_scores(fbeta_results, dataset_name, model_name):
    
    """ get thresholds and scores (fbeta score)"""
    
    y_test       = fbeta_results[dataset_name]['results'][model_name]['y_test']
    y_proba      = fbeta_results[dataset_name]['results'][model_name]['y_proba']

    thresholds   = []
    scores       = []

    for threshold in np.arange(0, 1.01, 0.01):

        y_pred   = np.where(y_proba > threshold, 1, 0)
        score    = metrics.fbeta_score(y_test, y_pred, beta=2)

        thresholds.append(round(threshold, 2))
        scores.append(score)
    
    return thresholds, scores

def plot_threshold_scores(fbeta_results, dataset_name, model_name):
            
    """ plot scores = f(threshlods) """
    
    thresholds, scores = get_threshold_scores(fbeta_results, dataset_name, model_name)
    
    best_score         = max(scores)
    best_score_idx     = scores.index(best_score)
    
    best_threshold     = thresholds[best_score_idx]
    
    fig, ax            = plt.subplots(figsize = (14, 6))
    
    plt.title('{} - Fbeta Score'.format(model_name),
              fontsize   = 15, 
              fontweight = "bold", 
              pad        = 20 
             )
    
    plt.plot(thresholds, scores)
    
    plt.vlines(best_threshold, -0.01, best_score, colors='r', linestyles='--', label=str(best_threshold))
    plt.hlines(best_score, 0, best_threshold, colors='r', linestyles='--', label=str(best_score))
    
    xt = ax.get_xticks() 
    xt = np.append(xt, best_threshold)
    
    ax.set_xticks(xt)
    ax.get_xticklabels()[np.where(xt == best_threshold)[0][0]].set_color("red")
    
    yt = ax.get_yticks() 
    yt = np.append(yt, best_score)
    
    ax.set_yticks(yt)
    ax.get_yticklabels()[np.where(yt == best_score)[0][0]].set_color("red")
    
    plt.xlabel('Threshold', fontsize=13)
    plt.ylabel('Fbeta Score', fontsize=13)
    
    plt.xlim([0, 1.05])
    plt.ylim([-0.01, best_score + 0.05])
    
    plt.show()
    
    return best_threshold


# # XI. Dashboard

# ## 1. Exploration Data

# In[32]:


def add_poly_feat(data, poly_transformer):
    
    """ add polynomiale features to dataset """
    
    ext_data_df       = data[['ext_source_1', 
                              'ext_source_2', 
                              'ext_source_3', 
                              'age', 
                              'work_exp', 
                             ]
                            ]
    
    ext_feat          = poly_transformer.fit_transform(ext_data_df)
    
    ext_data_df       = pd.DataFrame(ext_feat, 
                                     index   = ext_data_df.index,
                                     columns = poly_transformer.get_feature_names(['ext_source_1', 
                                                                                   'ext_source_2', 
                                                                                   'ext_source_3',  
                                                                                   'age', 
                                                                                   'work_exp',
                                                                                  ]
                                                                                  )
                                    )

    ext_data_df       = ext_data_df.drop(['ext_source_1', 
                                          'ext_source_2', 
                                          'ext_source_3',  
                                          'age', 
                                          'work_exp', 
                                          '1'
                                         ], axis=1
                                        )

    ext_data_df       = ext_data_df.loc[:, ~ext_data_df.columns.duplicated()]
    
    data              = data.merge(ext_data_df, right_index=True, left_index=True) 
    
    return data

def add_domain_feat(data, work_exp_outliers):
     
    """ add domain knowledge features to dataset """
        
    data['work_exp_outliers'] = work_exp_outliers
    
    data['debt_rate']         = (data['credit_amount'] / data['income_amount'] * 100).round(2)
    data['debt_load']         = (data['annuity'] / data['income_amount'] * 100).round(2)
    data['credit_term']       = (data['credit_amount'] / data['annuity']).round(2)

    data['cover_rate']        = (data['credit_amount'] / data['goods_price'] * 100).round(2)

    data['work_exp_rate']     = (data['work_exp'] / data['age'] * 100).round(2)    

    return data


# In[33]:


def get_exploration_data(raw_data, poly_transformer, cols_sort):
    
    """ get exploration data from raw data"""
    
    # Dataset & Target
    
    exp_data   = raw_data.copy()
    exp_target = exp_data.target
#     exp_data   = exp_data.drop("target", axis=1)
    
    
    # Days Features
    
    exp_data.age                       = (exp_data.age / -365.25).round(2)

    exp_data.credit_time_elapsed       = (exp_data.credit_time_elapsed / -365.25).round(2)
    exp_data.id_published_time_elapsed = (exp_data.id_published_time_elapsed / -365.25).round(2)

    exp_data.last_call_days            = (exp_data.last_call_days / -1).round(2)
    
    work_exp_outliers                  = exp_data.work_exp > 0
    exp_data.work_exp                  = (exp_data.work_exp.replace({365243: np.nan}) / -365.25).round(2)
    
    # Polynomial Features
    
#     exp_data = add_poly_feat(exp_data, poly_transformer)    # no imputing so no polynomial transformer
    
       
    # Domain Features
    
    exp_data = add_domain_feat(exp_data, work_exp_outliers)
    
    
    # Outputs
    
    exp_data = exp_data.reindex(columns=cols_sort)

    
    return exp_data.iloc[:, :-36]


# ## 2. Final Model

# In[34]:


def plot_lgbm_features_importances(light_gbm, data_template_cols, top_nb=40):
    
    """ plot most important features from light GBM model """
    
    feature_imp = pd.DataFrame(sorted(zip(light_gbm.feature_importances_, data_template_cols)), columns=['Value','Feature'])
    
    plt.figure(figsize=(20, 12))

    sns.barplot(x    = "Value", 
                y    = "Feature", 
                data = feature_imp.sort_values(by        = "Value", 
                                               ascending = False
                                              ).head(top_nb),
                palette = "hot"
               )

    plt.title('{} - Top {} Features'.format(model_name, top_nb),
              fontsize   = 20, 
              fontweight = "bold", 
              pad        = 20 
             )

    plt.xlabel("Value", fontsize=13)
    plt.ylabel("Feature", fontsize=13)

    plt.tight_layout()
    plt.show()


# ## 3. Scoring API

# In[35]:


def get_client_dict(client_id, raw_data):
    
    """ get client dictionnary from client id """
    
    client_data       = raw_data[raw_data.index == client_id]
    
    client_data['id'] = client_id

    return client_data.to_dict(orient='records')[0]

def get_client_data(client_dict):
    
    """ get client dataframe from client dictionnary """
    
    client_series      = pd.Series(client_dict)
    client_series.name = client_dict['id']
    
    client_df          = client_series.to_frame().T
    
    return client_df.drop('id', axis=1)


# In[36]:


def encode_client(data, cat_feats, kind, encoder): 

    """ encode categorical variables of a client dataset """
    
    if kind == "Dummy":
        
        data          = pd.get_dummies(data)
        encoder       = None
        
    
    elif kind == "OneHot":   
                
        one_hot_data  = data[cat_feats]
        
        for var in cat_feats:
            
            one_hot_data[var] = one_hot_data[var].str.replace(', ', '/')
            one_hot_data[var] = one_hot_data[var].str.replace(' / ', '/')
            one_hot_data[var] = one_hot_data[var].str.replace(' ', '_')
            one_hot_data[var] = one_hot_data[var].str.upper()
            
        encoded_array = encoder.transform(one_hot_data).toarray()
                
        encoded_data  = pd.DataFrame(encoded_array, 
                                     columns = encoder.get_feature_names_out(), 
                                     index   = data.index
                                    )
        
        data          = data.drop(cat_feats, axis=1)
        data          = data.merge(encoded_data, left_index=True, right_index=True)
        
        
    elif kind == "Label":
              
        if len(cat_feats) == 1:
                    
            results   = encoder.transform(data[[var]])        
            data[var] = list(results.reshape(1, -1)[0].astype(int))
            
        else:
                            
            for var in cat_feats:

                results       = encoder[var].transform.transform(data[[var]])
                data[var]     = list(results.reshape(1, -1)[0].astype(int))

                encoders[var] = encoder
                
            encoder   = encoders.copy()
            
        
    elif kind == "Ordinal":        
        
        ordinal_data  = data[cat_feats]
        encoded_array = encoder.transform(ordinal_data)

        encoded_data  = pd.DataFrame(encoded_array, 
                                     columns = cat_feats, 
                                     index   = data.index
                                    )

        data          = data.drop(cat_feats, axis=1)
        data          = data.merge(encoded_data, left_index=True, right_index=True)
    
    
    return data

def impute_client(data, imputer): 

    """ impute missing values of a client dataset """
    
    target       = data[['target']]
    X            = data.drop("target", axis=1)
    
    cleaned_data = imputer.transform(X)
    
    cleaned_data = pd.DataFrame(cleaned_data, index=X.index, columns=X.columns)   
    cleaned_data = cleaned_data.merge(target, left_index=True, right_index=True)
    
    return cleaned_data

def reverse_encoding(data, cat_feats, encoder):
    
    """ reverse encoding of categorical variables of a client dataset """
    
    categorical_data = data[cat_feats]
    cat_array        = encoder.inverse_transform(categorical_data)
    
    cat_data         = pd.DataFrame(cat_array, 
                                    columns = cat_feats, 
                                    index   = data.index
                                   )
    
    data             = data.drop(cat_feats, axis=1)
    data             = data.merge(cat_data, left_index=True, right_index=True)

    return data


# In[37]:


def standardize_client(client_data, domain_std_scaler):
    
    """ standardize (StandardScaler) the dataset of a client """
    
    X           = client_data.drop('target', axis=1)
    y           = client_data.target
    
    X_std       = domain_std_scaler.transform(X)
    
    client_data = pd.DataFrame(X_std, index=X.index, columns=X.columns)
    
    client_data['target'] = y
    
    return client_data


# In[38]:


def preprocess_client(client_data, data_encoder, simple_imputer, cols_names,
                      ordinal_feats, ordinal_encoder, 
                      one_hot_feats, one_hot_encoder, 
                      poly_transformer, domain_std_scaler):
    
    
    """ preprocess client dataset following the same strategy than the one to train model """
    
    cat_feats           = ordinal_feats + one_hot_feats


    # Cleaning

    (client_data, 
     work_exp_outliers) = clean_time_data(client_data)


     # Imputing

    client_data         = encode_client(client_data, cat_feats, "Ordinal", data_encoder)
    client_data         = impute_client(client_data, simple_imputer)
    client_data         = reverse_encoding(client_data, cat_feats, data_encoder)

    client_data         = client_data.reindex(columns=list(cols_names.values())).drop("id", axis=1)


    # Encoding
    
    client_data         = encode_client(client_data, ordinal_feats, "Ordinal", ordinal_encoder)
    client_data         = encode_client(client_data, one_hot_feats, "OneHot", one_hot_encoder)

    
    # Polynomial Features
    
    client_data         = add_poly_feat(client_data, poly_transformer)
    
       
    # Domain Features
    
    client_data         = add_domain_feat(client_data, work_exp_outliers)
    
    
    # Standardisation
    
    std_client_data     = standardize_client(client_data, domain_std_scaler)
    
    return std_client_data, client_data     


# In[393]:


def predict_client(cleaned_client_data, model):
    
    """ predict repayment probability of a client """

    client_X           = cleaned_client_data.drop('target', axis=1).values
    repaid_proba       = model.predict_proba(client_X)[0][0]
        
    return repaid_proba


# In[403]:


def send_client_data(serveur_url, data_dict):
    
    """ send client data through the body of a request """
    
    resp        = requests.get(serveur_url, 
                                data    = json.dumps(data_dict), 
                                headers = {'Content-type' : 'application/json', 
                                           'Accept'       : 'text/plain'
                                          }
                               )
    
    return resp

def get_request_data(resp, client_id):

    """ recover dataframe from data of the body of a request response """
    
#     predictions_dict      = json.loads(resp.json())
    predictions_dict      = resp.json()

    client_data           = pd.DataFrame(data    = predictions_dict['client_data'], 
                                         columns = [client_id],
                                         index   = predictions_dict['client_cols'],
                                        ).T
    
    std_client_data       = pd.DataFrame(data    = predictions_dict['std_client_data'], 
                                         columns = [client_id],
                                         index   = predictions_dict['std_client_cols'],
                                        ).T

    repaid_proba          = predictions_dict['repaid_proba']
    threshold             = predictions_dict['threshold']
    fbeta_score           = predictions_dict['fbeta_score']
    
    return client_data, std_client_data, repaid_proba, threshold, fbeta_score


# ## 4. Data Visualisation

# In[395]:


def get_client_result(repaid_proba, best_threshold):
    
    """ get client results regarding his repayment probability """
    
    if repaid_proba < best_threshold:
        bar_color      = "red"
        result         = "refused"

    elif repaid_proba >= best_threshold:
        bar_color      = "seagreen"
        result         = "accepted"
        
    return bar_color, result


# In[396]:


def get_gauge(repaid_proba, best_threshold, bar_color):
    
    """ plot the gauge chart of a client regarding his repayment probability """
    
    fig = plt.figure(figsize = (7, 7))

    fig = go.Figure(go.Indicator(domain = {'x': [0, 1], 
                                           'y': [0, 1]
                                          },
                                 value  = repaid_proba,
                                 mode   = "gauge+number+delta",
                                 title  = {'text'      : "Loan Repaid Probability", 
                                           'font'      : {'size': 24}
                                          },
                                 delta  = {'reference' : best_threshold},
                                 gauge  = {'axis'      :  {'range'    : [0, 1]},
                                           'steps'     : [{'range'    : [0, best_threshold], 
                                                           'color'    : "Pink"},
#                                                           {'range'    : [best_threshold*0.85, best_threshold*1.15], 
#                                                            'color'    : "gold"},
                                                          {'range'    : [best_threshold, 1], 
                                                           'color'    : "yellowgreen"}
                                                         ],
                                           'threshold' : {'line'      : {'color': "firebrick", 'width': 4}, 
                                                          'thickness' : 0.75, 
                                                          'value'     : best_threshold
                                                         },
                                           'bar'       : {'color'     : bar_color},
#                                            'bgcolor'   : {"gradient":True,"ranges":{"green":[0,6],"yellow":[6,8],"red":[8,10]}}

                                          }
                                )
                   )

    return fig

def get_shap_values(model, model_data):
    
    """ get shap values """
    
    explainer    = shap.TreeExplainer(model)    
    shap_values  = explainer.shap_values(model_data.drop('target', axis=1).values)
    
    return explainer, shap_values


def get_force_plot(model, model_data, shap_data, client_id):
    
    """ plot the force chart of a client """
    
    (explainer, 
     shap_values)    = get_shap_values(model, model_data)

    client_final_pos = model_data.index.tolist().index(client_id)
    client_shap_pos  = shap_data.index.tolist().index(client_id)
       
    shap_names       = model_data.drop('target', axis=1).columns.tolist()
    
    shap.initjs()
    
    plt.figure(figsize = (7, 7))

    fig = shap.force_plot(explainer.expected_value[1], 
                          shap_values[1][client_final_pos, :], 
                          shap_data.drop('target', axis=1).iloc[client_shap_pos, :], 
                          feature_names=shap_names,
                          plot_cmap="PkYg"
                         )
    
    return fig


# # CODE

# # I. Dataset Analysis

# In[43]:


print("\n> Files :\n")

for file_name in listdir("data/"):
    
    try:
        file       = pd.read_csv("data/" + file_name)
        rows, cols = file.shape
        
    except:
        rows, cols = np.NaN, np.NaN
        
    print("   - {:34.34} : {:8} loans and {:3} features".format(file_name, rows, cols))

print("")


# ## 1. Structure Cleaning

# In[44]:


data = pd.read_csv("data/application_train.csv")


# In[45]:


data       = data.rename(cols_names, axis=1)
data.index = data.id
data       = data.reindex(columns=list(cols_names.values()))
data       = data.drop('id', axis=1)

for var in cat_feats :
    data[var] = data[var].str.title()


# In[46]:


comp = get_data_completion(data, 80)
comp


# In[47]:


plot_na(data, sample=200)


# ## 2. Outputs

# In[48]:


data.to_csv("results/" + "data.csv", sep=',', index=True)


# # II. Preprocessing

# In[49]:


data = pd.read_csv("results/" + "data.csv", sep=',', index_col='id')


# In[50]:


cleaned_data = data.copy()


# ## 1. Work Experience Outliers

# In[51]:


cleaned_data.work_exp.describe().to_frame()


# In[52]:


fig   = cleaned_data.work_exp.plot.hist(bins=300, figsize=(18, 6))
title = plt.title("Work Experience Distribution (in days)", 
                  fontsize   = 15, 
                  fontweight = "bold"
                 )   


# In[53]:


outliers = cleaned_data[cleaned_data.work_exp > 0]
regular  = cleaned_data[~(cleaned_data.index.isin(outliers.index))]

print('\n> Work Experience :\n')
print('    - outliers : {}% default'.format(round(outliers.target.mean() * 100, 2)))
print('    - regular  : {}% default'.format(round(regular.target.mean() * 100, 2)))


# In[54]:


outliers.work_exp.value_counts().to_frame()


# ## 2. Time Variables Cleaning

# In[55]:


cleaned_data, work_exp_outliers = clean_time_data(cleaned_data)


# In[56]:


fig   = cleaned_data.work_exp.plot.hist(bins=35, figsize=(18, 6))
title = plt.title("Work Experience Distribution (in years)", 
                  fontsize   = 15, 
                  fontweight = "bold"
                 )   


# In[57]:


cleaned_data.age.describe().to_frame()


# In[58]:


fig   = cleaned_data.age.plot.hist(bins=49, figsize=(18, 6))
title = plt.title("Age Distribution", 
                  fontsize   = 15, 
                  fontweight = "bold"
                 )   


# In[59]:


cleaned_data = cleaned_data.replace({"Xna" : np.nan})


# ## 3. Categorical Variables Analysis

# In[60]:


cat_ds      = cleaned_data.select_dtypes('object').nunique(dropna=False)
cat_ds.name = 'val_nb'
cat_ds      = cat_ds.to_frame()

cat_ds


# In[61]:


for cat_var in cat_ds.index : 
    plot_pie(cleaned_data, cat_var, others_nb=25, counter=True)


# ## 4. Missing Values

# In[62]:


full_data, encoder = encode_data(cleaned_data, cat_feats, "Ordinal")
full_data, imputer = impute_missing(full_data, "simple")
full_data          = reverse_encoding(full_data, cat_feats, encoder)

full_data          = full_data.reindex(columns=list(cols_names.values())).drop("id", axis=1)


# In[63]:


full_data_comp = get_data_completion(full_data, 80)
full_data_comp


# ## 5. Categorical Variables Encoding

# In[64]:


full_data, ordinal_encoder  = encode_data(full_data, ordinal_feats, "Ordinal")


# In[65]:


full_data, one_hot_encoder  = encode_data(full_data, one_hot_feats, "OneHot")


# In[66]:


full_data


# ## 6. Outputs

# In[67]:


pickle.dump(encoder, open("results/data_encoder.pickle", "wb"))
pickle.dump(imputer, open("results/simple_imputer.pickle", "wb"))
pickle.dump(cols_names, open("results/cols_names.pickle", "wb"))

pickle.dump(ordinal_encoder, open("results/ordinal_encoder.pickle", "wb"))
pickle.dump(one_hot_encoder, open("results/one_hot_encoder.pickle", "wb"))

cleaned_data.to_csv("results/" + "cleaned_data.csv", sep=',', index=True)
full_data.to_csv("results/" + "full_data.csv", sep=',', index=True)


# # III. Analyse Exploratoire

# In[68]:


cleaned_data = pd.read_csv("results/" + "cleaned_data.csv", sep=',', index_col='id')


# ## 1. Target

# In[72]:


get_target_mix(cleaned_data)


# In[73]:


plot_pie(cleaned_data, "target", "Target Mix", 
         others_nb  = None, 
         pre_labels = ['repaid', 'default'], 
         counter    = True, 
         colors     = ['mediumaquamarine', "mistyrose"]
        )


# ## 2. Anti Correlated Variables

# In[69]:


corr      = cleaned_data.corr()['target'].sort_values(ascending=False)
corr.name = "target_correlation"
corr      = corr.to_frame()


# In[70]:


most_corr_var = corr.head(15)
most_corr_var


# In[71]:


most_anticorr_var = corr.sort_values(by="target_correlation").head(15)
most_anticorr_var


# ### a. Age

# In[74]:


plot_kde(cleaned_data, 'age', "Age (years)")


# In[75]:


age_df = cleaned_data[['target', 'age']]
age_df.describe()


# In[76]:


age_df['AGE_RANGE'] = pd.cut(age_df['age'], bins=np.linspace(20, 70, num=11))
age_df              = (age_df.groupby('AGE_RANGE')['target'].mean() * 100).round(2)

age_df.name         = "default_rate"

age_df              = age_df.to_frame()

age_df


# In[77]:


fig = age_df.sort_values(by="default_rate").plot(figsize = (13, 3 + 0.2 * age_df.shape [0]), 
                                                 kind    = 'barh', 
                                                 legend  = False,
                                                 color   = cm.Blues_r(np.linspace (0.2, 0.8, age_df.shape [0]))
                                                )

plt.title('Default Rate by Age Group',
          fontsize   = 15, 
          fontweight = "bold", 
          pad        = 20 
         )

ax  = plt.xlabel("default rate (%)")
ax  = plt.ylabel("age")


# ### b. Employed Days

# In[78]:


plot_kde(cleaned_data, 'work_exp', "Professionnal Experience (years)")


# ### c. External Sources

# In[79]:


ext_data_df   = cleaned_data[['target', 'ext_source_1', 'ext_source_2', 'ext_source_3', 
                              'age', 'work_exp', 'income_amount'
                             ]
                            ]

ext_data_corr = ext_data_df.corr()

plot_corr_heatmap(ext_data_corr, "External Sources & Extra Features")


# In[80]:


# la proba de remboursement d'un client augmente avec EXT_SOURCE_X, l'âge et l'expérience pro


# In[81]:


for ext_var in ['ext_source_1', 'ext_source_2', 'ext_source_3']: 
    
    plot_kde(cleaned_data, ext_var, None)


# In[82]:


# EXT_SOURCE_3 et repaid, mais faible, reste utilse pour le ml


# In[83]:


fig = sns.pairplot(data    = ext_data_df.drop(columns=['income_amount']), 
                   hue     = "target", 
                   corner  = True,
                   palette = ['forestgreen', 'firebrick']
                  )

tit = plt.title("External Sources & Age & Pro Exp Pairs Plot", 
               fontsize = 15, 
               y = 4
               )          


# # V. Modelisation : Tests
Mesures de performances :
    
    => pour la banque : plus grave d'octroyer un prêt à un client qui fera défaut, donc TPR, donc recall
    
    - erreur               = FP + FN
    
    - specificity          = taux de TN
                           = TN / (TN + FP)
                           => capacité à bien classer les négatifs
                           => si élevé, alors on ne manque d'identifier correctement qu'un faible nb de négatifs
                           
                           un modèle qui prédit systématiquement des positifs aura TN = FN = 0, donc recall = 1 ! 
                           
    - recall / sensitivity = taux de TP
                           = TP / (TP + FN)
                           => capacité à bien classer les positifs
                           => si élevé, alors on ne manque d'identifier correctement qu'un faible nb de positifs
                           
                           un modèle qui prédit systématiquement des positifs aura TN = FN = 0, donc recall = 1 ! 
                           
    - precision            = TP / (TP + FP)
                           => erreur sur les prédictions positives
                           => si élevée, alors la plupart des échantilons prédits positifs le sont effectivement
                           
                           un modèle qui prédit un seul TP aura TP = 1, FP = 0, FN = 9999, donc precision = 1 !
                           
    - F-score              = (2 * precision * recall) / (precision + recall)
                           = 2*TP / (2*TP + FP + FN)
               
    - courbe ROC           : x = recall, y = 1-specificity
                             point (0, 0) : specificity = 1
                                            rappel = 0
                                            seuil le plus grand
                                            toutes les observations sont prédites négatives 
                             point (1, 1) : specificity = 0
                                            rappel = 1
                                            seuil le plus petit
                                            toutes les observations sont prédites positives 
                                            
    - AUROC                : classifieur parfait  : ROC   = coin haut gauche
                                                    AUROC = 1
                             classifieur aléaoire : ROC   = diagonale
                                                    AUROC = 0.5
# In[84]:


full_data = pd.read_csv("results/" + "full_data.csv", sep=',', index_col='id')


# In[85]:


scores_df    = pd.DataFrame(columns=['Recall', 'F1_score', 'AUROC', 'Time'])
results_dict = collections.OrderedDict()


# ## 0. Preprocessing

# ### a. All Data

# In[86]:


all_data   = full_data.copy()

vars_names = all_data.columns.tolist()
vars_names.remove('target')

(X_all, 
 y_all, 
 X_all_train, 
 y_all_train, 
 X_all_test, 
 y_all_test, 
 X_all_train_ids, 
 X_all_test_ids, 
 all_std_scaler)   = split_dataset(all_data, "target", 0.3)

all_train_data     = all_data[all_data.index.isin(X_all_train_ids)]
all_test_data      = all_data[all_data.index.isin(X_all_test_ids)]

get_target_mix(all_data, y_all_train, y_all_test)


# ### b. Sampled Data

# In[87]:


(sampled_data, 
 X_s_train, 
 y_s_train)   = sample_train_data(all_data, all_train_data, all_test_data, 0.10)

get_target_mix(sampled_data, y_s_train, y_all_test)


# ## 1. Dummy Classifier

# In[88]:


(results_dict,
 scores_df)     = predict_with_dummy(X_s_train, y_s_train, X_all_test, y_all_test, 
                                     "stratified", results_dict, "Auroc", scores_df
                                    )


# In[89]:


scores_df


# ## 2. KNN

# In[90]:


knn_dict = {"model_name"   : "KNN",
            "model"        : neighbors.KNeighborsClassifier(), 
            "params_grid"  : {'n_neighbors' : [3, 5, 7, 9, 10, 15, 20, 50, 75, 100], 
                             }, 
           }     


# In[91]:


(results_dict, 
 scores_df, 
 knn_params) = predict_with_cv(X_s_train, y_s_train, X_all_test, y_all_test, 
                               knn_dict, results_dict, 'Auroc', scores_df, True
                              )


# In[92]:


scores_df


# ## 3. Logistic Regression

# In[93]:


log_reg_dict = {"model_name"   : "Logistic Regression",
                "model"        : linear_model.LogisticRegression(random_state=20), 
                "params_grid"  : {
#                                   'solver'   : ['lbfgs', 'newton-cg', 'liblinear'],             
                                  'penalty'  : ['l2', 'elasticnet'],                 
                                  'l1_ratio' : np.arange(0, 1.25, 0.25),
                                  'C'        : [100, 75, 50, 25, 10, 1.0, 0.1, 0.01, 0.001, 0.0001]
                                 }, 
               }       


# In[95]:


(results_dict, 
 scores_df, 
 log_reg_params) = predict_with_cv(X_s_train, y_s_train, X_all_test, y_all_test, 
                                   log_reg_dict, results_dict, 'Auroc', scores_df, True
                                  )


# In[96]:


scores_df


# ## 4. Bagging

# In[97]:


bag_dict = {"model_name"   : "Bagging",
            "model"        : BaggingClassifier(random_state=20), 
            "params_grid"  : {'n_estimators' : [10, 50], 
                              'max_samples'  : np.arange (0.2, 1.2, 0.2), 
                             }, 
           }       


# In[98]:


(results_dict, 
 scores_df, 
 bag_params) = predict_with_cv(X_s_train, y_s_train, X_all_test, y_all_test, 
                               bag_dict, results_dict, 'Auroc', scores_df, True
                              )


# In[99]:


scores_df


# ## 5. Random Forest

# ### a. General RF

# In[100]:


rf_dict = {"model_name"   : "Random Forest",
           "model"        : RandomForestClassifier(oob_score    = True, 
                                                   n_estimators = 200, 
                                                   random_state = 20
                                                  ), 
           "params_grid"  : {"max_depth" : [3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 50],  
                            }, 
          }    


# In[101]:


(results_dict, 
 scores_df, 
 rf_params, 
 rf)            = predict_with_cv(X_s_train, y_s_train, X_all_test, y_all_test, 
                                  rf_dict, results_dict, 'Auroc', scores_df, True
                                 )


# In[102]:


scores_df


# ### b. Selective RF

# In[104]:


X_train_rf, X_test_rf, rf_vars = rf_select_vars(X_s_train, X_all_test, rf, rf_params, vars_names) 


# In[105]:


select_rf_dict = {"model_name"   : "Selective Random Forest ({} feats)".format(X_train_rf.shape[1]),
                  "model"        : RandomForestClassifier(oob_score    = True, 
                                                          n_estimators = 200, 
                                                          random_state = 20
                                                         ), 
                  "params_grid"  : {"max_depth" : [rf_params['max_depth']],  
                                   }, 
                 }       


# In[106]:


(results_dict, 
 scores_df, 
 select_rf_params) = predict_with_cv(X_s_train, y_s_train, X_all_test, y_all_test, 
                                     select_rf_dict, results_dict, 'Auroc', scores_df, True
                                    )


# In[107]:


scores_df


# ## 6. XGBoost

# In[108]:


xgb_dict = {"model_name"   : "XG Boost",
            "model"        : GradientBoostingClassifier(n_estimators = 200,
                                                        random_state = 20
                                                      ), 
            "params_grid"  : {"max_depth" : [3, 5, 7],  
                             }, 
           }          


# In[109]:


(results_dict, 
 scores_df, 
 xgboost_params) = predict_with_cv(X_s_train, y_s_train, X_all_test, y_all_test, 
                                   xgb_dict, results_dict, 'Auroc', scores_df, True
                                  )


# In[110]:


scores_df


# ## 7. LigthGBM

# In[111]:


lgb_dict = {"model_name"   : "Light GBM",
            "model"        : lgb.LGBMClassifier(random_state=20), 
            "params_grid"  : {"max_depth" : [3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 50],  
                             }, 
            "score_name"   : 'roc_auc'
           }      


# In[112]:


(results_dict, 
 scores_df, 
 lgb_params) = predict_with_cv(X_s_train, y_s_train, X_all_test, y_all_test, 
                               lgb_dict, results_dict, 'Auroc', scores_df, True
                              )


# In[113]:


scores_df


# ## 8. CatBoost

# In[114]:


# catboost_dict  = {"model_name"   : "Cat Boost",
#                   "model"        : CatBoostClassifier(random_state=20), 
#                   "params_grid"  : {"max_depth" : [3, 7, 15],  
#                                    }, 
#                  }      


# In[115]:


# (results_dict, 
#  scores_df, 
#  catboost_params) = predict_with_cv(X_train, y_train, X_test, y_test, 
#                                     catboost_dict, results_dict, 'Auroc', scores_df, True
#                                    )


# In[116]:


# scores_df.style.format({'Recall'   : '{:.3f}', 
#                         'F1_score' : '{:.3f}', 
#                         'AUROC'    : '{:.3f}', 
#                         'Time'     : '{:.3f}'
#                        }
#                       )


# ## VI. Modelisation : AUROC Score

# In[117]:


score_name = "Auroc"


# In[118]:


all_data   = full_data.copy()

vars_names = all_data.columns.tolist()
vars_names.remove('target')

(X_all, 
 y_all, 
 X_all_train, 
 y_all_train, 
 X_all_test, 
 y_all_test, 
 X_all_train_ids, 
 X_all_test_ids, 
 all_std_scaler)   = split_dataset(all_data, "target", 0.3)

all_train_data     = all_data[all_data.index.isin(X_all_train_ids)]
all_test_data      = all_data[all_data.index.isin(X_all_test_ids)]

get_target_mix(all_data, y_all_train, y_all_test)


# ## 1. Sampled Data

# In[119]:


sampled_data, X_s_train, y_s_train = sample_train_data(all_data, all_train_data, all_test_data, 0.10)

get_target_mix(sampled_data, y_s_train, y_all_test)


# In[120]:


(init_roc_results_dict, 
 init_roc_scores_df, 
 init_roc_rf_vars)      = predict_with_models(X_s_train, y_s_train, X_all_test, y_all_test, 
                                              models_dict, score_name
                                             )


# In[121]:


init_roc_scores_df


# In[122]:


auroc_df = get_scorer_df(None, init_roc_scores_df, "AUROC", "Initial Data ({} obs.)".format(X_s_train.shape[0]))
auroc_df


# ## 2. Resample Data (imbalanced classes)

# In[123]:


resampled_data, X_res_train, y_res_train = resample_data(X_all_train, y_all_train, all_test_data, vars_names)    

get_target_mix(resampled_data, y_res_train, y_all_test)


# In[124]:


(res_roc_results_dict, 
 res_roc_scores_df, 
 res_roc_rf_vars)      = predict_with_models(X_res_train, y_res_train, X_all_test, y_all_test, 
                                             models_dict, score_name
                                            )


# In[125]:


res_roc_scores_df


# In[126]:


auroc_df = get_scorer_df(auroc_df, res_roc_scores_df, "AUROC", "Resample Data ({} obs.)".format(int(X_res_train.shape[0])))
auroc_df


# ## 3. Polynomial Features

# In[127]:


(ext_data_df, 
 most_corr_var, 
 most_anticorr_var, 
 poly_transformer)  = get_poly_features(all_data)


# In[128]:


pickle.dump(poly_transformer, open("results/poly_transformer.pickle", "wb"))


# In[129]:


poly_data           = all_data.drop("target", axis=1).merge(ext_data_df, right_index=True, left_index=True) 
poly_data['target'] = all_data['target']

vars_names          = poly_data.columns.tolist()
vars_names.remove('target')


# In[130]:


(X_poly_train, 
 y_poly_train, 
 X_poly_test, 
 y_poly_test, 
 poly_std_scaler) = manual_split(poly_data, X_all_train_ids, X_all_test_ids)

get_target_mix(poly_data, y_poly_train, y_poly_test)


# In[131]:


poly_test_data = poly_data[poly_data.index.isin(X_all_test_ids)]

poly_data, X_poly_train, y_poly_train = resample_data(X_poly_train, y_poly_train, poly_test_data, vars_names)    

get_target_mix(poly_data, y_poly_train, y_poly_test)


# In[132]:


(poly_roc_results_dict, 
 poly_roc_scores_df, 
 poly_roc_rf_vars)      = predict_with_models(X_poly_train, y_poly_train, X_poly_test, y_poly_test, 
                                              models_dict, score_name
                                             )


# In[133]:


poly_roc_scores_df


# In[134]:


auroc_df = get_scorer_df(auroc_df, poly_roc_scores_df, "AUROC", "Poly Data (+{} feat.)".format(X_poly_train.shape[1] - 
                                                                                               X_res_train.shape[1]
                                                                                              )
                        )

auroc_df


# ## 4. Domain Knowledge Features

# In[135]:


domain_data_df = get_domain_features(all_data, work_exp_outliers)


# In[136]:


domain_data           = all_data.drop("target", axis=1).merge(ext_data_df, right_index=True, left_index=True) 
domain_data           = domain_data.merge(domain_data_df, right_index=True, left_index=True) 
domain_data['target'] = all_data['target']

vars_names            = domain_data.columns.tolist()
vars_names.remove('target')


# In[137]:


pickle.dump(vars_names, open("results/domain_cols_names.pickle", "wb"))


# In[138]:


(X_domain_train, 
 y_domain_train, 
 X_domain_test, 
 y_domain_test, 
 domain_std_scaler) = manual_split(domain_data, X_all_train_ids, X_all_test_ids)

get_target_mix(domain_data, y_domain_train, y_domain_test)


# In[139]:


pickle.dump(domain_std_scaler, open("results/domain_std_scaler.pickle", "wb"))


# In[140]:


domain_test_data = domain_data[domain_data.index.isin(X_all_test_ids)]

domain_data, X_domain_train, y_domain_train = resample_data(X_domain_train, y_domain_train, domain_test_data, vars_names)    

get_target_mix(domain_data, y_domain_train, y_domain_test)


# In[141]:


(domain_roc_results_dict, 
 domain_roc_scores_df, 
 domain_roc_rf_vars)      = predict_with_models(X_domain_train, y_domain_train, X_domain_test, y_domain_test, 
                                                models_dict, score_name
                                               )


# In[142]:


domain_roc_scores_df


# In[143]:


auroc_df = get_scorer_df(auroc_df, domain_roc_scores_df, "AUROC", "Extra Data (+{} feat.)".format(domain_data.shape[1] - 
                                                                                                  poly_data.shape[1]
                                                                                                 )
                        )
auroc_df


# ## 5. Random Forest Selection

# In[144]:


rf_data           = all_data.drop("target", axis=1).merge(ext_data_df, right_index=True, left_index=True) 
rf_data           = rf_data.merge(domain_data_df, right_index=True, left_index=True) 
rf_data           = rf_data.reindex(columns=domain_roc_rf_vars)

rf_data['target'] = all_data['target']

vars_names        = rf_data.columns.tolist()
vars_names.remove('target')


# In[145]:


(X_rf_train, 
 y_rf_train, 
 X_rf_test, 
 y_rf_test, 
 rf_scaler)  = manual_split(rf_data, X_all_train_ids, X_all_test_ids)

get_target_mix(rf_data, y_rf_train, y_rf_test)


# In[146]:


rf_test_data = rf_data[rf_data.index.isin(X_all_test_ids)]

rf_data, X_rf_train, y_rf_train = resample_data(X_rf_train, y_rf_train, rf_test_data, vars_names)    

get_target_mix(rf_data, y_rf_train, y_rf_test)


# In[147]:


(rf_roc_results_dict, 
 rf_roc_scores_df, 
 rf_roc_rf_vars)      = predict_with_models(X_rf_train, y_rf_train, X_rf_test, y_rf_test, 
                                            models_dict, score_name
                                           )


# In[148]:


rf_roc_scores_df


# In[149]:


auroc_df = get_scorer_df(auroc_df, rf_roc_scores_df, "AUROC", "RF Data ({} feat.)".format(rf_data.shape[1]))
auroc_df


# ## 6. Logistic Regression Features Selection (with trained model)

# In[150]:


reg_log_data           = all_data.merge(ext_data_df, right_index=True, left_index=True) 
reg_log_data           = reg_log_data.merge(domain_data_df, right_index=True, left_index=True) 
reg_log_data           = select_features(reg_log_data, LogisticRegression())

vars_names   = reg_log_data.columns.tolist()
vars_names.remove('target')


# In[151]:


(X_reg_log_train, 
 y_reg_log_train, 
 X_reg_log_test, 
 y_reg_log_test,
 reg_log_scaler)  = manual_split(reg_log_data, X_all_train_ids, X_all_test_ids)

get_target_mix(reg_log_data, y_reg_log_train, y_reg_log_test)


# In[152]:


reg_log_test_data = reg_log_data[reg_log_data.index.isin(X_all_test_ids)]

reg_log_data, X_reg_log_train, y_reg_log_train = resample_data(X_reg_log_train, y_reg_log_train, reg_log_test_data, vars_names)

get_target_mix(reg_log_data, y_reg_log_train, y_reg_log_test)


# In[153]:


(reg_log_roc_results_dict, 
 reg_log_roc_scores_df, 
 reg_log_roc_rf_vars)      = predict_with_models(X_reg_log_train, y_reg_log_train, X_reg_log_test, y_reg_log_test, 
                                                 models_dict, score_name
                                                )


# In[154]:


reg_log_roc_scores_df


# In[155]:


auroc_df = get_scorer_df(auroc_df, reg_log_roc_scores_df, "AUROC", "Reg Log Data ({} feat.)".format(reg_log_data.shape[1]))
auroc_df


# ## 7. Results

# In[156]:


auroc_results = {'init_data'         : {'results' : init_roc_results_dict.copy(), 
                                        'scores'  : init_roc_scores_df, 
                                       }, 
                 'resampled_data'    : {'results' : res_roc_results_dict.copy(), 
                                        'scores'  : res_roc_scores_df, 
                                       }, 
                 'poly_feat'         : {'results' : poly_roc_results_dict.copy(), 
                                        'scores'  : poly_roc_scores_df, 
                                       }, 
                 'domain_feat'       : {'results' : domain_roc_results_dict.copy(), 
                                        'scores'  : domain_roc_scores_df, 
                                       }, 
                 'rf_selection'      : {'results' : rf_roc_results_dict.copy(), 
                                        'scores'  : rf_roc_scores_df, 
                                       }, 
                 'reg_log_selection' : {'results' : reg_log_roc_results_dict.copy(), 
                                        'scores'  : reg_log_roc_scores_df, 
                                       }, 
                 'auroc_scores'      : auroc_df
                }


# In[161]:


# for data_name in auroc_results.keys():
        
#     print("\n> Data : {}\n".format(data_name))
    
#     for model_name in list(auroc_results[data_name]['results'].keys())[1:]:
        
#         print("    - {}".format(model_name))
        
#         try:
#             auroc_results[data_name]['results'][model_name]['model'] = auroc_results[data_name]['results'][model_name]['model'].best_estimator_
            
#         except AttributeError:
#             print("error")


# In[162]:


pickle.dump(auroc_results, open("results/auroc_results.pickle", "wb"))


# # VII. Modelisation : F-Beta Score

# In[163]:


score_name = "Fbeta"


# ## 1. Sampled Data

# In[164]:


vars_names = all_data.columns.tolist()
vars_names.remove('target')


# In[165]:


(init_fb_results_dict, 
 init_fb_scores_df, 
 init_fb_rf_vars)       = predict_with_models(X_s_train, y_s_train, X_all_test, y_all_test, 
                                              models_dict, score_name
                                             )


# In[166]:


init_fb_scores_df


# In[167]:


fbeta_df = get_scorer_df(None, init_fb_scores_df, "Fbeta", "Initial Data ({} obs.)".format(X_s_train.shape[0]))
fbeta_df


# ## 2. Resample Data (imbalanced classes)

# In[168]:


(res_fb_results_dict, 
 res_fb_scores_df, 
 res_fb_rf_vars)        = predict_with_models(X_res_train, y_res_train, X_all_test, y_all_test, 
                                              models_dict, score_name
                                             )


# In[169]:


res_fb_scores_df


# In[170]:


fbeta_df = get_scorer_df(fbeta_df, res_fb_scores_df, "Fbeta", "Resample Data ({} obs.)".format(int(X_res_train.shape[0])))
fbeta_df


# ## 3. Polynomial Features

# In[171]:


vars_names          = poly_data.columns.tolist()
vars_names.remove('target')


# In[172]:


(poly_fb_results_dict, 
 poly_fb_scores_df, 
 poly_fb_rf_vars)       = predict_with_models(X_poly_train, y_poly_train, X_poly_test, y_poly_test, 
                                              models_dict, score_name
                                             )


# In[173]:


poly_fb_scores_df


# In[174]:


fbeta_df = get_scorer_df(fbeta_df, poly_fb_scores_df, "Fbeta", "Poly Data (+{} feat.)".format(X_poly_train.shape[1] - 
                                                                                              X_res_train.shape[1]
                                                                                             )
                        )

fbeta_df


# ## 4. Domain Knowledge Features

# In[175]:


vars_names            = domain_data.columns.tolist()
vars_names.remove('target')


# In[176]:


(domain_fb_results_dict, 
 domain_fb_scores_df, 
 domain_fb_rf_vars)       = predict_with_models(X_domain_train, y_domain_train, X_domain_test, y_domain_test, 
                                                models_dict, score_name
                                               )


# In[177]:


domain_fb_scores_df


# In[178]:


fbeta_df = get_scorer_df(fbeta_df, domain_fb_scores_df, "Fbeta", "Extra Data (+{} feat.)".format(domain_data.shape[1] - 
                                                                                                 poly_data.shape[1]
                                                                                                )
                        )
fbeta_df


# ## 5. Random Forest Selection

# In[179]:


vars_names        = rf_data.columns.tolist()
vars_names.remove('target')


# In[180]:


(rf_fb_results_dict, 
 rf_fb_scores_df, 
 rf_fb_rf_vars)      = predict_with_models(X_rf_train, y_rf_train, X_rf_test, y_rf_test, 
                                            models_dict, score_name
                                           )


# In[181]:


rf_fb_scores_df


# In[182]:


fbeta_df = get_scorer_df(fbeta_df, rf_fb_scores_df, "Fbeta", "RF Data ({} feat.)".format(rf_data.shape[1]))
fbeta_df


# ## 6. Logistic Regression Features Selection (with trained model)

# In[183]:


vars_names   = reg_log_data.columns.tolist()
vars_names.remove('target')


# In[184]:


(reg_log_fb_results_dict, 
 reg_log_fb_scores_df, 
 reg_log_fb_rf_vars)       = predict_with_models(X_reg_log_train, y_reg_log_train, X_reg_log_test, y_reg_log_test, 
                                                 models_dict, score_name
                                                )


# In[185]:


reg_log_fb_scores_df


# In[186]:


fbeta_df = get_scorer_df(fbeta_df, reg_log_fb_scores_df, "Fbeta", "Reg Log Data ({} feat.)".format(reg_log_data.shape[1]))
fbeta_df


# ## 7. Save Results

# In[265]:


fbeta_results = {'init_data'         : {'results' : init_fb_results_dict.copy(), 
                                        'scores'  : init_fb_scores_df, 
                                       }, 
                 'resampled_data'    : {'results' : res_fb_results_dict.copy(), 
                                        'scores'  : res_fb_scores_df, 
                                       }, 
                 'poly_feat'         : {'results' : poly_fb_results_dict.copy(), 
                                        'scores'  : poly_fb_scores_df, 
                                       }, 
                 'domain_feat'       : {'results' : domain_fb_results_dict.copy(), 
                                        'scores'  : domain_fb_scores_df, 
                                       }, 
                 'rf_selection'      : {'results' : rf_fb_results_dict.copy(), 
                                        'scores'  : rf_fb_scores_df, 
                                       }, 
                 'reg_log_selection' : {'results' : reg_log_fb_results_dict.copy(), 
                                        'scores'  : reg_log_fb_scores_df, 
                                       }, 
                 'fbeta_scores'      : fbeta_df
                }


# In[273]:


# for data_name in fbeta_results.keys():
        
#     print("\n> Data : {}\n".format(data_name))
    
#     for model_name in list(fbeta_results[data_name]['results'].keys())[1:]:
        
#         print("    - {}".format(model_name))
        
#         try:
#             fbeta_results[data_name]['results'][model_name]['model'] = fbeta_results[data_name]['results'][model_name]['model'].best_estimator_
            
#         except AttributeError:
#             print("error")


# In[274]:


pickle.dump(fbeta_results, open("results/fbeta_results.pickle", "wb"))


# # VIII. Thresholding

# In[307]:


fbeta_results          = pickle.load(open("results/fbeta_results.pickle", "rb"))


# In[309]:


thresholds_dict = {}

for dataset_name in fbeta_results.keys():
     
    print("\n> {}\n".format(dataset_name.upper()))
    
    fb_results_dict = fbeta_results[dataset_name]['results']
    
    dataset_dict    = {}

    for model_name in fb_results_dict.keys():
        
        dataset_dict[model_name]  = plot_threshold_scores(fbeta_results, dataset_name, model_name)    
        
    thresholds_dict[dataset_name] = dataset_dict  


# In[312]:


pickle.dump(thresholds_dict, open("results/thresholds_dict.pickle", "wb"))


# # IX. Dashboard 

# In[197]:


auroc_df.style.highlight_min (color = 'pink', axis = 0
                                       ).highlight_max (color = 'lightgreen', axis = 0)


# In[198]:


fbeta_df.style.highlight_min (color = 'pink', axis = 0
                                       ).highlight_max (color = 'lightgreen', axis = 0)


# ## 1. Raw Data (resample)

# In[247]:


raw_data           = pd.read_csv("results/" + "data.csv", sep=',', index_col='id')


# In[248]:


rus                           = RandomUnderSampler(sampling_strategy = 1, 
                                                   random_state      = 20
                                                  )

(X_data, 
 y_data)                      = rus.fit_resample(raw_data.drop('target', axis=1), raw_data.target)

resampled_raw_data            = X_data
resampled_raw_data.index      = rus.sample_indices_
resampled_raw_data.index.name = "id"

resampled_raw_data['target']  = y_data.values


# In[252]:


resampled_raw_data.to_csv("results/" + "resampled_data.csv", sep=',', index=True)


# ## 2. Exploration Data (no encoding, no imputing, no standardization) (resample)

# In[253]:


resampled_exp_data   = get_exploration_data(resampled_raw_data, poly_transformer, cols_sort)


# In[254]:


resampled_exp_data.to_csv("results/" + "resampled_exp_data.csv", sep=',', index=True)


# ## 3. Exploration Data (no encoding, no imputing, no standardization)

# In[255]:


exp_data   = get_exploration_data(raw_data, poly_transformer, cols_sort)


# In[256]:


exp_data.to_csv("results/" + "exp_data.csv", sep=',', index=True)


# ## 4. Models Dict

# In[313]:


thresholds_dict    = pickle.load(open("results/thresholds_dict.pickle", "rb"))
fbeta_results      = pickle.load(open("results/fbeta_results.pickle", "rb"))


# In[315]:


thresholds_dict['Raw Data']                           = thresholds_dict.pop('init_data')
thresholds_dict['Resampled Data']                     = thresholds_dict.pop('resampled_data')
thresholds_dict['Polynomial Data']                    = thresholds_dict.pop('poly_feat')
thresholds_dict['Domain Data']                        = thresholds_dict.pop('domain_feat')
thresholds_dict['Random Forest Selection Data']       = thresholds_dict.pop('rf_selection')
thresholds_dict['Logistic Regression Selection Data'] = thresholds_dict.pop('reg_log_selection')


# In[316]:


fbeta_results['Raw Data']                           = fbeta_results.pop('init_data')
fbeta_results['Resampled Data']                     = fbeta_results.pop('resampled_data')
fbeta_results['Polynomial Data']                    = fbeta_results.pop('poly_feat')
fbeta_results['Domain Data']                        = fbeta_results.pop('domain_feat')
fbeta_results['Random Forest Selection Data']       = fbeta_results.pop('rf_selection')
fbeta_results['Logistic Regression Selection Data'] = fbeta_results.pop('reg_log_selection')

del fbeta_results['fbeta_scores']    


# In[326]:


models_dict = {}

for dataset_name in fbeta_results.keys():
     
    print("\n> {}\n".format(dataset_name.upper()))
    
    fb_results_dict = fbeta_results[dataset_name]['results']
    
    dataset_dict    = {}

    for model_name in fb_results_dict.keys():

        model_title = model_name.split(' (')[0]
        
        print('    - {}'.format(model_title))
        
        thresholds, scores        = get_threshold_scores(fbeta_results, dataset_name, model_name)

        best_score                = max(scores)
        best_score_idx            = scores.index(best_score)

        best_threshold            = thresholds[best_score_idx]
    
        dataset_dict[model_title] = {'model'       : fb_results_dict[model_name]['model'], 
                                     'threshold'   : best_threshold,
                                     'fbeta_score' : best_score,
                                    }
        
    models_dict[dataset_name]     = dataset_dict


# In[327]:


pickle.dump(models_dict, open("results/models_dict.pickle", "wb"))


# ## 5. Domain Data Models Dict

# In[330]:


models_dict = pickle.load(open("results/models_dict.pickle", "rb"))


# In[333]:


models_names = ['KNN', 'Logistic Regression', 'Bagging', 'Random Forest', 'XG Boost', 'Light GBM']


# In[337]:


for model_name in models_names:
    
    domain_model_dict = models_dict['Domain Data'][model_name]
    pickle.dump(domain_model_dict, open("results/domain_{}_dict.pickle".format(model_name.replace(' ', '_').lower()), "wb"))


# ## 6. Final Model

# In[379]:


dataset_name = "Domain Data"
model_name   = "Light GBM"


# In[381]:


domain_cols_names = pickle.load(open("results/domain_cols_names.pickle", "rb"))

model_dict        = pickle.load(open("results/domain_{}_dict.pickle".format(model_name.replace(' ', '_').lower()), "rb"))  
model             = model_dict['model']  


# In[382]:


plot_lgbm_features_importances(model, domain_cols_names, top_nb=40)


# ## 7. Client Data

# In[384]:


resampled_raw_data = pd.read_csv("results/" + "resampled_data.csv", sep=',', index_col='id')


# In[416]:


client_id          = 138483


# In[417]:


client_dict        = get_client_dict(client_id, resampled_raw_data)
client_data        = get_client_data(client_dict)


# ## 8. Client Preprocessing

# In[418]:


data_encoder       = pickle.load(open("results/data_encoder.pickle", "rb"))
simple_imputer     = pickle.load(open("results/simple_imputer.pickle", "rb"))
cols_names         = pickle.load(open("results/cols_names.pickle", "rb"))

ordinal_encoder    = pickle.load(open("results/ordinal_encoder.pickle", "rb"))
one_hot_encoder    = pickle.load(open("results/one_hot_encoder.pickle", "rb"))

poly_transformer   = pickle.load(open("results/poly_transformer.pickle", "rb"))

domain_std_scaler  = pickle.load(open("results/domain_std_scaler.pickle", "rb"))


# In[419]:


(std_client_data, 
 client_data)      = preprocess_client(client_data, data_encoder, simple_imputer, cols_names,
                                       ordinal_feats, ordinal_encoder, 
                                       one_hot_feats, one_hot_encoder, 
                                       poly_transformer, domain_std_scaler
                                      )


# ## 9. Client Prediction

# In[397]:


repaid_proba       = predict_client(std_client_data, model)


# ## 10. Data Visualisation

# In[401]:


(bar_color, 
 result)    = get_client_result(repaid_proba, best_threshold)

gauge       = get_gauge(repaid_proba, best_threshold, bar_color)
gauge.show()


# In[402]:


force = get_force_plot(model, std_client_data, client_data, client_data.index[0])
force


# # X. Scoring API

# ## 3. Send Data

# In[450]:


client_id   = 92167

client_dict = get_client_dict(client_id, resampled_raw_data)
model_name  = 'KNN'

data_dict   = {'client'     : client_dict, 
               'model_name' : model_name
              }

model_dict  = pickle.load(open("results/domain_{}_dict.pickle".format(model_name.replace(' ', '_').lower()), "rb"))


# In[451]:


resp = send_client_data("http://127.0.0.1:8000/predict", data_dict)
resp.status_code


# In[453]:


resp = send_client_data("https://dashloan.herokuapp.com/predict", data_dict)
resp.status_code


# ## 2. Get Response

# In[454]:


(client_data, 
 std_client_data, 
 repaid_proba, 
 threshold, 
 fbeta_score) = get_request_data(resp, client_id)


# In[455]:


repaid_proba


# ## 3. API

# In[134]:


# web: gunicorn -w 4 -k uvicorn.workers.UvicornWorker P7_03_api:app

       
    
### LIBRAIRIES
        
    
    
from   fastapi           import FastAPI, Request
import uvicorn

import pandas                  as pd
import numpy                   as np
#import matplotlib.pyplot       as plt

#import shap
#import streamlit.components.v1 as components

import pickle
import json



### PARAMETERS



one_hot_feats = ["gender",
                 "family_status",
                 "type_suite",
                 "education_type",
                 "work_type",
                 "organization_type",
                 "income_type",
                 "appr_process_weekday",
                 "housing_type",
                 "fondkapremont_mode",
                 "house_type_mode",
                 "walls_material_mode",
                 "emergency_state_mode",
                ]   

ordinal_feats = ['contract', 
                 'own_car', 
                 'own_realty'
                ]


### FUNCTIONS



# DataFrame

def get_client_data(client_dict):
    
    """ get client dataframe from client dictionnary """
    
    client_series      = pd.Series(client_dict)
    client_series.name = client_dict['id']
    
    client_df          = client_series.to_frame().T
    
    return client_df.drop('id', axis=1)


# Preprocessing

def clean_time_data(data):
    
    """ clean time variables """
    
    work_exp_outliers                = data.work_exp > 0
    data.work_exp                    = data.work_exp.replace({365243: np.nan}) / -365.25
    
    data.age                         = data.age / -365.25
    
    data.credit_time_elapsed         = data.credit_time_elapsed / -365.25
    data.id_published_time_elapsed   = data.id_published_time_elapsed / -365.25

    data.last_call_days              = data.last_call_days / -1
    
    return data, work_exp_outliers

def encode_client(data, cat_feats, kind, encoder): 

    """ encode categorical variables of a client dataset """
    
    if kind == "Dummy":
        
        data          = pd.get_dummies(data)
        encoder       = None
        
    
    elif kind == "OneHot":   
                
        one_hot_data  = data[cat_feats]
        
        for var in cat_feats:
            
            one_hot_data[var] = one_hot_data[var].str.replace(', ', '/')
            one_hot_data[var] = one_hot_data[var].str.replace(' / ', '/')
            one_hot_data[var] = one_hot_data[var].str.replace(' ', '_')
            one_hot_data[var] = one_hot_data[var].str.upper()
            
        encoded_array = encoder.transform(one_hot_data).toarray()
                
        encoded_data  = pd.DataFrame(encoded_array, 
                                     columns = encoder.get_feature_names_out(), 
                                     index   = data.index
                                    )
        
        data          = data.drop(cat_feats, axis=1)
        data          = data.merge(encoded_data, left_index=True, right_index=True)
        
        
    elif kind == "Label":
              
        if len(cat_feats) == 1:
                    
            results   = encoder.transform(data[[var]])        
            data[var] = list(results.reshape(1, -1)[0].astype(int))
            
        else:
                            
            for var in cat_feats:

                results       = encoder[var].transform.transform(data[[var]])
                data[var]     = list(results.reshape(1, -1)[0].astype(int))

                encoders[var] = encoder
                
            encoder   = encoders.copy()
            
        
    elif kind == "Ordinal":        
        
        ordinal_data  = data[cat_feats]
        encoded_array = encoder.transform(ordinal_data)

        encoded_data  = pd.DataFrame(encoded_array, 
                                     columns = cat_feats, 
                                     index   = data.index
                                    )

        data          = data.drop(cat_feats, axis=1)
        data          = data.merge(encoded_data, left_index=True, right_index=True)
    
    
    return data

def impute_client(data, imputer): 

    """ impute missing values of a client dataset """
    
    target       = data[['target']]
    X            = data.drop("target", axis=1)
    
    cleaned_data = imputer.transform(X)
    
    cleaned_data = pd.DataFrame(cleaned_data, index=X.index, columns=X.columns)   
    cleaned_data = cleaned_data.merge(target, left_index=True, right_index=True)
    
    return cleaned_data

def reverse_encoding(data, cat_feats, encoder):
    
    """ reverse encoding and add decoded categorical variables to the dataset """
    
    categorical_data = data[cat_feats]
    cat_array        = encoder.inverse_transform(categorical_data)
    
    cat_data         = pd.DataFrame(cat_array, 
                                    columns = cat_feats, 
                                    index   = data.index
                                   )
    
    data             = data.drop(cat_feats, axis=1)
    data             = data.merge(cat_data, left_index=True, right_index=True)

    return data


def add_poly_feat(data, poly_transformer):
    
    """ add polynomiale features to dataset """
    
    ext_data_df       = data[['ext_source_1', 
                              'ext_source_2', 
                              'ext_source_3', 
                              'age', 
                              'work_exp', 
                             ]
                            ]
    
    ext_feat          = poly_transformer.fit_transform(ext_data_df)
    
    ext_data_df       = pd.DataFrame(ext_feat, 
                                     index   = ext_data_df.index,
                                     columns = poly_transformer.get_feature_names(['ext_source_1', 
                                                                                   'ext_source_2', 
                                                                                   'ext_source_3',  
                                                                                   'age', 
                                                                                   'work_exp',
                                                                                  ]
                                                                                  )
                                    )

    ext_data_df       = ext_data_df.drop(['ext_source_1', 
                                          'ext_source_2', 
                                          'ext_source_3',  
                                          'age', 
                                          'work_exp', 
                                          '1'
                                         ], axis=1
                                        )

    ext_data_df       = ext_data_df.loc[:, ~ext_data_df.columns.duplicated()]
    
    data              = data.merge(ext_data_df, right_index=True, left_index=True) 
    
    return data

def add_domain_feat(data, work_exp_outliers):
     
    """ add domain knowledge features to dataset """
    
    data['work_exp_outliers'] = work_exp_outliers
    
    data['debt_rate']         = (data['credit_amount'] / data['income_amount'] * 100).round(2)
    data['debt_load']         = (data['annuity'] / data['income_amount'] * 100).round(2)
    data['credit_term']       = (data['credit_amount'] / data['annuity']).round(2)

    data['cover_rate']        = (data['credit_amount'] / data['goods_price'] * 100).round(2)

    data['work_exp_rate']     = (data['work_exp'] / data['age'] * 100).round(2)    

    return data


def standardize_client(client_data, domain_std_scaler):
    
    """ standardize (StandardScaler) the dataset of a client """
    
    X           = client_data.drop('target', axis=1)
    y           = client_data.target
    
    X_std       = domain_std_scaler.transform(X)
    
    client_data = pd.DataFrame(X_std, index=X.index, columns=X.columns)
    
    client_data['target'] = y
    
    return client_data


def preprocess_client(client_data, data_encoder, simple_imputer, cols_names,
                      ordinal_feats, ordinal_encoder, 
                      one_hot_feats, one_hot_encoder, 
                      poly_transformer, domain_std_scaler):
    
    """ preprocess client dataset following the same strategy than the one to train model """
    
    cat_feats           = ordinal_feats + one_hot_feats


    # Cleaning

    (client_data, 
     work_exp_outliers) = clean_time_data(client_data)


     # Imputing

    client_data         = encode_client(client_data, cat_feats, "Ordinal", data_encoder)
    client_data         = impute_client(client_data, simple_imputer)
    client_data         = reverse_encoding(client_data, cat_feats, data_encoder)

    client_data         = client_data.reindex(columns=list(cols_names.values())).drop("id", axis=1)


    # Encoding
    
    client_data         = encode_client(client_data, ordinal_feats, "Ordinal", ordinal_encoder)
    client_data         = encode_client(client_data, one_hot_feats, "OneHot", one_hot_encoder)

    
    # Polynomial Features
    
    client_data         = add_poly_feat(client_data, poly_transformer)
    
       
    # Domain Features
    
    client_data         = add_domain_feat(client_data, work_exp_outliers)
    
    
    # Standardisation
    
    std_client_data     = standardize_client(client_data, domain_std_scaler)
    
    return std_client_data, client_data     


# Prediction

def predict_client(cleaned_client_data, model):
    
    """ predict repayment probability of a client """

    client_X           = cleaned_client_data.drop('target', axis=1).values
    repaid_proba       = model.predict_proba(client_X)[0][0]
        
    return repaid_proba


# Data Visualisation

#def get_shap_values(model, model_data):
    
#    """ get shap values """

#    explainer    = shap.TreeExplainer(model)    
#    shap_values  = explainer.shap_values(model_data.drop('target', axis=1).values)
    
#    return explainer, shap_values

#def get_force_plot(explainer, shap_values, model_data, shap_data, client_id):
    
#    """ plot the force chart of a client """

#    client_final_pos = model_data.index.tolist().index(client_id)
#    client_shap_pos  = shap_data.index.tolist().index(client_id)
       
#    shap_names       = model_data.drop('target', axis=1).columns.tolist()
    
#    shap.initjs()
    
#    plt.figure(figsize = (7, 7))

#    fig = shap.force_plot(explainer.expected_value[1], 
#                          shap_values[1][client_final_pos, :], 
#                          shap_data.drop('target', axis=1).iloc[client_shap_pos, :], 
#                          feature_names=shap_names,
#                          plot_cmap="PkYg"
#                         )
    
#    return fig

#def plot_shap(plot, height=None):

#    """ embed shap chart in html container """

#    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"

#    return components.html(shap_html, height=height)



### API



app = FastAPI()


@app.get("/predict")
async def predict(request: Request):

    """ get client data from request, preprocess it, and predict repayment probability of the client """

    # Get Data

    data_dict         = await request.json()


    # Get Client Data

    client_dict       = data_dict["client"]
    client_data       = get_client_data(client_dict)


    # Get Model

    model_name        = data_dict["model_name"]
    model_dict        = pickle.load(open("modelisation/domain_{}_dict.pickle".format(model_name.replace(' ', '_').lower()), "rb"))
    
    model             = model_dict['model']  


    # Get Preprocessing Elements

    cols_names        = pickle.load(open("preprocessing/cols_names.pickle", "rb"))

    data_encoder      = pickle.load(open("preprocessing/data_encoder.pickle", "rb"))
    simple_imputer    = pickle.load(open("preprocessing/simple_imputer.pickle", "rb"))

    ordinal_encoder   = pickle.load(open("preprocessing/ordinal_encoder.pickle", "rb"))
    one_hot_encoder   = pickle.load(open("preprocessing/one_hot_encoder.pickle", "rb"))

    poly_transformer  = pickle.load(open("preprocessing/poly_transformer.pickle", "rb"))

    domain_std_scaler = pickle.load(open("preprocessing/domain_std_scaler.pickle", "rb"))

    
    # Preprocess Client Data   

    (std_client_data, 
     client_data)     = preprocess_client(client_data, data_encoder, simple_imputer, cols_names,
                                          ordinal_feats, ordinal_encoder, 
                                          one_hot_feats, one_hot_encoder, 
                                          poly_transformer, domain_std_scaler
                                         )
    

    # Predict Client Score

    repaid_proba      = predict_client(std_client_data, model)
    
    force             = None

    #if model_name == 'Light GBM':

    #    (explainer, 
    #     shap_values) = get_shap_values(model, std_client_data)

    #    force         = get_force_plot(explainer, shap_values, std_client_data, client_data, client_data.index[0])
    #    force         = plot_shap(force)

    predictions_dict  = {"client_data"     : client_data.values[0].tolist(), 
                         "client_cols"     : client_data.columns.to_list(),
                         "std_client_data" : std_client_data.values[0].tolist(), 
                         "std_client_cols" : std_client_data.columns.to_list(),
                         "repaid_proba"    : repaid_proba, 
                         "threshold"       : model_dict['threshold'], 
                         "fbeta_score"     : model_dict['fbeta_score'], 
                         "force"           : "test",
                        }
    
    return predictions_dict

