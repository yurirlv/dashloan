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



