### SCORING DASHBOARD


# LIBRAIRIES


import numpy                           as np
import pandas                          as pd
import matplotlib.pyplot               as plt
import seaborn                         as sns

#from   matplotlib                      import cm

import shap
import plotly.graph_objects            as go


#from   sklearn.impute                  import SimpleImputer, IterativeImputer

#from   sklearn.preprocessing           import LabelEncoder, OneHotEncoder, OrdinalEncoder
#from   sklearn.preprocessing           import PolynomialFeatures

#from   sklearn.model_selection         import train_test_split

#from   sklearn                         import preprocessing

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


import ast
import time

import datetime
import re

import streamlit                       as st
import streamlit.components.v1         as components



### PARAMETERS

target              = []
age                 = ()    
gender              = []
edu                 = []
family_status       = []
family_nb           = ()
children_nb         = ()
work                = []
orga_type           = []
work_exp            = ()
income_amount       = ()
income_type         = []
contract            = []
credit_amount       = ()
goods_price         = ()
cover_rate          = ()
debt_rate           = ()
debt_load           = ()
annuity             = ()
credit_term         = ()
credit_time         = ()

best_feats     = ['target',
                  'age',
                  'gender',
                  'education_type',
                  'family_status',
                  'family_nb',
                  'children_nb',
                  'work_type',
                  'organization_type',
                  'work_exp',
                  'income_amount',
                  'income_type',
                  'contract',
                  'credit_amount',
                  'goods_price',
                  'cover_rate',
                  'debt_rate',
                  'debt_load',
                  'annuity',
                  'credit_term',
                  'credit_time_elapsed'
                  ]

#best_raw_feats = ['target',
#                  'age',
#                  'gender',
#                  'education_type',
#                  'family_status',
#                  'family_nb',
#                  'children_nb',
#                  'work_type',
#                  'organization_type',
#                  'work_exp',
#                  'income_amount',
#                  'income_type',
#                  'contract',
#                  'credit_amount',
#                  'goods_price',
#                  'annuity',
#                  'credit_time_elapsed'
#                  ]

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

label_feats   = ['contract', 
                 'own_car', 
                 'own_realty'
                ]



### FUNCTIONS


    # Filter

def check_filters(filters_dict, data):        

    new_dict = {}

    st.text(" \n")
    st.subheader('Selected Filters')
    #st.write("#### **Selected Filters**")

    for filter in filters_dict.keys():

        values = filters_dict[filter]

        if type(values) == list :
            
            if values != []:

                st.write("    **{}** : {}".format(filter.replace('_', ' ').title(), ', '.join(values)))
                new_dict[filter] = values

        elif type(values) == tuple:
            
            if values != ():

                if values[0] != data[filter].min() or values[1] != data[filter].max():

                    st.write("    **{}** : {}".format(filter.replace('_', ' ').title(), ' - '.join([str(values[0]), str(values[1])])))
                    new_dict[filter] = values              

    if not sum([True if (v != [] and v != ()) else False for v in new_dict.values()]):
        st.write("    {}".format("No filter"))

    return new_dict          

def filter_with_var(data, var, choices):

    filtered_data = data.copy()

    if type(choices) == list and choices != []:

        filtered_data = filtered_data[filtered_data[var].isin(choices)]

    elif type(choices) == tuple and choices != ():

        filtered_data = filtered_data[(filtered_data[var] >= choices[0]) & (filtered_data[var] <= choices[1])]

    return filtered_data
  
def filter_with_dict(data, filters_dict): 

    filtered_data = data.copy()

    for var in filters_dict.keys():

        filtered_data = filter_with_var(filtered_data, var, filters_dict[var])

    return filtered_data


    # Data

@st.cache(suppress_st_warning=True)
def load_data(file_name):

    state = st.text('Loading clients dataset...')

    data  = pd.read_csv("results/" + file_name, index_col='id')

    if "Unnamed: 0" in data.columns.to_list():
        data = data.drop("Unnamed: 0", axis=0)

    #data = data.rename(columns = {'target' : 'loan_refund'})

    data.target = data.target.replace({0 : 'repaid',
                                       1 : 'default',
                                       }
                                      )
                        

    state.text('Loading clients dataset... Completed !')

    return data


    # Buttons

#@st.cache
def create_slider(object, data, var):

    name    = var.replace('_', ' ').title()

    choices = object.slider(name, 
                            min_value = data[var].min(),
                            max_value = data[var].max(),
                            step      = 1.0                      
                            )

    return choices

#@st.cache
def create_double_slider(object, data, var):

    name    = var.replace('_', ' ').title()

    if type(data[var].min()) == float:
        step = 1.0

    else:
        step = 1

    choices = object.slider(name, 
                            value     = [data[var].min(), data[var].max()],
                            min_value = data[var].min(),
                            max_value = data[var].max(),
                            step      = step                    
                            )

    return choices

#@st.cache
def create_selectbox(object, data, var):

    name    = var.replace('_', ' ').title()

    if var == "index":
        values  = data.index.tolist()

    else:
        values  = data[var].unique().tolist()

    choices = object.selectbox(name, options=values)

    return choices

#@st.cache
def create_multiselect(object, data, var):

    name    = var.replace('_', ' ').title()
    values  = data[var].unique().tolist()

    choices = object.multiselect(name, options=values)

    return choices

#@st.cache
def create_checkbox(data, var):

    name   = var.replace('_', ' ').title()

    values = data[var].unique().tolist()

    for v in values:
        st.sidebar.checkbox(v.replace('_', ' ').title())

        year_choice = st.sidebar.selectbox('', years)


    # Graphs

@st.cache
def plot_kde(dataset, var, title_var, value):
    
    plt.figure(figsize = (5, 5))

    fig = sns.kdeplot(data      = dataset[dataset.target == "repaid"], 
                      x         = var, 
                      shade     = True, 
                      alpha     = 0.3, 
                      label     = "repaid",
                      color     = ['forestgreen'], 
                     )

    fig = sns.kdeplot(data      = dataset[dataset.target == "default"], 
                      x         = var, 
                      shade     = True, 
                      alpha     = 0.3, 
                      label     = "default",
                      color     = ['firebrick'], 
                     )

    if title_var is None:
        title_var = ' '.join(var.split('_')).title()
        
    plt.title('{}'.format(title_var),
              fontsize   = 15, 
              fontweight = "bold", 
              pad        = 20
             )

    ax  = plt.xlabel("")
    ax  = plt.ylabel("")

    #xt  = ax.get_xticks() 
    #xt  = np.append(xt, value)
    
    #ax.set_xticks(xt)
    #ax.get_xticklabels()[np.where(xt == value)[0][0]].set_color("red")

    var_data = data[var]

    plt.axvline(value, color='b', linestyle='--', linewidth=1.5, label=str(value))

    #plt.vlines(value, 0, 0.000001, colors='r', linestyles='--', label=str(value))

    plt.legend()

    plt.show()

    return fig


    # Preprocessing

def complete_client_df(client_df, data_template_cols, one_hot_feats):
    
    full_df = pd.DataFrame(client_df, columns=data_template_cols)
    
    for var in one_hot_feats:
        
        value    = client_df[var].values[0]
        oh_feats = [feat for feat in data_template_cols if feat.startswith(var)]
        
        if type(value) == str:

            for feat in oh_feats:

                if feat.endswith(value):
                    full_df[feat] = 1

                else:
                    full_df[feat] = 0
                
    return full_df

def add_poly_feat(data, poly_transformer):
    
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
     
    data['work_exp_outliers'] = work_exp_outliers
    
    data['debt_rate']         = (data['credit_amount'] / data['income_amount'] * 100).round(2)
    data['debt_load']         = (data['annuity'] / data['income_amount'] * 100).round(2)
    data['credit_term']       = (data['credit_amount'] / data['annuity']).round(2)

    data['cover_rate']        = (data['credit_amount'] / data['goods_price'] * 100).round(2)

    data['work_exp_rate']     = (data['work_exp'] / data['age'] * 100).round(2)    

    return data

def standardize_client_df(client_df, domain_std_scaler):
    
    X         = client_df.drop('target', axis=1)
    y         = client_df.target
    
    X_std     = domain_std_scaler.transform(X)
    
    client_df = pd.DataFrame(X_std, index=X.index, columns=X.columns)
    
    client_df['target'] = y
    
    return client_df

def clean_client_df(client_df, label_encoders, iterative_imputer, poly_transformer, domain_std_scaler, 
                    data_template_cols, label_feats, one_hot_feats):
    
    
    # Encoding
    
    for var in label_feats:
    
        results        = label_encoders[var].transform(client_df[[var]])
        client_df[var] = list(results.reshape(1, -1)[0].astype(int))
    
    client_df          = complete_client_df(client_df, data_template_cols, one_hot_feats)
    
        
    # Cleaning
    
    client_df.age            = client_df.age / -365.25

    work_experience_outliers = client_df.work_exp > 0
    client_df.work_exp       = client_df.work_exp.replace({365243: np.nan}) / -365.25

    
    # Imputing
    
    target    = client_df[['target']]
    X         = client_df.drop("target", axis=1)
    
    client_df = iterative_imputer.transform(X)
    
    client_df = pd.DataFrame(client_df, index=X.index, columns=X.columns)   
    client_df = client_df.merge(target, left_index=True, right_index=True)
    
    
    # Polynomial Features
    
    client_df = add_poly_feat(client_df, poly_transformer)
    
       
    # Domain Features
    
    client_df = add_domain_feat(client_df, work_experience_outliers)
    
    
    # Standardisation
    
    std_client_df = standardize_client_df(client_df, domain_std_scaler)
    
    return std_client_df, client_df


    # Modelisation

def get_client_dict(client_id, raw_data):
    
    client_data       = raw_data[raw_data.index == client_id]
    
    client_data['id'] = client_id

    return client_data.to_dict(orient='records')[0]

def get_client_df(client_dict):
    
    client_series      = pd.Series(client_dict)
    client_series.name = client_dict['id']
    
    client_df          = client_series.to_frame().T
    
    return client_df.drop('id', axis=1)


def get_threshold_scores(fbeta_results, dataset_name, model_name):
    
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

def predict_client(cleaned_client_df, fbeta_results, dataset_name, model_name):
    
    thresholds, scores = get_threshold_scores(fbeta_results, dataset_name, model_name)

    model              = fbeta_results[dataset_name]['results'][model_name]['model']

    best_score         = max(scores)
    best_score_idx     = scores.index(best_score)

    best_threshold     = thresholds[best_score_idx]

    client_X           = cleaned_client_df.drop('target', axis=1).values
    repaid_proba       = model.predict_proba(client_X)[0][0]
        
    return repaid_proba, best_threshold

def get_client_result(repaid_proba, best_threshold):
    
    if repaid_proba < best_threshold:
        bar_color      = "red"
        result         = "refused"

    elif repaid_proba >= best_threshold:
        bar_color      = "seagreen"
        result         = "accepted"
        
    return bar_color, result


    # Data Visualisation

@st.cache
def get_gauge(repaid_proba, best_threshold, bar_color):
    
    plt.figure(figsize = (7, 7))

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

@st.cache
def get_shap_values(model, model_data):
    
    explainer    = shap.TreeExplainer(model)    
    shap_values  = explainer.shap_values(model_data.drop('target', axis=1).values)
    
    return explainer, shap_values

@st.cache(allow_output_mutation=True)
def get_force_plot(model, model_data, shap_data, client_id):
    
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

#@st.cache(suppress_st_warning=True)
def plot_shap(plot, height=None):

    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"

    components.html(shap_html, height=height)

def plot_lgbm_features_importances(light_gbm, data_template_cols, top_nb=40):
    
    feature_imp = pd.DataFrame(sorted(zip(light_gbm.feature_importances_, data_template_cols)), columns=['Value','Feature'])
    
    plt.figure(figsize=(20, 12))

    fig = sns.barplot(x    = "Value", 
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
    #plt.show()

    return fig


### DASHBOARD APP

st.sidebar.markdown("# Navigation")
page = st.sidebar.radio("", ["Home", "Exploration", "Comparison", "Scoring", "Methodology"])


# HOME PAGE

if page == "Home":

    st.header("DashLoan")
    st.markdown("### The loan granding dashboard")


# EXPLORATION PAGE

if page == "Exploration":

    st.header("**Explore** clients data")


        # Data

    st.text(" \n")
    st.subheader('Clients Data')
    data = load_data("exploration_data.csv")
    st.dataframe(data.head(100))



        # Side Panel


    st.sidebar.markdown("# Filter Categories")
  
    basics_check = st.sidebar.checkbox('Overview',   key='overview')
    target_check = st.sidebar.checkbox('Target',     key='target')
    demo_check   = st.sidebar.checkbox('Demography', key='demo')
    work_check   = st.sidebar.checkbox('Work',       key='work')
    bank_check   = st.sidebar.checkbox('Bank',       key='bank')

    #apply        = st.sidebar.button("Apply filters")


            # Basics

    if basics_check:
    
        st.sidebar.subheader('Demography')
    
        gender        = create_multiselect   (st.sidebar, data, 'gender')   
        edu           = create_multiselect   (st.sidebar, data, 'education_type')
        family_status = create_multiselect   (st.sidebar, data, 'family_status')
        age           = create_double_slider (st.sidebar, data, "age")

        st.sidebar.subheader('Work')

        work          = create_multiselect   (st.sidebar, data, 'work_type')
        orga_type     = create_multiselect   (st.sidebar, data, 'organization_type')
        income_type   = create_multiselect   (st.sidebar, data, 'income_type')
        #income_amount = create_double_slider (st.sidebar, data, "income_amount")
    
        st.sidebar.subheader('Bank')

        contract      = create_multiselect   (st.sidebar, data, 'contract')
        credit_amount = create_double_slider (st.sidebar, data, "credit_amount")
        credit_term   = create_double_slider (st.sidebar, data, "credit_term")
        annuity       = create_double_slider (st.sidebar, data, "annuity")
        goods_price   = create_double_slider (st.sidebar, data, "goods_price")
    

            # Target

    if target_check:
    
        st.sidebar.subheader('Target')

        target        = create_multiselect   (st.sidebar, data, 'target')


            # Demography

    if demo_check:
    
        st.sidebar.subheader('Demography')

        age           = create_double_slider (st.sidebar, data, "age")
        gender        = create_multiselect   (st.sidebar, data, 'gender')   
        edu           = create_multiselect   (st.sidebar, data, 'education_type')

        family_status = create_multiselect   (st.sidebar, data, 'family_status')
        #type_suite    = create_multiselect   (st.sidebar, data, 'type_suite')
        family_nb     = create_double_slider (st.sidebar, data, "family_nb")
        children_nb   = create_double_slider (st.sidebar, data, "children_nb")
  

            # Work

    if work_check:
    
        st.sidebar.subheader('Work')

        work          = create_multiselect   (st.sidebar, data, 'work_type')
        orga_type     = create_multiselect   (st.sidebar, data, 'organization_type')
        work_exp      = create_double_slider (st.sidebar, data, "work_exp")

        income_amount = create_double_slider (st.sidebar, data, "income_amount")
        income_type   = create_multiselect   (st.sidebar, data, 'income_type')


            # Bank

    if bank_check:
    
        st.sidebar.subheader('Bank')

        contract      = create_multiselect   (st.sidebar, data, 'contract')
        credit_amount = create_double_slider (st.sidebar, data, "credit_amount")
        goods_price   = create_double_slider (st.sidebar, data, "goods_price")
        cover_rate    = create_double_slider (st.sidebar, data, "cover_rate")

        debt_rate     = create_double_slider (st.sidebar, data, "debt_rate")
        debt_load     = create_double_slider (st.sidebar, data, "debt_load")

        annuity       = create_double_slider (st.sidebar, data, "annuity")
        credit_term   = create_double_slider (st.sidebar, data, "credit_term")
        credit_time   = create_double_slider (st.sidebar, data, "credit_time_elapsed")


        # Filtering

    filters_dict    = {'target'              : target,
                       'age'                 : age,
                       'gender'              : gender, 
                       'education_type'      : edu,
                       'family_status'       : family_status,
                       'family_nb'           : family_nb,
                       'children_nb'         : children_nb,
                       'work_type'           : work,
                       'organization_type'   : orga_type, 
                       'work_exp'            : work_exp, 
                       'income_amount'       : income_amount,
                       'income_type'         : income_type,
                       'contract'            : contract,
                       'credit_amount'       : credit_amount, 
                       'goods_price'         : goods_price, 
                       'cover_rate'          : cover_rate,
                       'debt_rate'           : debt_rate,
                       'debt_load'           : debt_load,
                       'annuity'             : annuity,
                       'credit_term'         : credit_term,
                       'credit_time_elapsed' : credit_time,
                       }

    filtered_data   = data.copy()

    if basics_check or target_check or demo_check or work_check or bank_check:
        
        filters_dict = check_filters(filters_dict, data)
    
        if sum([True if (v != [] and v != ()) else False for v in filters_dict.values()]):
    
            filtered_data = filter_with_dict(data, filters_dict)

            st.text(" \n")
            st.subheader('Filtered Data : {} clients'.format(filtered_data.shape[0]))
            st.write(filtered_data.head(100))

            st.text(" \n")
            st.subheader('Statistics : {}% default rate'.format(round(filtered_data.target.replace({"repaid":0, "default":1}).mean()*100, 2)))
            stats         = filtered_data[best_feats].describe()
            st.write(stats.T[['mean', 'std', 'min', 'max']])


# COMPARISON PAGE

if page == "Comparison":

    st.header("**Compare** clients data")


        # Data

    data = load_data("exploration_data.csv")
    #st.write(data.head(100))


        # Side Panel

    st.sidebar.markdown("# Client Infos")

    client_id   = st.sidebar.selectbox("", options=data.index.tolist())
    client_data = data[data.index == float(client_id)][best_feats].drop(['family_nb', 'organization_type'], axis=1)

    st.sidebar.text(" \n")

    if client_id not in ["Enter the ID of a client...", None, ""]:
        for c in client_data.columns:
            st.sidebar.write('    **{}** : {}'.format(c.replace('_', ' ').title(), client_data.loc[float(client_id), c]))
    
    st.sidebar.markdown("# Filter Categories")
  
    basics_check = st.sidebar.checkbox('Overview',   key='overview')
    target_check = st.sidebar.checkbox('Target',     key='target')
    demo_check   = st.sidebar.checkbox('Demography', key='demo')
    work_check   = st.sidebar.checkbox('Work',       key='work')
    bank_check   = st.sidebar.checkbox('Bank',       key='bank')

    #apply        = st.sidebar.button("Apply filters")


            # Basics

    if basics_check:
    
        st.sidebar.subheader('Demography')
    
        gender        = create_multiselect   (st.sidebar, data, 'gender')   
        edu           = create_multiselect   (st.sidebar, data, 'education_type')
        family_status = create_multiselect   (st.sidebar, data, 'family_status')
        age           = create_double_slider (st.sidebar, data, "age")

        st.sidebar.subheader('Work')

        work          = create_multiselect   (st.sidebar, data, 'work_type')
        orga_type     = create_multiselect   (st.sidebar, data, 'organization_type')
        income_type   = create_multiselect   (st.sidebar, data, 'income_type')
        #income_amount = create_double_slider (st.sidebar, data, "income_amount")
    
        st.sidebar.subheader('Bank')

        contract      = create_multiselect   (st.sidebar, data, 'contract')
        credit_amount = create_double_slider (st.sidebar, data, "credit_amount")
        credit_term   = create_double_slider (st.sidebar, data, "credit_term")
        annuity       = create_double_slider (st.sidebar, data, "annuity")
        goods_price   = create_double_slider (st.sidebar, data, "goods_price")
    

            # Target

    if target_check:
    
        st.sidebar.subheader('Target')

        target        = create_multiselect   (st.sidebar, data, 'target')


            # Demography

    if demo_check:
    
        st.sidebar.subheader('Demography')

        age           = create_double_slider (st.sidebar, data, "age")
        gender        = create_multiselect   (st.sidebar, data, 'gender')   
        edu           = create_multiselect   (st.sidebar, data, 'education_type')

        family_status = create_multiselect   (st.sidebar, data, 'family_status')
        #type_suite    = create_multiselect   (st.sidebar, data, 'type_suite')
        family_nb     = create_double_slider (st.sidebar, data, "family_nb")
        children_nb   = create_double_slider (st.sidebar, data, "children_nb")
  

            # Work

    if work_check:
    
        st.sidebar.subheader('Work')

        work          = create_multiselect   (st.sidebar, data, 'work_type')
        orga_type     = create_multiselect   (st.sidebar, data, 'organization_type')
        work_exp      = create_double_slider (st.sidebar, data, "work_exp")

        income_amount = create_double_slider (st.sidebar, data, "income_amount")
        income_type   = create_multiselect   (st.sidebar, data, 'income_type')


            # Bank

    if bank_check:
    
        st.sidebar.subheader('Bank')

        contract      = create_multiselect   (st.sidebar, data, 'contract')
        credit_amount = create_double_slider (st.sidebar, data, "credit_amount")
        goods_price   = create_double_slider (st.sidebar, data, "goods_price")
        cover_rate    = create_double_slider (st.sidebar, data, "cover_rate")

        debt_rate     = create_double_slider (st.sidebar, data, "debt_rate")
        debt_load     = create_double_slider (st.sidebar, data, "debt_load")

        annuity       = create_double_slider (st.sidebar, data, "annuity")
        credit_term   = create_double_slider (st.sidebar, data, "credit_term")
        credit_time   = create_double_slider (st.sidebar, data, "credit_time_elapsed")


        # Filtering

    filters_dict    = {'target'              : target,
                       'age'                 : age,
                       'gender'              : gender, 
                       'education_type'      : edu,
                       'family_status'       : family_status,
                       'family_nb'           : family_nb,
                       'children_nb'         : children_nb,
                       'work_type'           : work,
                       'organization_type'   : orga_type, 
                       'work_exp'            : work_exp, 
                       'income_amount'       : income_amount,
                       'income_type'         : income_type,
                       'contract'            : contract,
                       'credit_amount'       : credit_amount, 
                       'goods_price'         : goods_price, 
                       'cover_rate'          : cover_rate,
                       'debt_rate'           : debt_rate,
                       'debt_load'           : debt_load,
                       'annuity'             : annuity,
                       'credit_term'         : credit_term,
                       'credit_time_elapsed' : credit_time,
                       }

    filtered_data = data.copy()

    if basics_check or target_check or demo_check or work_check or bank_check:
        
        filters_dict = check_filters(filters_dict, data)
    
        if sum([True if (v != [] and v != ()) else False for v in filters_dict.values()]):
    
            filtered_data = filter_with_dict(data, filters_dict)
    

        # Client Data

    if client_id not in ["Enter the ID of a client...", None, ""]:

        st.text(" \n")
        st.subheader('Client n°{}'.format(client_id))

        #st.area_chart(chart_data)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        
        col_1, col_2 = st.columns(2)
        
        var = "age"
        chart_1 = plot_kde(filtered_data, var, var.replace('_', ' ').title(), client_data.loc[float(client_id), var])
        col_1.pyplot()

        var = "work_exp"
        chart_2 = plot_kde(filtered_data, var, var.replace('_', ' ').title(), client_data.loc[float(client_id), var])
        col_2.pyplot()

        #var = "children_nb"
        #chart_3 = plot_kde(filtered_data, var, var.replace('_', ' ').title(), client_data.loc[float(client_id), var])
        #col_3.pyplot()
                 
        #var = "income_amount"
        #chart_4 = plot_kde(filtered_data, var, var.replace('_', ' ').title(), client_data.loc[float(client_id), var])
        #col_1.pyplot()

        var = "credit_amount"
        chart_5 = plot_kde(filtered_data, var, var.replace('_', ' ').title(), client_data.loc[float(client_id), var])
        col_1.pyplot()

        var = "goods_price"
        chart_6 = plot_kde(filtered_data, var, var.replace('_', ' ').title(), client_data.loc[float(client_id), var])
        col_2.pyplot()

        var = "debt_rate"
        chart_7 = plot_kde(filtered_data, var, var.replace('_', ' ').title(), client_data.loc[float(client_id), var])
        col_1.pyplot()

        var = "debt_load"
        chart_8 = plot_kde(filtered_data, var, var.replace('_', ' ').title(), client_data.loc[float(client_id), var])
        col_2.pyplot()

        var = "annuity"
        chart_9 = plot_kde(filtered_data, var, var.replace('_', ' ').title(), client_data.loc[float(client_id), var])
        col_1.pyplot()

        var = "credit_term"
        chart_10 = plot_kde(filtered_data, var, var.replace('_', ' ').title(), client_data.loc[float(client_id), var])
        col_2.pyplot()


# SCORING PAGE

if page == "Scoring":

    st.header("**Explain** scoring")


        # Data & Model

    data               = load_data("exploration_data.csv")
    raw_data           = load_data("data.csv")

    data_template_cols = pickle.load(open("results/data_template_cols.pickle", "rb"))

    label_encoders     = pickle.load(open("results/label_encoders.pickle", "rb"))
    iterative_imputer  = pickle.load(open("results/iterative_imputer.pickle", "rb"))
    poly_transformer   = pickle.load(open("results/poly_transformer.pickle", "rb"))
    domain_std_scaler  = pickle.load(open("results/domain_std_scaler.pickle", "rb"))

    dataset_name       = "domain_feat"
    model_title        = 'Light GBM'

    fbeta_results      = pickle.load(open("results/fbeta_results.pickle", "rb"))
    fb_results_dict    = fbeta_results[dataset_name]['results']

    model_name         = [name for name in fb_results_dict.keys() if name.startswith(model_title)][0]
    model              = fb_results_dict[model_name]['model']


    #st.write(data.head(100))


        # Side Panel

    st.sidebar.markdown("# Client Infos")

    client_id   = st.sidebar.selectbox("", options=data.index.tolist())

    client_dict = get_client_dict(client_id, raw_data)
    client_df   = get_client_df(client_dict)

    client_data = data[data.index == float(client_id)][best_feats].drop(['family_nb', 'organization_type'], axis=1)

    st.sidebar.text(" \n")

    if client_id not in ["Enter the ID of a client...", None, ""]:
        for c in client_data.columns:
            st.sidebar.write('    **{}** : {}'.format(c.replace('_', ' ').title(), client_data.loc[float(client_id), c]))


    # Main Area

    if client_id not in ["Enter the ID of a client...", None, ""]:


        # Preprocessing

        (std_client_df, 
         client_df)       = clean_client_df(client_df, label_encoders, iterative_imputer, poly_transformer, domain_std_scaler, 
                                            data_template_cols, label_feats, one_hot_feats
                                           )

        # Modelisation

        (repaid_proba, 
         best_threshold) = predict_client(client_df, fbeta_results, dataset_name, model_name)

        (bar_color, 
         result)         = get_client_result(repaid_proba, best_threshold)


        # Plot Gauge

        st.text(" \n")
        st.subheader('Decision : loan {}'.format(result))

        gauge = get_gauge(repaid_proba, best_threshold, bar_color)
        st.plotly_chart(gauge)


        # Plot Force 

        if model_title == 'Light GBM':

            st.text(" \n")
            st.subheader('Explanation : {}% repaid probability'.format(round(repaid_proba * 100, 2)))

            force = get_force_plot(model, std_client_df, client_df, client_df.index[0])
            plot_shap(force)


        # Plot Light GBM Features Importances

        #if model_title == 'Light GBM':

        #    top_nb = 40

        #    st.text(" \n")
        #    st.subheader('Model {} : most {} important features'.format(model_title, top_nb))

        #    feat_imp = plot_lgbm_features_importances(model, data_template_cols, top_nb)
        #    st.pyplot(feat_imp)


