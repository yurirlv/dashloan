### SCORING DASHBOARD


# LIBRAIRIES


import numpy                           as np
import pandas                          as pd
import matplotlib.pyplot               as plt
import seaborn                         as sns

import shap
import plotly.graph_objects            as go

import pickle
import json

import requests

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

models_names  = ['KNN', 
                 'Bagging', 
                 'Random Forest', 
                 'XG Boost', 
                 'Light GBM'
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

                if values[0] != int(data[filter].min()) or values[1] != int(data[filter].max()):

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

    try:
        data  = pd.read_csv(file_name, index_col='id')

    except:
        data  = pd.read_csv(file_name, index_col=0)

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
                            value     = [int(data[var].min()),int( data[var].max())],
                            min_value = int(data[var].min()),
                            max_value = int(data[var].max()),
                            #step      = step                  
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

    plt.axvline(value, color='b', linestyle='--', linewidth=1.5, label=str(value))

    #plt.vlines(value, 0, 0.000001, colors='r', linestyles='--', label=str(value))

    plt.legend()

    return fig



    # API

def get_client_dict(client_id, raw_data):
    
    client_data       = raw_data[raw_data.index == client_id]
    
    client_data['id'] = client_id

    return client_data.to_dict(orient='records')[0]

def send_request_data(serveur_url, data_dict):
    
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


    # Data Visualisation

def get_client_result(repaid_proba, best_threshold):
    
    if repaid_proba < best_threshold:
        bar_color      = "red"
        result         = "refused"

    elif repaid_proba >= best_threshold:
        bar_color      = "seagreen"
        result         = "accepted"
        
    return bar_color, result

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
    data = load_data("data/resampled_exp_data.csv")
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

    data = load_data("data/resampled_exp_data.csv")


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

    data               = load_data("data/resampled_exp_data.csv")
    raw_data           = load_data("data/resampled_data.csv")
    

        # Side Panel

    st.sidebar.markdown("# Model")

    model_name         = st.sidebar.selectbox("", options=models_names, index=4)


    st.sidebar.markdown("# Client Infos")

    client_id          = st.sidebar.selectbox("", options=data.index.tolist())
    client_dict        = get_client_dict(client_id, raw_data)

    client_data        = data[data.index == float(client_id)][best_feats].drop(['family_nb', 'organization_type'], axis=1)

    st.sidebar.text(" \n")

    if client_id not in ["Enter the ID of a client...", None, ""]:
        for c in client_data.columns:
            st.sidebar.write('    **{}** : {}'.format(c.replace('_', ' ').title(), client_data.loc[float(client_id), c]))


    # Main Area

    if client_id not in ["Enter the ID of a client...", None, ""]:

        data_dict      = {'client'     : client_dict, 
                          'model_name' : model_name
                         }


        # Get data from Scoring API

        api_resp       = send_request_data("http://127.0.0.1:8000/predict", data_dict)

        (client_data, 
         std_client_data, 
         repaid_proba, 
         threshold, 
         fbeta_score)  = get_request_data(api_resp, client_id)
        

        (bar_color, 
         result)       = get_client_result(repaid_proba, threshold)


        # Plot Gauge

        st.text(" \n")
        st.subheader('Decision : loan {}'.format(result))

        gauge = get_gauge(repaid_proba, threshold, bar_color)
        st.plotly_chart(gauge)


        # Plot Force 

        if model_name == 'Light GBM':

            st.text(" \n")
            st.subheader('Explanation : {}% repaid probability'.format(round(repaid_proba * 100, 2)))

            model_dict = pickle.load(open("modelisation/domain_{}_dict.pickle".format(model_name.replace(' ', '_').lower()), "rb"))  
            model      = model_dict['model']  

            force      = get_force_plot(model, std_client_data, client_data, client_data.index[0])
            plot_shap(force)


        # Plot Light GBM Features Importances

        #if model_title == 'Light GBM':

        #    top_nb = 40

        #    st.text(" \n")
        #    st.subheader('Model {} : most {} important features'.format(model_title, top_nb))

        #    feat_imp = plot_lgbm_features_importances(model, data_template_cols, top_nb)
        #    st.pyplot(feat_imp)


