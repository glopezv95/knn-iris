import os
import streamlit as st
import pandas as pd
import numpy as np

from modules.data import default_df
from modules.knn import col_score, k_score, gen_knn
from modules.graphs import generate_graph

styles_path = os.path.join(os.path.dirname(__file__), 'assets', 'styles.css')

st.set_page_config(layout = 'wide')

with open(styles_path) as f:
    st.write(f'<style>{f.read()}</style>', unsafe_allow_html = True)

if 'relevant_scores' not in st.session_state:
    st.session_state['relevant_scores'] = None
    
if 'k_value' not in st.session_state:
    st.session_state['k_value'] = None
    
if 'model_features' not in st.session_state:
    st.session_state['model_features'] = {}
    
if 'k_value_markdown' not in st.session_state:
    st.session_state['k_value_markdown'] = None

if 'df' not in st.session_state:
    df = default_df.dropna(axis = 1)
    st.session_state['df'] = df

side_menu = st.sidebar
    
with st.sidebar:
    st.title('K nearest neighbors metric calculator')
    st.divider()
    uploaded_file = st.file_uploader('Import a csv',
                                     type = 'csv',
                                     accept_multiple_files = False)
@st.cache_resource
def process_file(data):
    st.session_state['relevant_scores'] = None
    st.session_state['k_value'] = None
    st.session_state['model_features'] = {}
    st.session_state['k_value_markdown'] = None
    st.session_state['df'] = pd.read_csv(data).dropna(axis = 1)

if uploaded_file is not None:
    process_file(uploaded_file)

df = st.session_state['df']

col1, col2 = st.columns(2)

df_head = col1.dataframe(df.head(5), use_container_width = True,
                         hide_index = True)
        
y = col1.selectbox(
    label = 'Class',
    options = df.columns.to_list(),
    index = len(df.columns) - 1)

with col2.expander(
    label = 'Acuraccy analysis'):
    
    features = st.multiselect(
        label = 'Features',
        placeholder = 'Select features',
        options = df.columns.drop(y).to_list(),
        default = df.columns.drop(y).to_list())
    
    average = st.selectbox(
        label = 'Average selection',
        options = ['micro', 'macro', 'weighted'],
        key = 'micro',
        placeholder = '''Select an average type to perform score \
            calculation''')
    
    col_score_button = st.button('Calculate discrete scores')
    
    if col_score_button:
        
        st.session_state['relevant_scores'] = col_score(
            df,
            features,
            y,
            avg = average)
                
        st.markdown(
            '''<p id = "accuracy_title">Discrete accuracy values 
            for the selected features (k = 6)</p>''',
                    unsafe_allow_html = True)
    
        st.plotly_chart(
            use_container_width = True,
            figure_or_data = generate_graph(
                data = st.session_state['relevant_scores'],
                abs = 'variable',
                ord = ['accuracy', 'precision', 'recall', 'f1'],
                fig_type = 'Bar'))
        
        side_menu.markdown(
            f'''**Highest accuracy features**  \n- ''' + '''\
                \n- '''.join([f'{score["variable"]}: **{score["accuracy"]}**' \
                    for i, score in st.session_state['relevant_scores'].head(3).iterrows()]))
        
with col2.expander(
    label = 'k analysis'):
    if st.session_state['relevant_scores'] is not None:
            
        k_features = st.multiselect(
            label = 'Testing features',
            placeholder = 'Select features to optimize k',
            options = st.session_state['relevant_scores']['variable'],
            default = st.session_state['relevant_scores']['variable'][:3])
        
        k_max = st.number_input(
            label = 'Maximum k value',
            placeholder = 'Select the maximum k value to analyze',
            value = 9,
            min_value = 1,
            max_value = 60)
        
        k_score_button = st.button('Calculate k accuracy values')
        
        if k_score_button:
            
            k_scores = k_score(
                df,
                k_features,
                y,
                k_max)
            
            st.session_state['k_value'] = k_scores.sort_values(
                by = 'accuracy_test',
                ascending = False).reset_index(drop = True)['k'][0]
            
            side_menu.markdown(
            f'''**Highest accuracy features**  \n- ''' + '''\
                \n- '''.join([f'{score["variable"]}: {score["accuracy"]}' \
                    for i, score in st.session_state['relevant_scores'].head(3).iterrows()]))
            
            side_menu.markdown((f'**Optimum k value**: {st.session_state["k_value"]}'))
            
            if k_scores is not None:
                
                st.plotly_chart(
                    use_container_width = True,
                    figure_or_data = generate_graph(
                    data = k_scores,
                    abs = 'k',
                    ord = ['accuracy_test', 'accuracy_train'],
                    fig_type = 'Line'))
                
    else:
        st.write('Please evaluate the most relevant features first')
        
with col2.expander(
    label = 'Prediction model'):
    
    if st.session_state['k_value'] is not None:
    
        k_model = st.number_input(
            label = 'Model k value',
            placeholder = "Select prediction model's k",
            value = st.session_state['k_value'],
            min_value = 1,
            max_value = 60)
        
        model_features = st.multiselect(
            label = 'Model features',
            placeholder = "Select prediction model's features",
            options = st.session_state['relevant_scores']['variable'],
            default = st.session_state['relevant_scores']['variable'][:3])
        
        user_inputs = {}
        
        for item in model_features:
            default_value = st.session_state['model_features'].get(f'{item}', np.random.choice(df[item]))
            
            user_inputs[f'{item}'] = st.number_input(
                label = f'{item} value to predict',
                placeholder = f"Select model's feature {item} value to predict",
                value = default_value,
                min_value = 0.00,
                step = .01)
        
        st.session_state['model_features'] = user_inputs
        X_to_pred = np.array(list(user_inputs.values())).reshape(1, -1)
        
        subcol1, subcol2 = st.columns(2)
        prediction_button = subcol1.button('Generate prediction')    
        
        if prediction_button:
            prediction = gen_knn(
                data = df,
                X = model_features,
                X_to_pred = X_to_pred,
                y = y,
                k = k_model)
            
            side_menu.markdown(
            f'''**Highest accuracy features**  \n- ''' + '''\
                \n- '''.join([f'{score["variable"]}: {score["accuracy"]}' \
                    for i, score in st.session_state['relevant_scores'].head(3).iterrows()]))
        
            side_menu.markdown((f'**Optimum k value**: {st.session_state["k_value"]}'))
            
            subcol2.subheader(f'{prediction[0]}', divider = 'red')
        
    else:
        st.write('Please evaluate the most relevant features and optimal k first')
        
data_columns1, data_columns2= col2.columns([.65, .35])

with data_columns1.expander(
    label = 'Variables datatype'):

    types_df = pd.DataFrame(
        df.dtypes,
        columns = ['Data type']).reset_index(names = 'Variable')
    
    st.dataframe(types_df, use_container_width = True, hide_index = True)

with data_columns1.expander(
    label = 'Class instances'):
    
    class_instances = df.groupby(y).agg('count').iloc[:,0].reset_index()
    class_instances.columns = [f'{y.title()}', 'Instance']
    st.dataframe(class_instances, use_container_width = True, hide_index = True)
    
data_columns2.write(f'#### Data table size')
data_columns2.write(f'''Number of rows: {len(df)}  \n\
           Number of columns: {len(df.columns)}''')

side_menu.divider()

side_menu.write('Gabriel López Vinielles')

side_menu_col1, side_menu_col2 = side_menu.columns(2)

side_menu_col1.markdown(
    '''<a href="https://www.linkedin.com/in/gabriel-lópez-vinielles-87435713a" target="_blank">
         <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" fill="white" class="bi bi-linkedin" viewBox="0 0 16 16">
           <path d="M0 1.146C0 .513.526 0 1.175 0h13.65C15.474 0 16 .513 16 1.146v13.708c0 .633-.526 1.146-1.175 1.146H1.175C.526 16 0 15.487 0 14.854zm4.943 12.248V6.169H2.542v7.225zm-1.2-8.212c.837 0 1.358-.554 1.358-1.248-.015-.709-.52-1.248-1.342-1.248S2.4 3.226 2.4 3.934c0 .694.521 1.248 1.327 1.248zm4.908 8.212V9.359c0-.216.016-.432.08-.586.173-.431.568-.878 1.232-.878.869 0 1.216.662 1.216 1.634v3.865h2.401V9.25c0-2.22-1.184-3.252-2.764-3.252-1.274 0-1.845.7-2.165 1.193v.025h-.016l.016-.025V6.169h-2.4c.03.678 0 7.225 0 7.225z"/>
         </svg>
       </a>''', unsafe_allow_html = True)

side_menu_col2.markdown(
    '''<a href="https://github.com/glopezv95" target="_blank">
         <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" fill="white" class="bi bi-github" viewBox="0 0 16 16">
           <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8"/>
         </svg>
       </a>''', unsafe_allow_html = True)
# side_menu.link_button(label = 'LinkedIn',
#                           url = 'https://www.linkedin.com/in/gabriel-lópez-vinielles-87435713a',
#                           use_container_width = True)
# side_menu.link_button(label = 'GitHub',
#                           url = 'https://github.com/glopezv95',
#                           use_container_width = True)