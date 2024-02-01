import os
import streamlit as st
import pandas as pd
import numpy as np

from modules.data import default_df, col_score, k_score, gen_knn
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
    uploaded_file = st.file_uploader('Import a csv',
                                     type = 'csv',
                                     accept_multiple_files = False)

@st.cache_data()
def process_file(data):
    df = pd.read_csv(data).dropna(axis = 1)
    st.session_state['relevant_scores'] = None
    st.session_state['k_value'] = None
    st.session_state['model_features'] = {}
    st.session_state['k_value_markdown'] = None
    st.session_state['df'] = df

if uploaded_file is not None:
    process_file(uploaded_file)

df = st.session_state['df']

col1, col2 = st.columns(2)

df_head = col1.dataframe(df.head(5), use_container_width = True)
        
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
                \n- '''.join(st.session_state['relevant_scores']['variable'][:3]))
        
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
                    \n- '''.join(st.session_state['relevant_scores']['variable'][:3]))
            
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
        
    
        prediction_button = st.button('Generate prediction')    
        
        if prediction_button:
            prediction = gen_knn(
                data = df,
                X = model_features,
                X_to_pred = X_to_pred,
                y = y,
                k = k_model)
            
            side_menu.markdown(
            f'''**Highest accuracy features**  \n- ''' + '''\
                \n- '''.join(st.session_state['relevant_scores']['variable'][:3]))
        
            side_menu.markdown((f'**Optimum k value**: {st.session_state["k_value"]}'))
            
            st.write(f'{y.title()} prediction: {prediction[0]}')
        
    else:
        st.write('Please evaluate the most relevant features and optimal k first')