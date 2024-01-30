import os
import streamlit as st
import pandas as pd

from modules.data import default_df, col_score, k_score
from modules.graphs import generate_graph

styles_path = os.path.join(os.path.dirname(__file__), 'assets', 'styles.css')

with open(styles_path) as f:
    st.write(f'<style>{f.read()}</style>', unsafe_allow_html = True)

if 'relevant_scores' not in st.session_state:
    st.session_state['relevant_scores'] = None

st.title('K nearest neighbors metric calculator')
st.write('This is an app to allow users to get the most relevant metrics of the data imported')
uploaded_file = st.file_uploader('Import a csv', type = 'csv', accept_multiple_files = False)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
else:
    df = default_df
    
df_head = st.dataframe(df.head(5), use_container_width = True)
    
y = st.selectbox(
    label = 'Class',
    options = df.columns.to_list(),
    index = len(df.columns) - 1)

with st.expander(
    label = 'Acuraccy analysis'):
    
    features = st.multiselect(
        label = 'Features',
        placeholder = 'Select features',
        options = df.columns.drop(y).to_list(),
        default = df.columns.drop(y).to_list())
    
    col_score_button = st.button('Calculate discrete scores')
    
    if col_score_button:
        st.session_state['relevant_scores'] = col_score(
            df,
            features,
            y)
        
    if st.session_state['relevant_scores'] is not None:
            
        st.markdown(
            '''<p id = "accuracy_title">Discrete accuracy values 
            for the selected features (k = 6)</p>''',
                    unsafe_allow_html = True)
            
        st.plotly_chart(
                use_container_width = True,
                figure_or_data = generate_graph(
                    data = st.session_state['relevant_scores'],
                    abs = 'variable',
                    ord = 'accuracy',
                    fig_type = 'Bar'))
        
with st.expander(
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
            value = 9)
        
        k_score_button = st.button('Calculate k accuracy values')
        
        if k_score_button:
            
            k_scores = k_score(
                df,
                k_features,
                y,
                k_max)
            
            if k_scores is not None:
                
                st.plotly_chart(
                    use_container_width = True,
                    figure_or_data = generate_graph(
                    data = k_scores,
                    abs = 'k',
                    ord = 'accuracy',
                    fig_type = 'Line'))
        
    else:
        st.write('Please evaluate the most relevant features first')