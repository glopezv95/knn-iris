import os
import streamlit as st
import pandas as pd

from data import default_df, col_score

styles_path = os.path.join(os.path.dirname(__file__), 'assets', 'styles.css')

with open(styles_path) as f:
    st.write(f'<style>{f.read()}</style>', unsafe_allow_html = True)

st.title('K nearest neighbors metric calculator')
st.write('This is an app to allow users to get the most relevant metrics of the data imported')
uploaded_file = st.file_uploader('Import a csv', type = 'csv', accept_multiple_files = False)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
else:
    df = default_df

df_head = st.dataframe(df.head(5))

y = st.selectbox(
    label = 'Select the classification variable',
    placeholder = 'Select a classification variable',
    options = df.columns,
    index = len(df.columns) - 1)

X_list = df.columns.drop(y).to_list()

features = st.multiselect(
    label = 'Select the features to test for the model',
    placeholder = 'Select features',
    options = X_list,
    default = X_list)

col_score_button = st.button('Calculate discrete scores')

scores = None

if col_score_button:
    scores = col_score(
        default_df,
        features,
        y)
    
if scores is not None:
    scores_col1, scores_col2 = st.columns(2)
    
    with scores_col1:
        
        st.markdown(
            '''<p id = "accuracy_title">Discrete accuracy values 
            for the selected features (k = 6)</p>''',
                    unsafe_allow_html = True)
        
        st.dataframe(scores)