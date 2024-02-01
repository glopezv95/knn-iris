from ucimlrepo import fetch_ucirepo
import pandas as pd

# Fetch data for UCI Machine Learning Repository
iris = fetch_ucirepo(id = 53)
iris.data.keys()
# Generate pandas DataFrame
default_df = pd.DataFrame(
    data = iris.data.features,
    columns = iris.data.headers)

default_df['species'] = iris.data.targets
default_df = default_df.drop('class', axis = 1)
default_df.columns = default_df.columns.str.replace(' ', '_')