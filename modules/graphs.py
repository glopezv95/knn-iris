from plotly.subplots import make_subplots
import plotly.graph_objects as go

def generate_graph(data, abs:str, ord:list, fig_type:str):
    fig_type_allowed = ['Bar', 'Line']
    fig = go.Figure()
    
    if fig_type not in fig_type_allowed:
        raise ValueError('Expected one fig_type of {fig_type_allowed}')
    
    elif fig_type == 'Bar':
        
        for value in ord:
            fig.add_trace(trace = go.Bar(
                x = data[abs],
                y = data[value],
                hovertemplate = 'Variable: %{x}<br>Value: %{y}',
                name = value))
    
    else:
        for value in ord:
            fig.add_trace(trace = go.Line(
                x = data[abs],
                y = data[value],
                hovertemplate = 'Variable: %{x}<br>Value: %{y}',
                name = value.strip('accuracy_')))
    
    fig.update_layout(
        height = 300,
        margin = dict(l=20, r=20, t=5, b=5))
    
    return fig