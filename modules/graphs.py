from plotly.subplots import make_subplots
import plotly.graph_objects as go

def generate_graph(data, abs:str, ord:list, fig_type:str):
    fig_type_allowed = ['Bar', 'Line']
    fig = go.Figure()
    color_sequence = [
        '#4C3A51', '#774360', '#B25068', '#E7AB79']
    
    if fig_type not in fig_type_allowed:
        raise ValueError('Expected one fig_type of {fig_type_allowed}')
    
    elif fig_type == 'Bar':
        
        for index, value in enumerate(ord):
            fig.add_trace(trace = go.Bar(
                x = data[abs],
                y = data[value],
                hovertemplate = 'Variable: %{x}<br>Value: %{y}',
                name = value,
                marker = dict(color = color_sequence[index])))
    
    else:
        for index, value in enumerate(ord):
            fig.add_trace(trace = go.Scatter(
                x = data[abs],
                y = data[value],
                hovertemplate = 'Variable: %{x}<br>Value: %{y}',
                name = value.strip('accuracy_'),
                mode = 'lines',
                line = dict(color = color_sequence[index])))
    
    fig.update_layout(
        height = 300,
        margin = dict(l=20, r=20, t=5, b=5))
    
    return fig