import plotly.graph_objects as go

def generate_graph(data, abs:str, ord:str, fig_type:str):
    fig_type_allowed = ['Bar', 'Line']
    
    if fig_type not in fig_type_allowed:
        raise ValueError('Expected one fig_type of {fig_type_allowed}')
    
    elif fig_type == 'Bar':
        fig = go.Figure(go.Bar(
            x = data[abs],
            y = data[ord],
            hovertemplate = 'Variable: %{x}<br>Accuracy: %{y}',
            name = ''))
    
    else:
        fig = go.Figure(go.Line(
            x = data[abs],
            y = data[ord],
            hovertemplate = 'Variable: %{x}<br>Accuracy: %{y}',
            name = ''))
    
    fig.update_layout(
        height = 300,
        margin = dict(l=20, r=20, t=5, b=5)
    )
    
    return fig