import plotly.graph_objects as go

def generate_bar(data, abs, ord):
    fig = go.Figure(go.Bar(
        x = data[abs],
        y = data[ord],
        hovertemplate = 'Variable: %{x}<br>Accuracy: %{y}',
        name = ''))
    
    fig.update_layout(
        height = 300,
        margin = dict(l=20, r=20, t=5, b=5)
    )
    
    return fig

def generate_line(data, abs, ord):
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