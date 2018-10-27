# -*- coding: utf-8 -*-
import os
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import glob
import flask

results_root = '/Users/zhichao.lu/Documents/GitHub/results/Analysis_MNIST_Run_1'


def generate_table(dataframe, max_rows=20):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_csv(os.path.join(results_root, 'visualization', 'summary.csv'))

# add an image for testing
image_directory = os.path.join(results_root, 'visualization/')
list_of_images = [os.path.basename(x) for x in glob.glob('{}*.png'.format(image_directory))]
static_image_route = '/static/'

available_indicators = ['Design ID', 'Active Nodes', 'Accuracy', 'Params', 'FLOPs', 'Robustness']

df_experiment_args = pd.read_csv(os.path.join(results_root, 'experiment_args.txt'))


markdown_text = '''
[NSGA-Net](https://arxiv.org/abs/1810.03522): A multi-objective evolutionary framework for neural architecture search.
'''

app.layout = html.Div([
    html.H1(children='NSGA-Net Postprocessing and Visualization'),

    html.Div([
        dcc.Markdown(children=markdown_text)
    ]),


    # html.Div(children=[
    #     html.H4(children='Experiment Detail'),
    #     generate_table(df_experiment_args),
    # ]),
    #
    # html.Div([
    #     html.Img(id='image', style={'width': '500px'})
    # ]),

    html.Div([
        html.Div(children=[
            html.H4(children='Experiment Detail'),
            generate_table(df_experiment_args)
        ], style={'width': '49%', 'display': 'inline-block'}),

        html.Div([
            html.H4(children='Architecture'),
            html.Img(id='image', style={'width': '800px'})
        ], style={'width': '49%', 'float': 'bottom',
                  'display': 'inline-block',
                  'vertical-align': 'bottom',
                  }),
    ]),

    html.Div([

        html.Div([
            dcc.Dropdown(
                id='crossfilter-xaxis-column',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value='FLOPs'
            ),
            dcc.RadioItems(
                id='crossfilter-xaxis-type',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                value='Linear',
                labelStyle={'display': 'inline-block'}
            )
        ], style={'width': '24%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='crossfilter-yaxis-column',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value='Accuracy'
            ),
            dcc.RadioItems(
                id='crossfilter-yaxis-type',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                value='Linear',
                labelStyle={'display': 'inline-block'}
            )
        ], style={'width': '24%', 'float': 'middle', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='crossfilter-color',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value='Design ID'
            ),
            dcc.RadioItems(
                id='crossfilter-color-type',
                options=[{'label': i, 'value': i} for i in ['Color']],
                value='Linear',
                labelStyle={'display': 'inline-block'}
            )
        ], style={'width': '24%', 'float': 'middle', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='crossfilter-size',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value='Params'
            ),
            dcc.RadioItems(
                id='crossfilter-size-type',
                options=[{'label': i, 'value': i} for i in ['Size']],
                value='Linear',
                labelStyle={'display': 'inline-block'}
            )
        ], style={'width': '24%', 'float': 'middle', 'display': 'inline-block'}),

    ], style={
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(240, 240, 240)',
        'padding': '10px 5px'
    }),

    html.Div([
        dcc.Graph(
            id='crossfilter-indicator-scatter',
            hoverData={'points': [{'customdata': 'Design_1'}]}
        )
    ], style={'width': '65%', 'display': 'inline-block', 'padding': '0 20'}),

    html.Div([
        dcc.Graph(id='back-prop'),
        dcc.Graph(id='performance-table'),
    ], style={'display': 'inline-block', 'width': '34%'}),

    # html.Div(dcc.Slider(
    #     id='crossfilter-year--slider',
    #     min=df['Year'].min(),
    #     max=df['Year'].max(),
    #     value=df['Year'].max(),
    #     marks={str(year): str(year) for year in df['Year'].unique()}
    # ), style={'width': '49%', 'padding': '0px 20px 20px 20px'})
])


def normalize_to_range(x, lb, ub):
    x = np.array(x)
    x_norm = (x - x.min()) / (x.max() - x.min())
    return list(x_norm*(ub - lb) + lb)


@app.callback(
    dash.dependencies.Output('crossfilter-indicator-scatter', 'figure'),
    [dash.dependencies.Input('crossfilter-xaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-yaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-xaxis-type', 'value'),
     dash.dependencies.Input('crossfilter-yaxis-type', 'value'),
     dash.dependencies.Input('crossfilter-color', 'value'),
     dash.dependencies.Input('crossfilter-size', 'value')
     # dash.dependencies.Input('crossfilter-year--slider', 'value')
     ])
def update_graph(xaxis_column_name, yaxis_column_name,
                 xaxis_type, yaxis_type,
                 color_option, size_option):
    dff = df

    return {
        'data': [go.Scatter(
            x=list(dff[xaxis_column_name]),
            y=list(dff[yaxis_column_name]),
            text=['Design_{}'.format(x) for x in list(dff['Design ID'])],
            customdata=['Design_{}'.format(x) for x in list(dff['Design ID'])],
            mode='markers',
            marker={
                'color': list(dff[color_option]),
                'size': normalize_to_range(list(dff[size_option]), 5, 15),
                'opacity': 1.0,
                'line': {'width': 0.5, 'color': 'white'},
                'colorscale': 'Reds',
                # 'reversescale': True,
                # 'showscale': True,
            }
        )],
        'layout': go.Layout(
            xaxis={
                'title': xaxis_column_name,
                'type': 'linear' if xaxis_type == 'Linear' else 'log'
            },
            yaxis={
                'title': yaxis_column_name,
                'type': 'linear' if yaxis_type == 'Linear' else 'log'
            },
            margin={'l': 40, 'b': 30, 't': 10, 'r': 10},
            height=450,
            hovermode='closest'
        )
    }


def create_backprob_history(dff, title):
    return {
        'data': [go.Scatter(
            x=list(range(1, dff.shape[0])),
            y=list(dff['Loss']),
            mode='lines+markers',
            name='Loss'
        ),
            go.Scatter(
                x=list(range(1, dff.shape[0])),
                y=list(np.array(list(dff['Accuracy']))/100.0),
                mode='lines+markers',
                name='Accuracy'
            )
        ],
        'layout': {
            'height': 225,
            'margin': {'l': 20, 'b': 30, 'r': 10, 't': 10},
            'annotations': [{
                'x': 0, 'y': 0.85, 'xanchor': 'left', 'yanchor': 'bottom',
                'xref': 'paper', 'yref': 'paper', 'showarrow': False,
                'align': 'left', 'bgcolor': 'rgba(255, 255, 255, 0.5)',
                'text': title
            }],
            'yaxis': {'type': 'linear', 'range': [0, 1.2]},
            'xaxis': {'title': 'Epoch', 'showgrid': True},
            'legend': dict(
                x=0.7,
                y=0.5,
                traceorder='normal',
                font=dict(
                    family='sans-serif',
                    size=12,
                    color='#000'
                ),
                bgcolor='#E2E2E2',
                bordercolor='#FFFFFF',
                borderwidth=2
            )
        }
    }


def create_design_table(dff):
    values = [['Design ID', 'Genome', 'Active Nodes', 'Accuracy', '# of Params',
               '# of FLOPs', 'Robustness'],
              [list(dff['Design ID']), list(dff['Genome']),
               list(dff['Active Nodes']), '{}%'.format(list(dff['Accuracy'])[0]),
               '{} M'.format(list(dff['Params'])[0]),
               '{} M'.format(list(dff['FLOPs'])[0]), list(dff['Robustness'])]]
    return {
        'data': [go.Table(
            columnorder=[1, 2],
            columnwidth=[150, 300],
            header=dict(
                values=[['<b>ATTRIBUTES</b>'],
                        ['<b>VALUES</b>']],
                line=dict(color='#506784'),
                fill=dict(color='#119DFF'),
                align=['left', 'center'],
                font=dict(color='white', size=12),
                height=35
            ),
            cells=dict(
                values=values,
                line=dict(color='#506784'),
                fill=dict(color=['#25FEFD', 'white']),
                align=['left', 'center'],
                font=dict(color='#506784', size=12),
                height=30
            )
        )
        ],
        'layout': {
            'height': 225,
            'margin': {'l': 20, 'b': 30, 'r': 10, 't': 10},
            'annotations': [{
                'x': 0.0, 'y': -0.85, 'xanchor': 'left', 'yanchor': 'bottom',
                'xref': 'paper', 'yref': 'paper', 'showarrow': False,
                'align': 'left', 'bgcolor': 'rgba(255, 255, 255, 0.5)',
                'text': '{}'.format([]),
            }],
        }
    }


@app.callback(
    dash.dependencies.Output('back-prop', 'figure'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData'),
     # dash.dependencies.Input('crossfilter-xaxis-column', 'value'),
     # dash.dependencies.Input('crossfilter-xaxis-type', 'value')
     ])
def update_design_backprop(hoverData):
    design_id = hoverData['points'][0]['customdata']
    df_backprob = pd.read_csv(os.path.join(results_root,
                                           design_id,
                                           'TrainLogger.txt'),
                              sep='\t', lineterminator='\n')
    title = '<b>{}</b><br>{}'.format(design_id, 'Back-propagation training log')
    return create_backprob_history(df_backprob, title)


@app.callback(
    dash.dependencies.Output('performance-table', 'figure'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData'),
     # dash.dependencies.Input('crossfilter-xaxis-column', 'value'),
     # dash.dependencies.Input('crossfilter-xaxis-type', 'value')
     ])
def update_design_table(hoverData):
    design = hoverData['points'][0]['customdata']
    design_id = eval(design[int(design.find('_')+1):])
    dff = df[df['Design ID'] == design_id]
    # title = '{} {}'.format(design, 'Details')
    return create_design_table(dff)

@app.callback(
    dash.dependencies.Output('image', 'src'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData')])
def update_image_src(hoverData):
    return static_image_route + '{}.png'.format(hoverData['points'][0]['customdata'])


# Add a static image route that serves images from desktop
# Be *very* careful here - you don't want to serve arbitrary files
# from your computer or server
@app.server.route('{}<image_path>.png'.format(static_image_route))
def serve_image(image_path):
    image_name = '{}.png'.format(image_path)
    if image_name not in list_of_images:
        raise Exception('"{}" is excluded from the allowed static files'.format(image_path))
    return flask.send_from_directory(image_directory, image_name)


if __name__ == '__main__':
    app.run_server()