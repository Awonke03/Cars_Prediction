#!/usr/bin/env python
# coding: utf-8

# In[530]:


import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
import dash_table
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns


# In[531]:
url = "https://raw.githubusercontent.com/Awonke03/Cars_Prediction/4b18890d6645b97c3caafe73489c4e226f58a3a6/car%20data.csv"

# Read CSV into Pandas DataFrame
cars= pd.read_csv(url)
cars.sample(20)


# In[532]:


cars.shape


# In[533]:


cars.Transmission.unique()


# In[534]:


cars.Car_Name.unique()


# In[535]:


cars.info()


# In[536]:


prediction_screen_layout = html.Div([
    html.H1('Prediction Screen Page', style={'text-align': 'center', 'color': 'orange'}),
    # Add content for this page
    dbc.Row([
        dbc.Col(
            dbc.ButtonGroup(
                [
                    dbc.Button('Interactive Search in Visuals', href='/interactive-search-visuals', active=True, className='mb-2 rounded-pill', style={'background-color': 'rgb(54, 54, 54)', 'color': 'white', 'transition-duration': '0.4s'}),
                    dbc.Button('Visualizations', href='/visualizations', active=True, className='mb-2 rounded-pill', style={'background-color': 'rgb(54, 54, 54)', 'color': 'white', 'transition-duration': '0.4s'}),
                    dbc.Button('Prediction Screen', href='/prediction-screen', active=True, className='mb-2 rounded-pill', style={'background-color': 'rgb(54, 54, 54)', 'color': 'white', 'transition-duration': '0.4s'}),
                    dbc.Button('Search in Dataset', href='/search-dataset', active=True, className='mb-2 rounded-pill', style={'background-color': 'rgb(54, 54, 54)', 'color': 'white', 'transition-duration': '0.4s'}),
                    dbc.Button('About', href='/about', active=True, className='mb-2 rounded-pill', style={'background-color': 'rgb(54, 54, 54)', 'color': 'white', 'transition-duration': '0.4s'}),
                ],
            ),
            width=12, style={'justify-content': 'center'}
        ),
    ]),
], style={'background-color': 'rgb(54, 54, 54)', 'color': 'white'})


# In[537]:


fig1 = px.scatter(cars, x='Driven_kms', y='Selling_Price', size='Selling_Price', color=None, 
                 title='Bubble Plot of Selling Price by Kilometers Driven')
fig1.show()


# In[538]:


fig2= px.area(cars, x='Year', y='Selling_Price', color='Fuel_Type')

fig2.update_layout(title='Selling Price Trends by Year and Fuel Type')

fig2.show()


# In[539]:


fuel_type_counts = cars['Fuel_Type'].value_counts()

# Creating the donut chart
fig3= px.pie(fuel_type_counts, 
             names=fuel_type_counts.index, 
             title='Fuel Type Distribution', 
             hole=0.5)

# Showing the donut chart
fig3.show()


# In[540]:


fig4 = px.bar(cars, 
              x='Year', 
              y='Selling_Price', 
              color='Transmission', 
              barmode='stack',
              title='Selling Price by Year and Transmission',
              labels={'Selling_Price': 'Selling Price', 'Year': 'Year'},
              template='plotly_white',
              color_discrete_map={'Manual': '#FFD700', 'Automatic': '#4B0082'})  

fig4.update_layout(
    xaxis=dict(title='Year'),
    yaxis=dict(title='Selling Price'),
)

fig4.show()


# In[541]:


fig6 = px.scatter(cars, 
                 x='Present_Price', 
                 y='Selling_Price', 
                 size='Year', 
                 color='Year', 
                 hover_name='Car_Name', 
                 trendline='ols',
                 title='Selling Price vs Present Price by Year',
                 labels={'Present_Price': 'Present Price', 'Selling_Price': 'Selling Price', 'Year': 'Year'},
                 template='plotly_white')

fig6.update_layout(
    xaxis=dict(title='Present Price'),  
    yaxis=dict(title='Selling Price'),  
)

fig6.show()


# In[542]:


fig5 = px.histogram(cars, x='Year', y='Selling_Price', title='Selling Price Over Time by Selling Type',
                    labels={'Year': 'Year', 'Selling_Price': 'Selling Price', 'Selling_type': 'Selling Type'},
                    color='Selling_type', barmode='stack', 
                    color_discrete_sequence=['#00BFFF', '#FFD700', '#FFF44F'])

fig5.update_layout(bargap=0.05, bargroupgap=0.1, xaxis_tickangle=-45,
                   xaxis=dict(showline=True, showgrid=True, showticklabels=True, linecolor='rgb(204, 204, 204)', linewidth=2, ticks='outside', tickfont=dict(family='Arial', size=12, color='rgb(82, 82, 82)')),
                   yaxis=dict(showline=True, showgrid=True, showticklabels=True, linecolor='rgb(204, 204, 204)', linewidth=2, ticks='outside', tickfont=dict(family='Arial', size=12, color='rgb(82, 82, 82)')),
                   legend=dict(x=0.02, y=0.98),
                   barmode='stack')

fig5.show()


# In[543]:


fig7 = px.scatter(cars, x='Driven_kms', y='Selling_Price', size='Selling_Price', hover_name='Car_Name', log_x=True, color='Year')

fig7.update_layout(title='Selling Price vs Driven Kilometers',
                  xaxis_title='Driven Kilometers',
                  yaxis_title='Selling Price',
                  coloraxis_colorbar=dict(title='Year'),
                  margin=dict(t=40)  
                 )


fig7.show()


# In[544]:


# Plot 8: Sunburst Plot for Fuel Type, Selling Type, Transmission, Year, and Car Name
fig8 = px.sunburst(cars, path=['Fuel_Type', 'Selling_type', 'Transmission', 'Year', 'Car_Name'], values='Selling_Price',
                   title='Car Price Breakdown by Fuel Type, Selling Type, Transmission, Year, and Car Name')
fig8.show()


# In[545]:


# Layout for the Visualizations page
visualizations_layout = html.Div([
    html.H1('Visualizations Page', style={'text-align': 'center', 'color': 'orange'}),
    
    # Navigation Buttons
    dbc.Row([
        dbc.Col(
            dbc.ButtonGroup(
                [
                    dbc.Button('Interactive Search in Visuals', href='/interactive-search-visuals', active=True, className='mb-2 rounded-pill', style={'background-color': 'rgb(54, 54, 54)', 'color': 'white', 'transition-duration': '0.4s'}),
                    dbc.Button('Visualizations', href='/visualizations', active=True, className='mb-2 rounded-pill', style={'background-color': 'rgb(54, 54, 54)', 'color': 'white', 'transition-duration': '0.4s'}),
                    dbc.Button('Prediction Screen', href='/prediction-screen', active=True, className='mb-2 rounded-pill', style={'background-color': 'rgb(54, 54, 54)', 'color': 'white', 'transition-duration': '0.4s'}),
                    dbc.Button('Search in Dataset', href='/search-dataset', active=True, className='mb-2 rounded-pill', style={'background-color': 'rgb(54, 54, 54)', 'color': 'white', 'transition-duration': '0.4s'}),
                    dbc.Button('About', href='/about', active=True, className='mb-2 rounded-pill', style={'background-color': 'rgb(54, 54, 54)', 'color': 'white', 'transition-duration': '0.4s'}),
                ],
            ),
            width=12, style={'justify-content': 'center'}
        ),
    ]),
    

    dbc.Row([
        dbc.Col(dcc.Graph(figure=fig3), width=6),  
        dbc.Col(dcc.Graph(figure=fig2), width=6), 
    ], style={'margin-top': '20px'}), 
    
    dbc.Row([
        dbc.Col(dcc.Graph(figure=fig1), width=6), 
        dbc.Col(dcc.Graph(figure=fig4), width=6),  
    ], style={'margin-top': '20px'}),  
    
    
    dbc.Row([
        dbc.Col(dcc.Graph(figure=fig5), width=6),  
        dbc.Col(dcc.Graph(figure=fig6), width=6),  
    ], style={'margin-top': '20px'}),  
   
    dbc.Row([
        dbc.Col(dcc.Graph(figure=fig7), width=6),  
        dbc.Col(dcc.Graph(figure=fig8), width=6), 
    ], style={'margin-top': '20px'}), 
], style={'background-color': 'white', 'color': 'black'}) 


# In[546]:


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = 'Get a good car'


interactive_search_layout = html.Div([
    html.H1('Interactive Search in Visuals Page', style={'text-align': 'center', 'color': 'orange'}),
 
    dbc.Row([
        dbc.Col(
            dbc.ButtonGroup(
                [
                    dbc.Button('Interactive Search in Visuals', href='/interactive-search-visuals', active=True, className='mb-2 rounded-pill', style={'background-color': 'rgb(54, 54, 54)', 'color': 'white', 'transition-duration': '0.4s'}),
                    dbc.Button('Visualizations', href='/visualizations', active=True, className='mb-2 rounded-pill', style={'background-color': 'rgb(54, 54, 54)', 'color': 'white', 'transition-duration': '0.4s'}),
                    dbc.Button('Prediction Screen', href='/prediction-screen', active=True, className='mb-2 rounded-pill', style={'background-color': 'rgb(54, 54, 54)', 'color': 'white', 'transition-duration': '0.4s'}),
                    dbc.Button('Search in Dataset', href='/search-dataset', active=True, className='mb-2 rounded-pill', style={'background-color': 'rgb(54, 54, 54)', 'color': 'white', 'transition-duration': '0.4s'}),
                    dbc.Button('About', href='/about', active=True, className='mb-2 rounded-pill', style={'background-color': 'rgb(54, 54, 54)', 'color': 'white', 'transition-duration': '0.4s'}),
                ],
            ),
            width=12, style={'justify-content': 'center'}
        ),
    ]),
], style={'background-color': 'rgb(54, 54, 54)', 'color': 'white'})





search_dataset_layout = html.Div([
    html.H1('Search in Dataset Page', style={'text-align': 'center', 'color': 'orange'}),
    dcc.Dropdown(
        id='search_dropdown',
        options=[{'label': i, 'value': i} for i in ['Car_Name', 'Year', 'Transmission', 'Fuel_Type']],
        placeholder="Select Column to Search...",
        multi=False
    ),
    dcc.Input(
        id='search_input',
        placeholder='Enter search term...',
        type='text',
        value=''
    ),
    html.Button('Search', id='search_button', n_clicks=0),
    html.Button('Reset', id='reset_button', n_clicks=0),
    dash_table.DataTable(
        id='datatable',
        columns=[{"name": i, "id": i} for i in cars.columns],
        data=cars.to_dict('records'),
        page_size=15,  # Display 10 records per page
        style_table={'background-color': 'darkgrey', 'padding': '10px'},
        style_cell={'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white'},
        style_header={'backgroundColor': 'rgb(30, 30, 30)', 'fontWeight': 'bold'},
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(40, 40, 40)'
            },
            {
                'if': {'column_id': 'Car_Name'},
                'textAlign': 'left'
            }
        ],
        style_cell_conditional=[
            {
                'if': {'column_id': 'search_button'},
                'width': '10%'
            },
            {
                'if': {'column_id': 'reset_button'},
                'width': '10%'
            }
        ]
    ),
    dbc.Row([
        dbc.Col(
            dbc.ButtonGroup(
                [
                    dbc.Button('Interactive Search in Visuals', href='/interactive-search-visuals', active=True, className='mb-2 rounded-pill', style={'background-color': 'rgb(54, 54, 54)', 'color': 'white', 'transition-duration': '0.4s'}),
                    dbc.Button('Visualizations', href='/visualizations', active=True, className='mb-2 rounded-pill', style={'background-color': 'rgb(54, 54, 54)', 'color': 'white', 'transition-duration': '0.4s'}),
                    dbc.Button('Prediction Screen', href='/prediction-screen', active=True, className='mb-2 rounded-pill', style={'background-color': 'rgb(54, 54, 54)', 'color': 'white', 'transition-duration': '0.4s'}),
                    dbc.Button('Search in Dataset', href='/search-dataset', active=True, className='mb-2 rounded-pill', style={'background-color': 'rgb(54, 54, 54)', 'color': 'white', 'transition-duration': '0.4s'}),
                    dbc.Button('About', href='/about', active=True, className='mb-2 rounded-pill', style={'background-color': 'rgb(54, 54, 54)', 'color': 'white', 'transition-duration': '0.4s'}),
                ],
            ),
            width=12, style={'justify-content': 'center'}
        ),
    ]),
], style={'background-color': 'rgb(54, 54, 54)', 'color': 'white'})

about_layout = dbc.Card(
    dbc.CardBody([
        html.H1('About Page', style={'text-align': 'center', 'color': 'orange'}),
        dbc.Row([
            dbc.Col(
                html.Img(src="C:\\Users\\Admin\\Desktop\\Programming\\python\\Car Prediction\\mine.JPG", style={'width': '200px', 'height': '200px', 'border-radius': '50%', 'margin': 'auto'}),
                width=3
            ),
            dbc.Col(
                dbc.Row([
                    dbc.Col(html.I(className='fa fa-envelope'), width=1),
                    dbc.Col(html.P('nomandondo.awonke@outlook.com'), width=11),
                ]),
                width=12
            ),
            dbc.Col(
                dbc.Row([
                    dbc.Col(html.I(className='fa fa-phone'), width=1),
                    dbc.Col(html.P('+27 (0) 789751929'), width=11),
                ]),
                width=12
            ),
            dbc.Col(
                dbc.Row([
                    dbc.Col(html.I(className='fa fa-map-marker'), width=1),
                    dbc.Col(html.P('PO Box 369, Qumbu 5180'), width=11),
                ]),
                width=12
            ),
            dbc.Col(
                html.P('No machine can do the work of one extraordinary man. For a successful technology, reality must take precedence over public relations, for Nature cannot be fooled. Any sufficiently advanced technology is indistinguishable from magic.'),
                width=12
            )
        ])
    ]),
    style={'background-color': 'rgb(200, 200, 200)', 'color': 'black'}
)


app.layout = dbc.Container(
    [
        dcc.Location(id='url', refresh=False),
        dbc.Row([
            dbc.Col(
                html.Div(id='page-content')
            ),
        ], style={'height': '100vh', 'background-color': 'rgb(54, 54, 54)', 'justify-content': 'center'}),
    ],
    fluid=True
)

# Callback to update the page content based on the URL
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/interactive-search-visuals':
        return interactive_search_layout
    elif pathname == '/visualizations':
        return visualizations_layout
    elif pathname == '/prediction-screen':
        return prediction_screen_layout
    elif pathname == '/search-dataset':
        return search_dataset_layout
    elif pathname == '/about':
        return about_layout
    else:
        return interactive_search_layout

# Callback to update the data table based on dropdown selections
@app.callback(
    [Output('datatable', 'data'),
     Output('search_dropdown', 'value'),
     Output('search_input', 'value')],
    [Input('search_button', 'n_clicks'),
     Input('reset_button', 'n_clicks')],
    [State('search_dropdown', 'value'),
     State('search_input', 'value')]
)
def update_table(search_clicks, reset_clicks, column_name, search_term):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'search_button':
            # Handle search button click
            filtered_data = cars.copy()
            if column_name and search_term:
                filtered_data = filtered_data[filtered_data[column_name].astype(str).str.contains(search_term, case=False)]
            return filtered_data.to_dict('records'), dash.no_update, dash.no_update
        elif button_id == 'reset_button':
            # Handle reset button click
            return cars.to_dict('records'), None, None

# Add CSS to apply hover effect
app.css.append_css({
    'external_url': (
        'https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css'
    )
})
if __name__ == "__main__":
    app.run_server(debug=True, port=8020)

