#!/usr/bin/env python
# coding: utf-8

# In[22]:


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
from jupyter_dash import JupyterDash
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error
import pickle
from sklearn.model_selection import GridSearchCV


# In[23]:

# Replace the URL with the raw GitHub URL if needed
url = "https://raw.githubusercontent.com/Awonke03/Cars_Prediction/90aacff7219a6e1983586d438cd8ef650084523b/car%20data.csv"

# Load the dataset into a DataFrame
cars = pd.read_csv(url)
cars.sample(20)


# In[24]:


cars.shape


# In[25]:


cars.Transmission.unique()


# In[26]:


cars.Car_Name.unique()


# In[27]:


cars.info()


# In[28]:


app = JupyterDash(__name__, external_stylesheets=[dbc.themes.SLATE], suppress_callback_exceptions=True)
app.title = 'Get a good car'


# In[29]:


fig1 = px.scatter(cars, x='Driven_kms', y='Selling_Price', size='Selling_Price', color=None, 
                 title='Bubble Plot of Selling Price by Kilometers Driven')
fig1.show()


# In[30]:


fig2= px.area(cars, x='Year', y='Selling_Price', color='Fuel_Type')

fig2.update_layout(title='Selling Price Trends by Year and Fuel Type')

fig2.show()


# In[31]:


fuel_type_counts = cars['Fuel_Type'].value_counts()

# Creating the donut chart
fig3= px.pie(fuel_type_counts, 
             names=fuel_type_counts.index, 
             title='Fuel Type Distribution', 
             hole=0.5)

# Showing the donut chart
fig3.show()


# In[32]:


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


# In[33]:


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


# In[34]:


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


# In[35]:


fig7 = px.scatter(cars, x='Driven_kms', y='Selling_Price', size='Selling_Price', hover_name='Car_Name', log_x=True, color='Year')

fig7.update_layout(title='Selling Price vs Driven Kilometers',
                  xaxis_title='Driven Kilometers',
                  yaxis_title='Selling Price',
                  coloraxis_colorbar=dict(title='Year'),
                  margin=dict(t=40)  
                 )


fig7.show()


# In[36]:


# Plot 8: Sunburst Plot for Fuel Type, Selling Type, Transmission, Year, and Car Name
fig8 = px.sunburst(cars, path=['Fuel_Type', 'Selling_type', 'Transmission', 'Year', 'Car_Name'], values='Selling_Price',
                   title='Car Price Breakdown by Fuel Type, Selling Type, Transmission, Year, and Car Name')
fig8.show()


# In[37]:


# Layout for the Visualizations page
visualizations_layout = html.Div([
    html.H1('Visualizations Page', style={'text-align': 'center', 'color': 'orange'}),
    
    # Navigation Buttons
    dbc.Row([
        dbc.Col(
            dbc.ButtonGroup(
                [
                    #dbc.Button('Interactive Search in Visuals', href='/interactive-search-visuals', active=True, className='mb-2 rounded-pill', style={'background-color': 'rgb(54, 54, 54)', 'color': 'white', 'transition-duration': '0.4s'}),
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
], style={'background-color': 'rgba(120, 120, 120, 0.4)', 'color': 'rgba(120, 120, 120, 0.4)'}) 


# In[38]:


about_layout = dbc.Card(
    dbc.CardBody([
        html.H1('About Page', style={'text-align': 'center', 'color': 'orange'}),
        dbc.Row([
            dbc.Col(
                html.Img(src='"C:\\Users\\Admin\\Desktop\\Programming\\python\\Car Prediction\\mine.JPG"', style={'width': '100%', 'height': 'auto', 'borderRadius': '15px', 'objectFit': 'cover'}),
                width=12
            ),
            dbc.Col(
                html.H4('Awonke Nomandondo', className='text-center mb-3 p-3'),
                width=12
            ),
            dbc.Col(
                html.A(html.I(className='fab fa-linkedin'), href='https://www.linkedin.com/in/awonke-nomandondo-637548239'),
                width=12
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
        ]),
        html.Hr(),  # Add a horizontal line before the buttons
        dbc.Row([
            dbc.Col(
                dbc.ButtonGroup(
                    [
                        dbc.Button('Visualizations', href='/visualizations', active=True, className='mb-2 rounded-pill', style={'background-color': 'rgb(54, 54, 54)', 'color': 'white', 'transition-duration': '0.4s'}),
                        dbc.Button('Prediction Screen', href='/prediction-screen', active=True, className='mb-2 rounded-pill', style={'background-color': 'rgb(54, 54, 54)', 'color': 'white', 'transition-duration': '0.4s'}),
                        dbc.Button('Search in Dataset', href='/search-dataset', active=True, className='mb-2 rounded-pill', style={'background-color': 'rgb(54, 54, 54)', 'color': 'white', 'transition-duration': '0.4s'}),
                        dbc.Button('About', href='/about', active=True, className='mb-2 rounded-pill', style={'background-color': 'rgb(54, 54, 54)', 'color': 'white', 'transition-duration': '0.4s'}),
                    ],
                ),
                width=12, style={'justify-content': 'center'}
            ),
        ]),
    ]),
    style={'background-color': 'rgba(120, 120, 120, 0.4)', 'color': 'black', 'width': '50%', 'margin': 'auto', 'z-index': '2'}
)


# In[39]:


numerical_cols = cars.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = cars.select_dtypes(include=['object']).columns

# Impute missing values for numerical columns
imputer = SimpleImputer(strategy='mean')
cars[numerical_cols] = imputer.fit_transform(cars[numerical_cols])

# Encode categorical columns (one-hot encoding)
cars = pd.get_dummies(cars, drop_first=True)

# Split the data into features (X) and target variable (y)
X = cars.drop(columns=['Selling_Price'])
y = cars['Selling_Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameters grid for Random Forest
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Define hyperparameters grid for Decision Tree
param_grid_dt = {
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Define models with default parameters
models = {
    'Random Forest': RandomForestRegressor(),
    'Decision Tree': DecisionTreeRegressor(),
    'Linear Regression': LinearRegression()
}

# Initialize model_metrics dictionary
model_metrics = {}

# Add Random Forest and Decision Tree with parameter tuning
for name, param_grid in [('Random Forest', param_grid_rf), ('Decision Tree', param_grid_dt)]:
    grid_search = GridSearchCV(models[name], param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    model_metrics[name + ' (Tuned)'] = {'mse_train': mse_train, 'mse_test': mse_test}
    with open(f'{name}_tuned_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)

# Display model metrics
for model_name, metrics in model_metrics.items():
    print(f"Model: {model_name}")
    print(f"Train MSE: {metrics['mse_train']}")
    print(f"Test MSE: {metrics['mse_test']}")
    print()


# In[40]:


# Extract model names, mse_train, and mse_test (excluding Naive Bayes)
model_names = list(model_metrics.keys())
mse_train_values = [model_metrics[model]['mse_train'] for model in model_names]
mse_test_values = [model_metrics[model]['mse_test'] for model in model_names]

# Create a bar plot for MSE
mse_fig = go.Figure(data=[
    go.Bar(name='Train MSE', x=model_names, y=mse_train_values),
    go.Bar(name='Test MSE', x=model_names, y=mse_test_values)
])
mse_fig.update_layout(title='Mean Squared Error Comparison', xaxis_title='Model', yaxis_title='MSE', barmode='group')

# Plot the figure
mse_fig.show()

# Extract accuracies (excluding Naive Bayes)
accuracies = [1 - mse_test / y_test.var() for mse_test in mse_test_values]

# Create a bar plot for accuracies
accuracy_fig = px.bar(x=model_names, y=accuracies, labels={'x': 'Model', 'y': 'Accuracy'}, title='Accuracy Comparison')
accuracy_fig.show()


# In[41]:


# Load the Random Forest model
with open('Random Forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the layout
prediction_screen_layout = dbc.Container([
    html.H1('Prediction Screen Page', style={'text-align': 'center', 'color': 'orange'}),
    dbc.Row([
        dbc.Col(
            dbc.Form([
                html.Div([
                    dbc.Label('Car Name', className='mr-2'),
                    dcc.Input(id='car-name', type='text', placeholder='Enter the car name', pattern='^.*\S.*$')
                ], className='mb-3'),
                html.Div([
                    dbc.Label('Year', className='mr-2'),
                    dcc.Input(id='year', type='number', placeholder='Enter the year')
                ], className='mb-3'),
                html.Div([
                    dbc.Label('Selling Price', className='mr-2'),
                    dcc.Input(id='selling-price', type='number', placeholder='Enter the selling price')
                ], className='mb-3'),
                html.Div([
                    dbc.Label('Present Price', className='mr-2'),
                    dcc.Input(id='present-price', type='number', placeholder='Enter the present price')
                ], className='mb-3'),
                html.Div([
                    dbc.Label('Driven Kms', className='mr-2'),
                    dcc.Input(id='driven-kms', type='number', placeholder='Enter the driven kms')
                ], className='mb-3'),
                html.Div([
                    dbc.Label('Fuel Type', className='mr-2'),
                    dcc.Input(id='fuel-type', type='text', placeholder='Enter the fuel type')
                ], className='mb-3'),
                html.Div([
                    dbc.Label('Selling Type', className='mr-2'),
                    dcc.Input(id='selling-type', type='text', placeholder='Enter the selling type')
                ], className='mb-3'),
                html.Div([
                    dbc.Label('Transmission', className='mr-2'),
                    dcc.Input(id='transmission', type='text', placeholder='Enter the transmission')
                ], className='mb-3'),
                html.Div([
                    dbc.Label('Owner', className='mr-2'),
                    dcc.Input(id='owner', type='number', placeholder='Enter the owner')
                ], className='mb-3'),
            ]),
            width=12, style={'justify-content': 'center'}
        ),
    ]),
    # Add a button to trigger the prediction
    dbc.Button('Predict', id='predict-button', color='primary', className='mb-2'),
    
    # Add a div to display the predicted selling price
    html.Div(id='predicted-price', style={'text-align': 'center', 'color': 'orange'}),
    
    # Add the navigation buttons
    dbc.Row([
        dbc.Col(
            dbc.ButtonGroup(
                [
                    dbc.Button('Visualizations', href='/visualizations', active=True, className='mb-2 rounded-pill', style={'background-color': 'rgb(54, 54, 54)', 'color': 'white', 'transition-duration': '0.4s'}),
                    dbc.Button('Prediction Screen', href='/prediction-screen', active=True, className='mb-2 rounded-pill', style={'background-color': 'rgb(54, 54, 54)', 'color': 'white', 'transition-duration': '0.4s'}),
                    dbc.Button('Search in Dataset', href='/search-dataset', active=True, className='mb-2 rounded-pill', style={'background-color': 'rgb(54, 54, 54)', 'color': 'white', 'transition-duration': '0.4s'}),
                    dbc.Button('About', href='/about', active=True, className='mb-2 rounded-pill', style={'background-color': 'rgb(54, 54, 54)', 'color': 'white', 'transition-duration': '0.4s'}),
                ],
            ),
            width=12, style={'justify-content': 'center'}
        ),
    ]),
], style={'background-color': 'rgba(120, 120, 120, 0.4)', 'color': 'white'})

# Callback to predict the selling price
@app.callback(
    Output('predicted-price', 'children'),
    Input('predict-button', 'n_clicks'),
    [
        Input('car-name', 'value'),
        Input('year', 'value'),
        Input('selling-price', 'value'),
        Input('present-price', 'value'),
        Input('driven-kms', 'value'),
        Input('fuel-type', 'value'),
        Input('selling-type', 'value'),
        Input('transmission', 'value'),
        Input('owner', 'value')
    ]
)
def predict_selling_price(n_clicks, car_name, year, selling_price, present_price, driven_kms, fuel_type, selling_type, transmission, owner):
    if not all([car_name, year, selling_price, present_price, driven_kms, fuel_type, selling_type, transmission, owner]):
        return 'Please fill in all fields'
    
    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'Car_Name': [car_name],
        'Year': [year],
        'Selling_Price': [selling_price],
        'Present_Price': [present_price],
        'Kms_Driven': [driven_kms],
        'Fuel_Type': [fuel_type],
        'Seller_Type': [selling_type],
        'Transmission': [transmission],
        'Owner': [owner]
    })
    
    # Perform prediction
    predicted_price = model.predict(input_data)[0]
    
    return f'Predicted selling price: ${predicted_price:.2f}'


# In[42]:


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
                   # dbc.Button('Interactive Search in Visuals', href='/interactive-search-visuals', active=True, className='mb-2 rounded-pill', style={'background-color': 'rgb(54, 54, 54)', 'color': 'white', 'transition-duration': '0.4s'}),
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


app.layout = dbc.Container(
    [
        dcc.Location(id='url', refresh=False),
        dbc.Row([
            dbc.Col(
                html.Div(id='page-content')
            ),
        ], style={'height': '100vh', 'background-color': 'rgba(120, 120, 120, 0.4)', 'justify-content': 'center'}),
    ],
    fluid=True
)

# Callback to update the page content based on the URL
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/visualizations':
        return visualizations_layout
    elif pathname == '/prediction-screen':
        return prediction_screen_layout
    elif pathname == '/search-dataset':
        return search_dataset_layout
    elif pathname == '/about':
        return about_layout
    else:
        return visualizations_layout

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

