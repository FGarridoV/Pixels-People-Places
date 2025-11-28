from socket import timeout
from jupyter_dash import JupyterDash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State

import multiprocessing

class InteractiveMap:

    zoom_map = 10

    def __init__(self, map_center):
        self.study_area_points = None
        self.click_study_area = 0
        self.center = map_center
        self.process = None
        self.app = JupyterDash(__name__)
        self.set_layout()
        self.add_callback()

    def set_layout(self):
        figure = go.Figure(
                     go.Scattermapbox(),
                     layout = go.Layout(
                         mapbox = {'style': "open-street-map", 
                                   'center': self.center, 
                                   'zoom': InteractiveMap.zoom_map},
                         showlegend = False,
                         selectdirection = 'd',
                         dragmode = 'select',
                         modebar = {'activecolor':'rgb(63,129,220)', 
                                    'uirevision': 'select'},
                         margin=dict(l=5, r=5, b=5, t=35, pad=4)))

        graph = dcc.Graph(
                    style = {'border-style': 'solid', 
                             'border-color': 'lightgrey',
                             'border-width': 'thin'},
                    figure = figure,
                    id = 'graph',
                    className="mb-1",
                    config={'modeBarButtonsToRemove':['toImage', 'hoverClosestPie', 'toggleHover'],
                            'modeBarButtonsToAdd': ['lasso2d', 'select2d'],
                            'displayModeBar': True,
                            'displaylogo': False,
                            'frameMargins': 0})
        
        buttons = dbc.Button(
                      children = 'Set study area', 
                      id = 'set-study-area-button',
                      color = 'primary',
                      size = 'bg')

        self.app.layout = html.Div([buttons, graph])

    def add_callback(self):
        @self.app.callback(
            Output('graph','figure'),
            [Input('graph','selectedData'),
            Input("set-study-area-button", "n_clicks"),
            State('graph','figure')])
        def draw_study_area(study_area, click, current_figure):
            if study_area != None:
                try:
                    p = study_area['range']['mapbox']
                    lon_min = min(p[0][0], p[1][0])
                    lon_max = max(p[0][0], p[1][0])
                    lat_min = min(p[0][1], p[1][1])
                    lat_max = max(p[0][1], p[1][1])
                    lons = [lon_min, lon_max, lon_max , lon_min]
                    lats = [lat_min, lat_min, lat_max, lat_max]
                except:
                    p = study_area['lassoPoints']['mapbox']
                    lons = [x[0] for x in p]
                    lats = [x[1] for x in p]
                if 0 if click == None else click > self.click_study_area:
                    self.study_area_points = [[x,y] for x,y in zip(lons, lats)]
                    newfig = go.Figure(go.Scattermapbox(
                    fill = "toself",
                    lon = lons, lat = lats,
                    marker = { 'size': 2, 'color': "green" }))
                    newfig.update_layout(current_figure['layout'])
                    self.click_study_area += 1
                    return newfig
                else:
                    self.study_area_points = None
                    newfig = go.Figure(go.Scattermapbox(
                    fill = "toself",
                    lon = lons, lat = lats,
                    marker = { 'size': 2, 'color': "orange" }))
                    newfig.update_layout(current_figure['layout'])
                    return newfig
            else: 
                new_fig = go.Figure(
                    go.Scattermapbox(),
                    layout = go.Layout(
                        mapbox = {'style': "open-street-map", 
                                'center': self.center,
                                'zoom': InteractiveMap.zoom_map},
                        showlegend = False,
                        selectdirection = 'd',
                        dragmode = 'select',
                        modebar = {'activecolor':'rgb(63,129,220)', 'uirevision': 'select'},
                        margin=dict(l=5, r=5, b=5, t=35, pad=4)))
                return new_fig
    
    def run_app(self):
        self.process = multiprocessing.Process(target=self.app.run_server(mode='inline'), name="App", kwargs=dict(debug=False))
        self.process.start()

    
    def close_app(self):
        self.process.terminate()
        self.process.join()

        #self.app.run_server(mode='inline',)
        


