#!/usr/bin/env python
# coding: utf-8

# In[1]:


import plotly.graph_objects as go
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import random
from perlin_noise import PerlinNoise
import heapq


# In[2]:


def get_point_coordinate(f, x, xm, delx):
    x_low = max(x - int((f * xm)// delx * delx), 0)
    x_high = min(x + int((f * xm)//delx * delx), xm)
    arr1 = np.arange(0, x_low, delx)
    arr2 = np.arange(x_high, xm, delx)
    arr = np.concatenate((arr1, arr2))
    return arr[random.randrange(len(arr))]


# In[3]:


def euclidean_dist(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
def segment_length(a, b):
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2)


# In[4]:


def a_less_than_b(a, b):
    return 1 if (a <= b) else -1;


# In[5]:


def generate_terrain(gap, fz, n1, n2, n3, rw, pc, ec):
    x_max = 1200 # in metres
    y_max = 600 # in metres
    
    nx = x_max// gap + 1 # number of points in x-direction
    ny = y_max// gap + 1 # number of points in y-direction
    x = np.linspace(0, x_max, nx) # array of x-coordinates along x-direction
    y = np.linspace(0, y_max, ny) # array of y-coordinates along y-direction
    # assuming x-direction is horizontal while y-direction is vertical
    X, Y = np.meshgrid(x, y) # generate coordinate matrix

    # 3d-land generation using perlin noise
    Z = np.zeros((ny, nx)) # "ny by nx" zero matrix that stores altitude values at different coordinates
    noise1 = PerlinNoise(octaves = n1) # more octaves implies more undulations
    noise2 = PerlinNoise(octaves = n2)
    noise3 = PerlinNoise(octaves = n3)
    for j in range(ny):
        for i in range(nx):
            Z[j][i] =         noise1([j/ ny, i/ nx])
            Z[j][i] += 0.3 *  noise2([j/ ny, i/ nx])
            Z[j][i] += 0.05 * noise3([j/ ny, i/ nx])
    Z = Z * fz

    # plot surface using plotly library
    fig = go.Figure(data = [go.Surface(
        x = X, 
        y = Y, 
        z = Z, 
        colorscale = 'turbid', 
        opacity = 1, 
        reversescale = True
    )])
    fig.update_scenes(
        aspectmode = 'data', # enables axes scaling proportionally
        xaxis = dict(nticks = nx), 
        yaxis = dict(nticks = ny)
    )

    # selection of a pair of coordinates between which road has to be built
    start_x = random.randint(0, nx-1) * gap
    start_y = random.randint(0, ny-1) * gap
    start_z = Z[start_y// gap][start_x// gap]

    f = 0.4 #consider a square with start as its center and edge length <=80% of plot size
    end_x = get_point_coordinate(f, start_x, x_max, gap)
    end_y = get_point_coordinate(f, start_y, y_max, gap) 
    end_z = Z[end_y// gap][end_x// gap]

    # plot a straight line connecting the two points
    xP = [start_x, end_x]
    yP = [start_y, end_y]
    zP = [start_z, end_z]
    
    ############# NEXT PART: ESTABLISH DIRECT PATH #############
    
    # starting point (point 1) is assumed to be the one with lesser x-value
    # the other point is the ending point (point 2)
    xP1 = int(min(xP))
    ind_xP1 = xP.index(xP1)
    yP1 = int(yP[ind_xP1])
    ind_xP2 = int(not ind_xP1)
    xP2 = int(xP[ind_xP2])
    yP2 = int(yP[ind_xP2])

    # we now connect the two points using the points that form the land (surface)
    # the vertical projection of the connecting trace will be similar to that of the straight line seen earlier

    pts_x = []
    pts_y = []
    pts_z = []
    direct_earthwork_volume = 0
    if xP1 != xP2:
        g_xy = (yP2 - yP1)/ (xP2 - xP1)
        g_xz = (Z[yP2// gap][xP2// gap] - Z[yP1// gap][xP1// gap])/ (xP2 - xP1)
        
        #x, y, z are coordinates of straight line joining start and end
        y = yP1
        z = Z[yP1// gap][xP1// gap]
        pts_x = np.arange(xP1, xP2 + 1, gap)
        pts_y.append(y)
        pts_z.append(z)
        separation = np.sqrt(gap**2 * (1 + g_xy**2))
        prev_delz = 0
        for x in np.arange(xP1 + gap, xP2 + 1, gap):
            y += g_xy * gap
            z += g_xz * gap
    
            zg = (y/gap - y// gap) * Z[min(int(y// gap) + 1, ny-1)][x// gap] + ((y// gap + 1) - y/gap) * Z[max(int(y// gap), 0)][x// gap]
            curr_delz = z - zg 
            if (prev_delz * curr_delz >= 0):
                direct_earthwork_volume += 0.5 * abs(prev_delz + curr_delz) * separation
            else:
                direct_earthwork_volume += 0.5 * (prev_delz ** 2 + curr_delz ** 2)/ (abs(prev_delz) + abs(curr_delz)) * separation

            pts_y.append(y)
            pts_z.append(zg)
            prev_delz = curr_delz
    else:
        g_yz = (Z[yP2// gap][xP2// gap] - Z[yP1// gap][xP1// gap])/ (yP2 - yP1)
        pts_x = np.repeat(xP1, (yP2 - yP1)// gap + 1)
        pts_y = np.arange(yP1, yP2 + 1, gap)
        z = Z[yP1// gap][xP1// gap]
        prev_delz = 0
        for y in np.arange(yP1, yP2 + 1, gap):
            z += g_yz * gap
            zg = Z[y// gap][xP1// gap]
            curr_delz = z - zg
            if (prev_delz * curr_delz >= 0):
                direct_earthwork_volume += 0.5 * abs(prev_delz + curr_delz) * gap
            else:
                direct_earthwork_volume += 0.5 * (prev_delz ** 2 + curr_delz ** 2)/ (abs(prev_delz) + abs(curr_delz)) * gap

            pts_z.append(zg)
            prev_delz = curr_delz
            
    
    cost1 = rw * direct_earthwork_volume/ (gap**3) * ec + rw * segment_length((xP[0], yP[0], zP[0]), (xP[1], yP[1], zP[1]))/ (gap**2) * pc
    fig.add_trace(
        go.Scatter3d(
            x = xP, 
            y = yP, 
            z = zP, 
            name = 'Direct Line; cost:{}'.format(round(cost1, 6))
        )
    )
    fig.add_trace(go.Scatter3d(
        x = pts_x, 
        y = pts_y, 
        z = pts_z, 
        mode = 'lines', 
        name = 'On-terrain Direct Path'
    ))
    
    return fig, pts_x, pts_y, pts_z, Z, nx, ny


# In[6]:


def initial_profiles(pts_x, pts_y, pts_z):
    long_length = [0] # longitudinal length of the vertical projection of highway
    l = 0
    max_grade = 0
    for i in range(len(pts_x) - 1):
        del_l = np.sqrt((pts_x[i + 1] - pts_x[i]) ** 2 + (pts_y[i + 1] - pts_y[i]) ** 2)
        l += del_l
        long_length.append(l)
        
    profile = go.Figure()
    profile.add_trace(go.Scatter(x = long_length, y = pts_z))
    v_proj = go.Figure()
    v_proj.add_trace(go.Scatter(x = pts_x, y = pts_y))

    return profile, v_proj, l


# In[7]:


def scale_values(r, a):
    return (r * a[0], r * a[1])

def get_intermediate_node(r, P2, P1):
    # in order to have valid indices for Z[][], y and x must be integral
    y = int(r * P2[0] + (1 - r) * P1[0]) 
    x = int(r * P2[1] + (1 - r) * P1[1])
    z = (r * P2[2] + (1 - r) * P1[2])
    return (y, x, z)

def get_profile_details(x, y, z, zg):
    long_length = [0] # longitudinal length of the vertical projection of highway
    l = 0
    max_grade = 0
    max_depth = 0
    for i in range(len(x) - 1):
        del_l = np.sqrt((x[i + 1] - x[i]) ** 2 + (y[i + 1] - y[i]) ** 2)
        l += del_l
        long_length.append(l)
        grade = abs(z[i + 1] - z[i])/ del_l
        if grade > max_grade:
            max_grade = grade
        depth = abs(z[i] - zg[i])
        if depth > max_depth:
            max_depth = depth
    return long_length, max_grade, max_depth


# In[8]:


def optimal_path(start_node, goal_node, grid_xlim, grid_ylim, pc, ec, w, Z, grade_lim, R, d_max, blocked_nodes):
    unvisited_nodes = []
    min_cost = {}
    node_direction = {} # stores wh associated with each node
    route = []
    predecessor = {}
    visited_nodes = set()
    true_cost = {}
    
    start_goal_h_cost = (segment_length(start_node, goal_node) * pc) * w
    #start_goal_h_cost = 0
    heapq.heappush(unvisited_nodes, (start_goal_h_cost, start_node)) # (cost, node)
    min_cost[start_node] = unvisited_nodes[0][0]
    true_cost[start_node] = 0
    node_direction[start_node] = (0, 0) # (wh, wv)
    horz_axes = {0: (1, 0), 2: (1, 1), 4: (0, 1), 6: (-1, 1), 8: (-1, 0), 10: (-1, -1), 12: (0, -1), 14: (1, -1), 
                1: (2, 1), 3: (1, 2), 5: (-1, 2), 7: (-2, 1), 9: (-2, -1), 11: (-1, -2), 13: (1, -2), 15: (2, -1)}

    while unvisited_nodes:
        min_node_details = heapq.heappop(unvisited_nodes)
        min_node = min_node_details[1]
        if (min_node[0], min_node[1]) == (goal_node[0], goal_node[1]): # checks if the vertical projection of the nodes coincide
        # z values may not be equal, hence ignored; otherwise an infinite loop will occur
            node = min_node
            while node != start_node:
                route.append(node)
                node = predecessor[node]
            route.append(node) # node = start_node
            route = route[:: -1]
            route.remove(min_node)
            route.append(goal_node)
#             print('check')
            return route, true_cost[min_node]
        
        wh = node_direction[min_node][0]
        wv = node_direction[min_node][1]
        if min_node == start_node:
            wh_ = [wh + i for i in horz_axes] # wh of start = 0, hence need not use '%'
        else:
            wh_ = [(wh + i) % len(horz_axes) for i in [-1, 0, 1]]
        for horz_id in wh_:
            # r: factor by which child coordinates change
            # if id is even, r = 1, 3, .., (Ne*2 - 1) => allowed_r = list(range(1, Ne * 2, 2))
            if horz_id % 2 == 0: # if id = even, r = 1, 2, .., R
                allowed_r = list(range(1, R + 1)) 
            else: # if id is odd, r = 1, 2, .., No
                allowed_r = list(range(1, int(R/ 2) + 1))
                
            for r in allowed_r:
                horz_segment_length = segment_length((0, 0, 0), scale_values(r, horz_axes[horz_id]) + (0,))
                delH_upper = horz_segment_length * grade_lim
                delH_lower = -delH_upper
                wv_ = list({min(wv + 1, 1), wv, max(wv - 1, -1)}) # set removes duplicate elements
                if 1 in wv_ and -1 in wv_:
                    del_z = np.linspace(delH_lower, delH_upper, 5) 
                elif 1 in wv_:
                    del_z = np.linspace(0, delH_upper, 3)
                else:
                    del_z = np.linspace(delH_lower, 0, 3)
                for dz in del_z: 
                    child_node = (min_node[0] + r * horz_axes[horz_id][0], min_node[1] + r * horz_axes[horz_id][1], min_node[2] + dz)
                    if child_node[0] < 0 or child_node[0] > grid_ylim or child_node[1] < 0 or child_node[1] > grid_xlim or child_node[: 2] in visited_nodes or child_node[: 2] in blocked_nodes:
                        break
                    pavement_area = np.sqrt(horz_segment_length ** 2 + dz ** 2) # per unit width
                    
                    # computing earthwork volume
                    # using section formula, get intermediate coordinates
                    # if (x1, y1, z1) and (x2, y2, z2) are divided by (x, y, z) in the ratio m:n,
                    # P = (m * P2 + n * P1)/ (m + n) where P = x, y, z  
                    # in our case, m + n = r and m = 1, 2,..., r - 1 and child_node = P2, min_node = P1
                    earthwork_volume = 0 # per unit width
                    n1 = min_node
                    for m in range(1, r):
                        intermediate_node = get_intermediate_node(m/ r, child_node, min_node)
                        n2 = intermediate_node
                        z1 = n1[2] - Z[n1[0]][n1[1]]
                        z2 = n2[2] - Z[n2[0]][n2[1]]
                        
                        if z1 * z2 >= 0:
                            earthwork_volume += 0.5 * abs(z1 + z2) * segment_length(n1[: 2] + (0,), n2[: 2] + (0,))
                        else:
                            earthwork_volume += 0.5 * (z1 ** 2 + z2 ** 2)/ (abs(z1) + abs(z2)) * segment_length(n1[: 2] + (0,), n2[: 2] + (0,))
                        n1 = n2
                    n2 = child_node
                    z1 = n1[2] - Z[n1[0]][n1[1]]
                    z2 = n2[2] - Z[n2[0]][n2[1]]
                    if abs(z2) > d_max: # cut/ fill depth should not exceed max limit
                        continue
                    if z1 * z2 >= 0:
                        earthwork_volume += 0.5 * abs(z1 + z2) * segment_length(n1[: 2] + (0,), n2[: 2] + (0,))
                    else:
                        earthwork_volume += 0.5 * (z1 ** 2 + z2 ** 2)/ (abs(z1) + abs(z2)) * segment_length(n1[: 2] + (0,), n2[: 2] + (0,))
                    
                    h_cost = (segment_length(child_node, goal_node) * pc) * w
                    #h_cost = 0
                    tentative_cost = true_cost[min_node] + w * (pavement_area * pc + earthwork_volume * ec) + h_cost
                    current_cost = min_cost.get(child_node, np.inf)
                    
                    if tentative_cost < current_cost: # relaxation: if d[u] + c(u, v) < d[v]: d[v] = d[u] + c(u, v)
                        min_cost[child_node] = tentative_cost
                        true_cost[child_node] = tentative_cost - h_cost
                        if dz == 0:
                            node_direction[child_node] = (horz_id, 0)
                        elif dz > 0:
                            node_direction[child_node] = (horz_id, 1)
                        else:
                            node_direction[child_node] = (horz_id, -1)
                        predecessor[child_node] = min_node
                        heapq.heappush(unvisited_nodes, (tentative_cost, child_node))
            
            visited_nodes.add(min_node[: 2])


# In[9]:


# Create Dash app
app = dash.Dash(__name__)

# Define the layout
def generate_gap_component():
    gap_options = [6, 8, 10, 12, 15, 20, 25, 30]
    gap_dict = [{'label': str(i), 'value': i} for i in gap_options]
    return html.Div([
        html.Div(
            'Cell Size:', 
            style={'font-size': '18px', 'display': 'inline-block', 'margin-right': '10px'}
        ),
        dcc.Dropdown(
            id='gap-dropdown',
            options=gap_dict,
            value=None,
            style={'width': '80px'}
        )], 
        style={'display': 'flex', 'flex-direction': 'row', 'align-items': 'center'}
    )
def generate_amplify_component():
    return html.Div([
        html.Div(
            'Amplify z-values:', 
            style={'font-size': '18px', 'display': 'inline-block'}
        ),
        html.Div([
            dcc.Slider(
                id='amplify-slider',
                min=30,
                max=80,
                step=10
            )], 
            style={'width': '200px'}
        )],
        style={'width': '250px', 'display': 'flex', 'flex-direction': 'row', 'align-items': 'center'}
    )
def generate_primary_octave_component():
    return html.Div([
        html.Div(
            'Primary:', 
            style={'display': 'inline-block'}
        ),
        html.Div([
            dcc.Slider(
                id='prim-oct-slider',
                min=2,
                max=5,
                step=1
            )], 
            style={'width': '200px'}
        )], 
        style = {'width': '350px', 'display': 'flex', 'flex-direction': 'row', 'justify-content': 'space-between'}
    )
def generate_secondary_octave_component():
    return html.Div([
        html.Div(
            'Secondary:', 
            style={'display': 'inline-block'}
        ),
        html.Div([
            dcc.Slider(
                id='sec-oct-slider',
                min=6,
                max=14,
                step=1
            )], 
            style={'width': '250px'}
        )], 
        style = {'width': '350px', 'display': 'flex', 'flex-direction': 'row', 'justify-content': 'space-between'}
    )
def generate_tertiary_octave_component():
    return html.Div([
        html.Div(
            'Tertiary:', 
            style={'display': 'inline-block'}
        ),
        html.Div([
            dcc.Slider(
                id='tert-oct-slider',
                min=10,
                max=24,
                step=2
            )], 
            style={'width': '250px'}
        )], 
        style = {'width': '350px', 'display': 'flex', 'flex-direction': 'row', 'justify-content': 'space-between'}
    )
def generate_octave_component():
    return html.Div([
        html.Div(
            'No. of octaves:', 
            style={'font-size': '18px', 'display': 'inline-block', 'margin-bottom': '15px'}
        ),
        generate_primary_octave_component(), 
        generate_secondary_octave_component(), 
        generate_tertiary_octave_component()], 
        style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center'}
    )


# In[10]:


def generate_paving_rate_component():
#     arr = [i for i in range(11)]
    return html.Div([
        html.Div(
            'Paving rate:', 
            style={'display': 'inline-block', 'font-size': '18px'}
        ),
        html.Div([
            dcc.Slider(
                id='pav-slider',
                min=1,
                max=10,
                step=0.2, 
                marks=[{'label': str(i), 'value': i} for i in range(11)]
            )], 
            style={'width': '200px'}
        )], 
        style = {
#             'width': '350px', 
            'display': 'flex', 
            'flex-direction': 'row', 
            'justify-content': 'space-between'
        }
    )
def generate_earthwork_rate_component():
    return html.Div([
        html.Div(
            'Earthwork rate:', 
            style={'display': 'inline-block', 'font-size': '18px'}
        ),
        html.Div([
            dcc.Slider(
                id='ewk-slider',
                min=1,
                max=10,
                step=0.2, 
                marks=[{'label': str(i), 'value': i} for i in range(11)]
            )], 
            style={'width': '200px'}
        )], 
        style = {
#             'width': '350px', 
            'display': 'flex', 
            'flex-direction': 'row', 
            'justify-content': 'space-between'
        }
    )

def generate_rate_component():
    return html.Div([
        generate_paving_rate_component(), 
        generate_earthwork_rate_component()], 
        style={
            'display': 'flex', 
            'flex-direction': 'column', 
            'align-items': 'stretch',
        }
    )
def generate_grade_limit_component():
    return html.Div([
        html.Div(
            'Grade limit:', 
            style={'font-size': '18px', 'display': 'inline-block'}
        ),
        html.Div([
            dcc.Slider(
                id='gradelim-slider',
                min=3,
                max=9,
                step=1
            )], 
            style={'width': '180px'}
        )],
        style={
            'display': 'flex', 
            'flex-direction': 'row', 
            'align-items': 'center', 
            'justify-content': 'space-between'
        }
    )
def generate_max_depth_component():
    return html.Div([
        html.Div(
            'Max cut/fill depth:', 
            style={'font-size': '18px', 'display': 'inline-block'}
        ),
        html.Div([
            dcc.Slider(
                id='maxdepth-slider',
                min=3,
                max=9,
                step=1
            )], 
            style={'width': '200px'}
        )],
        style={
            'display': 'flex', 
            'flex-direction': 'row', 
            'align-items': 'center',
            'justify-content': 'space-between'
        }
    )
def generate_num_paths_component():
    return html.Div([
        html.Div(
            'No. of paths:', 
            style={'font-size': '18px', 'display': 'inline-block'}
        ),
        html.Div([
            dcc.Slider(
                id='numpaths-slider',
                min=1,
                max=4,
                step=1
            )], 
            style={'width': '200px'}
        )],
        style={
            'display': 'flex', 
            'flex-direction': 'row', 
            'justify-content': 'space-between',
        }
    )


# In[11]:


app.layout = html.Div([
    html.Div([
        html.Div([
            html.Div([
                generate_gap_component(),
                generate_amplify_component(),
                generate_octave_component()],
                style={
                    'display': 'flex', 
                    'flex-direction': 'row', 
                    'justify-content': 'space-around', 
                    'align-items': 'center'
                }
            ),
            html.Div([
                html.Div([
                    generate_rate_component(), 
                    generate_grade_limit_component()], 
                    style={
                        'display': 'flex', 
                        'flex-direction': 'column', 
                        'align-items': 'stretch', 
                        'justify-content': 'space-around'
                    }
                ), 
                html.Div([
                    generate_max_depth_component(), 
                    generate_num_paths_component()], 
                    style={
                        'display': 'flex', 
                        'flex-direction': 'column', 
                        'align-items': 'stretch', 
                        'justify-content': 'space-around'
                    }
                )],
                style={
                    'display': 'flex', 
                    'flex-direction': 'row', 
                    'justify-content': 'space-around', 
                    'align-items': 'center'
                }
            )],
            style={
                'display': 'flex', 
                'flex-direction': 'column', 
                'align-items': 'stretch'
            }
        ),
        html.Button('Generate Terrain & Get Paths', id='btn', style={'height': '70px'}, n_clicks=0, disabled=True)],
        style={
            'display': 'flex', 
            'flex-direction': 'row', 
            'justify-content': 'space-around', 
            'align-items': 'center'
        }
    ), 
    html.Div(id='graph-container')],
)


# In[12]:


# Define the callback function to handle dropdown selections and enable/disable the button
@app.callback(
    Output('btn', 'disabled'),
    [
        Input('gap-dropdown', 'value'), 
        Input('amplify-slider', 'value'),
        Input('prim-oct-slider', 'value'),
        Input('sec-oct-slider', 'value'),
        Input('tert-oct-slider', 'value'), 
        Input('pav-slider', 'value'), 
        Input('ewk-slider', 'value'),
        Input('gradelim-slider', 'value'),
        Input('maxdepth-slider', 'value'),
        Input('numpaths-slider', 'value')
    ]
)
def disable_button(*args):
    if all(value is not None for value in args):
        return False
    else:
        return True


# In[13]:


# Define the callback function
@app.callback(
    Output('graph-container', 'children'), 
    [
        Input('btn', 'n_clicks')
    ], 
    [
        State('gap-dropdown',    'value'), 
        State('amplify-slider', 'value'), 
        State('prim-oct-slider', 'value'),
        State('sec-oct-slider', 'value'),
        State('tert-oct-slider', 'value'), 
        State('pav-slider', 'value'), 
        State('ewk-slider', 'value'),
        State('gradelim-slider', 'value'),
        State('maxdepth-slider', 'value'),
        State('numpaths-slider', 'value')
    ]
)
def update_graph(
    n_clicks, gap, factor_z, 
    num_prim_octaves, num_sec_octaves, num_tert_octaves, 
    pc, ec, glim, maxd, npaths
):
    # INPUT VARIABLES
    road_width = 7 # in metres
    R = 3
    if n_clicks > 0:
        # Generate data based on selected options
        # Update the figure with new data
        fig1, pts_x, pts_y, pts_z, Z, nx, ny = generate_terrain(gap, factor_z, num_prim_octaves, num_sec_octaves, num_tert_octaves, road_width, pc, ec)
        fig2, fig3, onterrain_direct_path_length = initial_profiles(pts_x, pts_y, pts_z)
        
        # the complete 3-d model is scaled down by a factor 'gap'
        start = (int(pts_y[0]/ gap), int(pts_x[0]/ gap), Z[int(pts_y[0]/ gap)][int(pts_x[0]/ gap)]/ gap)
        goal = (int(pts_y[-1]/ gap), int(pts_x[-1]/ gap), Z[int(pts_y[0]/ gap)][int(pts_x[-1]/ gap)]/ gap)

        blocked_nodes = []
        glim = glim * 0.01
        for _ in range(npaths):
            route, cost = optimal_path(start, goal, nx-1, ny-1, pc, ec, road_width/ gap, Z/ gap, glim, R, maxd/ gap, blocked_nodes) 
            for i in range(1, len(route) - 1):
                blocked_nodes.append(route[i][: 2])

            xvals = []
            yvals = []
            zvals = []
            zvals_ground = []
            for i in range(len(route)):
                # points in route need to be rescaled
                xvals.append(gap * route[i][1])
                yvals.append(gap * route[i][0])
                zvals.append(gap * route[i][2])
                zvals_ground.append(Z[route[i][0]][route[i][1]])

            parameters_text = "pc:{}, ec:{}, cost:{}".format(pc, ec, round(cost, 6))
            fig1.add_trace(
                go.Scatter3d(
                    x = xvals, 
                    y = yvals, 
                    z = zvals, 
                    name = 'road: ' + parameters_text, 
                    mode = 'lines', 
                )
            )
        #     fig.add_trace(go.Scatter3d(x = xvals, y = yvals, z = zvals_ground, name = 'ground trace', mode = 'lines'))
            long_length, max_grade, max_depth = get_profile_details(xvals, yvals, zvals, zvals_ground)
            fig2.add_trace(go.Scatter(x = long_length, y = zvals))
        #     profile.add_trace(go.Scatter(x = long_length, y = zvals_ground))
            fig3.add_traces(go.Scatter(x = xvals, y = yvals))
            
        return [
            dcc.Graph(figure=fig1), 
            dcc.Graph(figure=fig2), 
            dcc.Graph(figure=fig3)
        ]
    else:
        # Return an empty figure if the button hasn't been clicked yet
        return [
            dcc.Graph(figure=go.Figure())
        ]

# Run the app
if __name__ == '__main__':
    app.run_server(mode='inline')


# In[15]:


# for the scaled-down model:

# direct (i.e. through terrain)path
# print(direct_earthwork_volume/ (gap**3) * ec + segment_length(start, goal) * pc)
# on-terrain path
# print(onterrain_direct_path_length/ gap * pc)


# In[ ]:




