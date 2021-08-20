#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import numpy as np
from perlin_noise import PerlinNoise
import plotly.graph_objects as go

x_max = 500 # in metres
y_max = 800 # in metres
gap = 20 # should be an integer
nx = int(x_max/ gap) + 1 # number of points in x-direction
ny = int(y_max/ gap) + 1 # number of points in y-direction
x = np.linspace(0, x_max, nx) # array of x-coordinates along x-direction
y = np.linspace(0, y_max, ny) # array of y-coordinates along y-direction
# assuming x-direction is horizontal while y-direction is vertical
X, Y = np.meshgrid(x, y) # generate coordinate matrix

# 3d-land generation using perlin noise
Z = np.zeros((ny, nx)) # "ny by nx" zero matrix that stores altitude values at different coordinates
noise1 = PerlinNoise(octaves = 2) # more octaves implies more undulations
noise2 = PerlinNoise(octaves = 8)
noise3 = PerlinNoise(octaves = 80)
for j in range(ny):
    for i in range(nx):
        Z[j][i] =         noise1([j/ ny, i/ nx])
        Z[j][i] += 0.3 *  noise2([j/ ny, i/ nx])
        Z[j][i] += 0.05 * noise3([j/ ny, i/ nx])
Z = Z * 65

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
x_avl = list(x) # x available
y_avl = list(y) # y available
xP = random.sample(x_avl, 1)
yP = random.sample(y_avl, 1)
zP = [Z[int(yP[0]/ gap)][int(xP[0]/ gap)]]

f = 0.15
x_low = max(xP[0] - int(f * x_max/ gap) * gap, 0)
x_high = min(xP[0] + int(f * x_max/ gap) * gap, x_max)
for el in np.linspace(x_low, x_high, int((x_high - x_low)/ gap) + 1):
    x_avl.remove(el)
y_low = max(yP[0] - int(f * y_max/ gap) * gap, 0)
y_high = min(yP[0] + int(f * y_max/ gap) * gap, y_max)
for el in np.linspace(y_low, y_high, int((y_high - y_low)/ gap) + 1):
    y_avl.remove(el)
xP.append(random.sample(x_avl, 1)[0])
yP.append(random.sample(y_avl, 1)[0])
zP.append(Z[int(yP[1]/ gap)][int(xP[1]/ gap)])

# plot a straight line connecting the two points
fig.add_trace(go.Scatter3d(x = xP, y = yP, z = zP))


# In[2]:


def euclidean_dist(x, y, x0, y0):
    return np.sqrt((x - x0) ** 2 + (y - y0) ** 2)

def a_less_than_b(a, b):
    if a <= b:
        return 1
    else:
        return -1
    
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
pts_x.append(xP1)
pts_y.append(yP1)
pts_z.append(Z[int(yP1/ gap)][int(xP1/ gap)])

if xP1 != xP2 and yP1 != yP2:
    interm_pts = [] # stores data related to the intermediate points that will form the desired trace
    m = (yP2 - yP1)/ (xP2 - xP1) # slope of the projection of the straight line in 2-d plane
    for xi in range(xP1 + gap, xP2, gap):
        yi = m * (xi - xP1) + yP1
        d = euclidean_dist(xi, yi, xP1, yP1)
        interm_pts.append([xi, yi, d, xi, int(yi/ gap) * gap, xi, int(yi/ gap + 1) * gap])
        
    yP1_smaller_than_yP2 = a_less_than_b(yP1, yP2) # if yP1 <= yP2, value = 1; else -1
    for yi in range(yP1 + yP1_smaller_than_yP2 * gap, yP2, yP1_smaller_than_yP2 * gap):
        xi = 1/ m * (yi - yP1) + xP1
        d = euclidean_dist(xi, yi, xP1, yP1)
        interm_pts.append([xi, yi, d, int(xi/ gap) * gap, yi, int(xi/ gap + 1) * gap, yi])
    interm_pts.sort(key = lambda i: i[2]) # sorts the data based on the distance of intermediate point from the start point (point 1)
    
    for i in range(len(interm_pts)):
        x1 = interm_pts[i][0]
        y1 = interm_pts[i][1]
        x2 = interm_pts[i][3]
        y2 = interm_pts[i][4]
        x3 = interm_pts[i][5]
        y3 = interm_pts[i][6]
        if euclidean_dist(x1, y1, x2, y2) < euclidean_dist(x1, y1, x3, y3):
            pts_x.append(x2)
            pts_y.append(y2)
            pts_z.append(Z[int(y2/ gap)][int(x2/ gap)])
        else:
            pts_x.append(x3)
            pts_y.append(y3)
            pts_z.append(Z[int(y3/ gap)][int(x3/ gap)])
            
elif xP1 != xP2: # if yP1 = yP2
    for xi in range(xP1 + gap, xP2, gap):
        pts_x.append(xi)
        pts_y.append(yP1)
        pts_z.append(Z[int(yP1/ gap)][int(xi/ gap)])
        
elif yP1 != yP2: # assuming start and destination points are not one above the other 
    yP1_smaller_than_yP2 = a_smaller_than_b(yP1, yP2) # if yP1 is smaller than yP2, value = 1; else -1
    for yi in range(yP1 + yP1_smaller_than_yP2 * gap, yP2, yP1_smaller_than_yP2 * gap):
        pts_x.append(xP1)
        pts_y.append(yi)
        pts_z.append(Z[int(yi/ gap)][int(xP1/ gap)])
    
pts_x.append(xP2)
pts_y.append(yP2)
pts_z.append(Z[int(yP2/ gap)][int(xP2/ gap)])

fig.add_trace(go.Scatter3d(
    x = pts_x, 
    y = pts_y, 
    z = pts_z, 
    mode = 'lines'
))
fig.show()


long_length = [0] # longitudinal length of the vertical projection of highway
l = 0
max_grade = 0
for i in range(len(pts_x) - 1):
    del_l = np.sqrt((pts_x[i + 1] - pts_x[i]) ** 2 + (pts_y[i + 1] - pts_y[i]) ** 2)
    l += del_l
    long_length.append(l)
    
    grade = abs(pts_z[i + 1] - pts_z[i])/ del_l
    if grade > max_grade:
        max_grade = grade

profile = go.Figure()
profile.add_trace(go.Scatter(x = long_length, y = pts_z))
profile.show()

v_proj = go.Figure()
v_proj.add_trace(go.Scatter(x = pts_x, y = pts_y))
v_proj.show()


# In[12]:


import heapq

def segment_length(a, b):
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2)

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
            print('check')
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


# In[13]:


# the complete 3-d model is scaled down by a factor 'gap'
start = (int(yP1/ gap), int(xP1/ gap), Z[int(yP1/ gap)][int(xP1/ gap)]/ gap)
goal = (int(yP2/ gap), int(xP2/ gap), Z[int(yP2/ gap)][int(xP2/ gap)]/ gap)

# INPUT VARIABLES
paving_cost = 1 # in monetary units/ metre^2
earthwork_cost = 2 # in monetary units/ metre^3
carriageway_width = 7 # in metres
grade_lim = 0.05 # slope limit
d_max = 7 # maximum depth of cut/ fill in metres
R = 3
number_of_paths = 1
blocked_nodes = []

for k in range(number_of_paths):
    route, cost = optimal_path(start, goal, nx - 1, ny - 1, paving_cost, earthwork_cost, carriageway_width/ gap, Z/ gap, grade_lim, R, d_max/ gap, blocked_nodes) 
    print(cost)
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

    parameters_text = 'pc=' + str(paving_cost) + ';ec=' + str(earthwork_cost) + ';grade_lim=' + str(grade_lim) + ';cost=' + str(round(cost, 3))
    fig.add_trace(go.Scatter3d(x = xvals, y = yvals, z = zvals, name = 'road:' + parameters_text, mode = 'lines'))
    fig.add_trace(go.Scatter3d(x = xvals, y = yvals, z = zvals_ground, name = 'ground trace', mode = 'lines'))
    
    long_length, max_grade, max_depth = get_profile_details(xvals, yvals, zvals, zvals_ground)
    profile.add_trace(go.Scatter(x = long_length, y = zvals))
    profile.add_trace(go.Scatter(x = long_length, y = zvals_ground))
   
    v_proj.add_traces(go.Scatter(x = xvals, y = yvals))
    
fig.show()
profile.show()
v_proj.show()

