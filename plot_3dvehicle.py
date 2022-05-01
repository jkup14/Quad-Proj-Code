import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go


def plot3dvehicle(x0,xf, T, X, U, n, n_agents):
    ax = plt.axes(projection='3d')
    for iii in range(n_agents):
        x_idx, y_idx, z_idx = iii*n+9, iii*n+10, iii*n+11
        ax.plot3D(X[x_idx, :], X[y_idx, :], X[z_idx, :], color=colors[iii])
        ax.scatter(x0[x_idx], x0[y_idx], x0[z_idx], color=colors[iii], s=100)
        ax.scatter(xf[x_idx], xf[y_idx], xf[z_idx], color=colors[iii], marker='X', s=100)


# ax.plot3D(X[9, :], X[10, :], X[11, :], 'gray')
    # draw sphere
    # u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    # x = r*np.cos(u)*np.sin(v) + o1
    # y = r*np.sin(u)*np.sin(v) + o2
    # z = r*np.cos(v) + o3
    # ax.plot_wireframe(x, y, z, color="r")
    # ax.scatter(xf[9], xf[10], xf[11], color="g", s=100)
    # ax.scatter(x0[9], x0[10], x0[11], color="r", s=100)
    # ax.scatter(xf[21], xf[22], xf[23], color="g", s=100)
    # ax.scatter(x0[21], x0[22], x0[23], color="r", s=100)

    ax.set_xlabel('$x$', size=12, math_fontfamily='cm', fontname='Times New Roman')
    ax.set_ylabel('$y$', size=12, math_fontfamily='cm', fontname='Times New Roman')
    ax.set_zlabel('$z$', size=12, math_fontfamily='cm', fontname='Times New Roman')

    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_zlim(-6, 6)
# plt.figure(2)
# plt.plot(t, X[:, 9])
# plt.figure(3)
# plt.plot(t, X[:, 10])
# plt.figure(4)
# plt.plot(t, X[:, 11])
# plt.xlabel('Time (s)')
# plt.ylabel('States')
# plt.title("States vs Time", fontsize='large')

# plt.figure(5)
# plt.plot(t, X[:, -1:])
# plt.xlabel('Time (s)')
# plt.ylabel('Barrier State (z)')
# plt.title("BaS vs Time", fontsize='large')


def animate3dVehicle_Multi(n_agents, n1, delta, x0, xf, T, X, U):
    fig2, ax2 = plt.subplots(1)
    ax2.plot(T[0:-1], U.T)
    # collided = False
    # time = 0
    # collided_agents = None
    # for ii in range(n_agents):
    #     for jj in range(ii + 1, n_agents):
    #         dists_ij = np.sqrt(np.sum((X[ii * n1:ii * n1 + 2, :] - X[jj * n1:jj * n1 + 2, :]) ** 2, 0))
    #         # ax[1].plot(T, dists_ij)
    #         if (np.array(dists_ij) < delta).any():
    #             collided = True
    #             time = T[np.where((np.array(dists_ij) < delta))]
    #             collided_agents = (ii, jj)
    #
    #
    # if collided:
    #     ii, jj = collided_agents
    #     dists_ij = np.sqrt(np.sum((X[ii * n1:ii * n1 + 2, :] - X[jj * n1:jj * n1 + 2, :]) ** 2, 0))
    #     print("Agents", collided_agents, "Collided at time = ", time, "with distance", dists_ij[np.where((np.array(dists_ij) < delta))])
    # else:
    #     print("Did not collide")

    drawing_dt = 0.1
    dt = T[1]-T[0]
    step = int(drawing_dt/dt)
    indices = np.array(list(range(len(T))[::step]) + [len(T)-1])

    data = dict()
    data['t'] = np.array([])
    data['x'] = np.array([])
    data['y'] = np.array([])
    data['z'] = np.array([])
    data['type'] = []
    data['size'] = []
    category_orders = []
    for n in range(n_agents):
        data['t'] = np.hstack([data['t'], T[indices]])
        data['x'] = np.hstack([data['x'], X[n*n1+9, indices]])
        data['y'] = np.hstack([data['y'], X[n*n1+10, indices]])
        data['z'] = np.hstack([data['z'], X[n*n1+11, indices]])
        data['type'] = data['type'] + ['Agent '+str(n)] * len(indices)
        category_orders = category_orders + ['Agent '+str(n)]

    minx, maxx = np.min(data['x']), np.max(data['x'])
    miny, maxy = np.min(data['y']), np.max(data['y'])
    minz, maxz = np.min(data['z']), np.max(data['z'])
    xrange = 2+maxx-minx
    yrange = 2+maxy-miny
    zrange = 2+maxz-minz
    xfactor = 860 # needs to be tuned for your plot
    yfactor = 340 # needs to be tuned for your plot
    zfactor = 640 # needs to be tuned for your plot
    plotter_aspect_ratio = 2 # width/height

    for n in range(n_agents): # To make marker size accurate
        if xrange/yrange >= plotter_aspect_ratio:
            data['size'] = data['size'] + [delta*1/(2+maxx-minx)*xfactor] * len(indices)
        else:
            data['size'] = data['size'] + [delta*1/(2+maxy-miny)*yfactor] * len(indices)

    fig = px.scatter_3d(
        data,
        x='x',
        y='y',
        z='z',
        animation_frame='t',
        animation_group='type',
        color='type',
        category_orders={'type': category_orders},
        color_discrete_sequence=colorsHex[0:n_agents],
        size='size',
        size_max=data['size'][-1],
        hover_name='type',
        template='plotly_white',         # different plotly templates and backgrounds available
        range_x=(minx-1, maxx+1),
        range_y=(miny-1, maxy+1),
        range_z=(minz-1, maxz+1),
        # range_x=(minx, maxx),
        # range_y=(miny, maxy),
        height=700,
    )
    # Make equal one meter grid
    fig.update_xaxes(
        dtick=1.0,
        showline=False
    )
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
        showline=False,
        dtick=1.0
    )

    for n in range(n_agents):
        subject_line = px.line_3d(
            x=X[n*n1+9, :],
            y=X[n*n1+10, :],
            z=X[n*n1+11, :]
        ).data[0]
        subject_line.line['color'] = colorsHex[n]
        subject_line.line['width'] = 1
        fig.add_trace(subject_line)
        xf_i = go.Scatter3d(x=xf[n*n1+9], y=xf[n*n1+10], z=xf[n*n1+11], marker_color=colorsHex[n], marker_symbol='x', marker_size=5)
        fig.add_trace(xf_i)

    fig.show()
    return fig

def animate3dVehicle_Multi_track(n_agents, n1, delta, x0, xf, x_des, T, X, U, obs_data):
    fig2, ax2 = plt.subplots(1)
    ax2.plot(T[0:-1], U.T)
    ax2.set_title('controls')
    # collided = False
    # time = 0
    # collided_agents = None
    # for ii in range(n_agents):
    #     for jj in range(ii + 1, n_agents):
    #         dists_ij = np.sqrt(np.sum((X[ii * n1:ii * n1 + 2, :] - X[jj * n1:jj * n1 + 2, :]) ** 2, 0))
    #         # ax[1].plot(T, dists_ij)
    #         if (np.array(dists_ij) < delta).any():
    #             collided = True
    #             time = T[np.where((np.array(dists_ij) < delta))]
    #             collided_agents = (ii, jj)
    #
    #
    # if collided:
    #     ii, jj = collided_agents
    #     dists_ij = np.sqrt(np.sum((X[ii * n1:ii * n1 + 2, :] - X[jj * n1:jj * n1 + 2, :]) ** 2, 0))
    #     print("Agents", collided_agents, "Collided at time = ", time, "with distance", dists_ij[np.where((np.array(dists_ij) < delta))])
    # else:
    #     print("Did not collide")

    drawing_dt = 0.02
    dt = T[1]-T[0]
    step = int(drawing_dt/dt)
    indices = np.array(list(range(len(T))[::step]) + [len(T)-1])

    data = dict()
    data['t'] = np.array([])
    data['x'] = np.array([])
    data['y'] = np.array([])
    data['z'] = np.array([])
    data['type'] = []
    data['size'] = []
    category_orders = []
    for n in range(n_agents):
        data['t'] = np.hstack([data['t'], T[indices]])
        data['x'] = np.hstack([data['x'], X[n*n1+9, indices]])
        data['y'] = np.hstack([data['y'], X[n*n1+10, indices]])
        data['z'] = np.hstack([data['z'], X[n*n1+11, indices]])
        data['type'] = data['type'] + ['Agent '+str(n)] * len(indices)
        category_orders = category_orders + ['Agent '+str(n)]

    minx, maxx = np.min(data['x']), np.max(data['x'])
    miny, maxy = np.min(data['y']), np.max(data['y'])
    minz, maxz = np.min(data['z']), np.max(data['z'])
    xrange = 2+maxx-minx
    yrange = 2+maxy-miny
    zrange = 2+maxz-minz
    xfactor = 860 # needs to be tuned for your plot
    yfactor = 340 # needs to be tuned for your plot
    zfactor = 640 # needs to be tuned for your plot
    plotter_aspect_ratio = 2 # width/height
    data['size'] = [20]*len(indices)
    # for n in range(n_agents): # To make marker size accurate
    #     if xrange/yrange >= plotter_aspect_ratio:
    #         data['size'] = data['size'] + [delta*1/(2+maxx-minx)*xfactor] * len(indices)
    #     else:
    #         data['size'] = data['size'] + [delta*1/(2+maxy-miny)*yfactor] * len(indices)

    fig = px.scatter_3d(
        data,
        x='x',
        y='y',
        z='z',
        animation_frame='t',
        animation_group='type',
        color='type',
        category_orders={'type': category_orders},
        color_discrete_sequence=colorsHex[0:n_agents],
        size='size',
        size_max=data['size'][-1],
        hover_name='type',
        template='plotly_white',         # different plotly templates and backgrounds available
        # range_x=(minx-1, maxx+1),
        # range_y=(miny-1, maxy+1),
        # range_z=(minz-1, maxz+1),
        # range_x=(minx, maxx),
        # range_y=(miny, maxy),
        height=700,
    )
    # Make equal one meter grid
    fig.update_xaxes(
        # dtick=1.0,
        showline=False
    )
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
        showline=False,
        # dtick=1.0
    )
    # to remove grids, ticks and labels:
    # fig.update_layout(scene = dict(xaxis = dict(showgrid = False, visible = False, showticklabels = False),
    #                 yaxis = dict(showgrid = False, visible = False, showticklabels = False),
    #                 zaxis = dict(showgrid = False, visible = False, showticklabels = False)))


    invisible_scale = go.Scatter3d(
        name="",
        visible=True,
        showlegend=False,
        opacity=0,
        hoverinfo='none',
        x=[0, -4, 4],
        y=[0, -4, 4],
        z=[0, -4, 4]
    )
    fig.add_trace(invisible_scale)

    for n in range(n_agents):
        subject_line = px.line_3d(
            x=X[n*n1+9, :],
            y=X[n*n1+10, :],
            z=X[n*n1+11, :]
        ).data[0]
        subject_line.line['color'] = "green"
        subject_line.line['width'] = 5
        subject_line.line['dash'] = "dashdot"
        # fig.add_trace(subject_line)

        tracking_line = px.line_3d(
            x=x_des[n * n1 + 9, :],
            y=x_des[n * n1 + 10, :],
            z=x_des[n * n1 + 11, :]
        ).data[0]
        tracking_line.line['color'] = "blue"
        tracking_line.line['width'] = 2
        tracking_line.line['dash'] = "dot"
        fig.add_trace(tracking_line)
        xf_i = go.Scatter3d(x=X[n*n1+9, [-1]], y=X[n*n1+10, [-1]], z=X[n*n1+11, [-1]], marker_color=colorsHex[n], marker_symbol='x', marker_size=5)
        xi_i = go.Scatter3d(x=x0[n*n1+9], y=x0[n*n1+10], z=x0[n*n1+11], marker_color=colorsHex[n], marker_symbol='circle', marker_size=5)
        # xf_i = go.Scatter3d(x=xf[n*n1+9], y=xf[n*n1+10], z=xf[n*n1+11], marker_color=colorsHex[n], marker_symbol='X', marker_size=5)
        fig.add_trace(xf_i)
        fig.add_trace(xi_i)

    obs_plot_data = []
    for n in range(obs_data.shape[0]):
        x,y,z = ms(obs_data[n, 0], obs_data[n, 1], obs_data[n, 2], obs_data[n, 3])
        surf = go.Surface(x=x, y=y, z=z, opacity=0.5, colorscale=[[0,"darkred"], [1,"darkred"]])
        fig.add_trace(surf)

    frames = [go.Frame(data= [go.Scatter3d(
                                            x=X[9, :k+1],
                                            y=X[10, :k+1],
                                            z=X[11, :k+1]
                                          )
                             ],
                        traces= [0],
                        name=f'frame{k}'
                        )for k in range(X.shape[1])
             ]

    fig.update(frames=frames)

    fig.show()
    return fig

def ms(x, y, z, radius, resolution=20):
    """Return the coordinates for plotting a sphere centered at (x,y,z)"""
    u, v = np.mgrid[0:2*np.pi:resolution*2j, 0:np.pi:resolution*1j]
    X = radius * np.cos(u)*np.sin(v) + x
    Y = radius * np.sin(u)*np.sin(v) + y
    Z = radius * np.cos(v) + z
    return (X, Y, Z)

colors = ['dimgrey','rosybrown','maroon','sienna','darkorange','gold','olive','darkseagreen','lime','darkslategrey',
          'aqua','dodgerblue','slateblue','blueviolet','fuchsia','cyan','lavender','orchid','indianred','yellow']
colorsHex = [matplotlib.colors.cnames[colors[i]] for i in range(len(colors))]
