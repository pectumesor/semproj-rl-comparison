"""
Script with functions that triangulate the rooms to create a navigation mesh and makes use of funnel to find the shortest path

"""
import json
import numpy as np
import triangle as tr
import networkx as nx
import matplotlib.pyplot as plt
import wandb

def sign(p, a, b):
    """
     Check to which side of the edge (a,b) that point p is.
    """
    return (p[0]-b[0])*(a[1]-b[1]) - (a[0]-b[0])*(p[1]-b[1])

def segment_cross(p, q, r, s):
    d1 = sign(r,s,p)
    d2 = sign(r,s,q)
    d3 = sign(p,q,r)
    d4 = sign(p,q,s)
    if ((d1 > 0) != (d2 >0) and (d3 > 0) != (d4 > 0)):
        return True
    return False

def point_in_triangle(p, a, b, c):
    """
     Check if p is on the same side of each edge of the triangle T = (a,b,c).
     This happens when the sign of each edge is the same
    """

    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)

def assign_triangle(p, triangles):
    """
    Find inside which delaunay triangle some point is
    """
    for i, t in enumerate(triangles):
        if point_in_triangle(p, *t):
            return i

def extract_vertices_and_segments(json_path: str):
    vertex_index = {}
    vertices = []
    segments = []

    def get_index(point):
        key = (point["x"], point["y"])
        if key not in vertex_index:
            vertex_index[key] = len(vertices)
            vertices.append([point["x"], point["y"]])
        return vertex_index[key]

    with open(json_path) as f:
        data = json.load(f)

    for edge in data["edges"]:
        u = get_index(edge["from"])
        v = get_index(edge["to"])
        segments.append([u, v])


    start_pos = [data["start_pos"]["x"], data["start_pos"]["y"]]
    end_pos = [data["goal_pos"]["x"], data["goal_pos"]["y"]]

    return dict(
        vertices=np.array(vertices, dtype=np.float64),
        segments=np.array(segments, dtype=np.int32),
    ), start_pos, end_pos


def is_wall_segment(u, v, segments):
    """Check whether (u, v) appears, in either direction, as a row of `segments`."""
    return np.any(np.all(segments == (u, v), axis=1)) or np.any(np.all(segments == (v, u), axis=1))

def delaunay_edges_from_triangle(triangle, segments):
    """Build the 3 graph edges of a triangle, tagged by vertex index (not coordinates,
    since those aren't hashable and can't be used as networkx node ids)."""
    i, j, k = triangle
    edges = []
    for u, v in ((i, j), (j, k), (i, k)):
        color = "red" if is_wall_segment(u, v, segments) else "blue"
        edges.append((u, v, {"color": color}))
    return edges

def delauney_graph(vertices, triangles, segments):

    edge_list = []
    for triangle in triangles:
        edge_list.extend(delaunay_edges_from_triangle(triangle, segments))

    G = nx.Graph(edge_list)
    nx.set_node_attributes(G, {i: tuple(pos) for i, pos in enumerate(vertices)}, "pos")
    return G

# Cost added to a triangle-graph edge that crosses a wall. Far larger than any
# wall-free route through a room, so A* only crosses a wall when a region is
# otherwise sealed off (and then crosses as few as possible).
WALL_CROSSING_PENALTY = 1e6


def triangle_adjacency_graph(vertices, triangles, triangle_neighbors, segments):
    
    centroids = vertices[triangles].mean(axis=1)
    G = nx.Graph()
    for i, c in enumerate(centroids):
        G.add_node(i, centroid=c)

    for i, neighbors in enumerate(triangle_neighbors):
        for k, n in enumerate(neighbors):
            if n == -1:
                continue
            u, v = triangles[i][(k + 1) % 3], triangles[i][(k + 2) % 3]
            w = float(np.linalg.norm(centroids[i] - centroids[n])) # Weight is centroid distance between neighboring triangles
            if is_wall_segment(u, v, segments): # If triangle share a wall segment, then crossing here has a huge penalty
                w += WALL_CROSSING_PENALTY
            G.add_edge(i, n, weight=w)

    return G

def generate_graphs(triangulation):

    delauney = delauney_graph(triangulation["vertices"],
                              triangulation["triangles"], triangulation["segments"])

    triangle_graph = triangle_adjacency_graph(triangulation["vertices"],
                                              triangulation["triangles"],
                                              triangulation["neighbors"],
                                              triangulation["segments"])

    return delauney, triangle_graph

def shortest_triangle_path(T: nx.Graph, vertices, triangles, start, end):
    """
    Find the sequence of delaunay triangles of the shortest path from start to end
    using the A* algorithm
    """
    coord_triangles = [(vertices[i], vertices[j], vertices[k]) for i, j, k in triangles]
    T_start = assign_triangle(start, coord_triangles)
    T_end = assign_triangle(end, coord_triangles)
    return nx.astar_path(T, T_start, T_end)


def triangle_centroid(vertices, triangle):
    return vertices[list(triangle)].mean(axis=0)

def shortest_centroid_path(vertices, triangles, path):
    return [triangle_centroid(vertices, triangles[i]) for i in path]

def plot_path(ax, vertices, triangles, path, start, end):
    """Draw the start -> end route through the centroids of the triangle-index path."""
    centroids = shortest_centroid_path(vertices, triangles, path)
    points = np.array([start] + centroids + [end])

    ax.plot(points[:, 0], points[:, 1], color="green", linewidth=2, marker="o", zorder=5)
    ax.plot(*start, marker="s", color="green", markersize=8, zorder=6)
    ax.plot(*end, marker="*", color="red", markersize=12, zorder=6)


def common_vertices_triangles(t1, t2, triangulation):

    common_indices = []
    for i in t1:
        if i in t2:
            common_indices.append(i)

    return triangulation["vertices"][common_indices]


def create_funnel_portals(centroid_path, triangle_path, triangulation, start, end):

    portal_left = [start]
    portal_right = [start]

    for j in range(1, len(triangle_path)):

        T_u = triangulation["triangles"][triangle_path[j-1]]
        T_v = triangulation["triangles"][triangle_path[j]]
        vertices = common_vertices_triangles(T_u, T_v, triangulation)

        c_u = centroid_path[j-1] # centre of the triangle we are leaving
        a, b = vertices[0], vertices[1]
        if sign(c_u, a, b) < 0:
            a, b = b, a

        portal_left.append(a)   # to our right when leaving T_u
        portal_right.append(b)  # to our left

    portal_left.append(end)
    portal_right.append(end)

    return portal_left, portal_right

def string_pull(portals_left, portals_right):

    portal_apex = portals_left[0]
    right_apex = portals_right[0]
    left_apex = portals_left[0]

    tail = []
    tail.append(portal_apex)

    apex_index, left_index, right_index = 0, 0, 0

    i = 1
    while i < len(portals_left):

        left = portals_left[i]
        right = portals_right[i]

        # sign <= 0, is to the left or collinear
        # sign > 0, is to the right

        if sign(right, portal_apex, right_apex) <= 0.0: # new right point is to the left, possibly tightening funnel

            # We updated the apex last step  or Tightening the funnel keeps the right border to the right of the left border
            if np.allclose(portal_apex, right_apex) or sign(right, portal_apex, left_apex) > 0:
                right_apex = right
                right_index = i
            else: # Tightening would cross over the left border, need to add point
                tail.append(left_apex)
                # Set new apex to the left node
                portal_apex = left_apex
                apex_index = left_index
                left_apex = portal_apex
                right_apex = portal_apex
                left_index = apex_index
                right_index = apex_index

                # Restart Scan
                i = apex_index + 1
                continue

        if sign(left, portal_apex, left_apex) >= 0.0: # New left point is to the right, possibly tightening the funnel

            # We updated the apex last step or Tightening the funnel keeps the left border to the left of the right border
            if np.allclose(portal_apex, left_apex) or sign(left, portal_apex, right_apex) < 0.0:
                left_apex = left
                left_index = i
            else: # Tightening would cros over the right border, need to add point
                tail.append(right_apex)

                # Set new apex to the right node
                portal_apex = right_apex
                apex_index = right_index
                left_apex = portal_apex
                right_apex = portal_apex
                left_index = apex_index
                right_index = apex_index

                # Reset scan
                i = apex_index + 1
                continue

        i += 1

    if not np.allclose(tail[-1], portals_left[-1]):
        tail.append(portals_left[-1]) # Add end point to path

    # Clean path in case doubled entries are still present
    points = [tail[0]]
    for p in tail[1:]:
        if not np.allclose(p, points[-1]):
            points.append(p)

    return points
        
def funnel_algorithm(centroid_path, triangle_path, triangulation, start, end):

   portals_left, portals_right = create_funnel_portals(centroid_path, triangle_path,
                                                       triangulation, start, end)


   return string_pull(portals_left, portals_right)

def generate_reference_trajectory(json_path: str):

    room, start, end = extract_vertices_and_segments(json_path)

    t = tr.triangulate(room, 'pn')

    _, T = generate_graphs(t)

    path = shortest_triangle_path(T, t["vertices"], t["triangles"], start, end)

    centroid_path = shortest_centroid_path(t["vertices"], t["triangles"], path)

    funnel_path = np.array(funnel_algorithm(centroid_path, path, t, start, end))

    log_room_and_path(t, room, path, funnel_path, start, end)

    return funnel_path

def log_room_and_path(triangulation, room,
                   shortest_path, funnel_path, start, end):

    tr.compare(plt, room ,triangulation)
    ax = plt.gcf().axes[-1]
    plot_path(ax, triangulation["vertices"],
               triangulation["triangles"], shortest_path, start, end)
    ax.plot(funnel_path[:, 0], funnel_path[:, 1], color="magenta", linewidth=2,
            marker="o", markersize=4, zorder=7, label="funnel path")
    ax.plot(*start, marker="s", color="green", markersize=8, zorder=8)
    ax.plot(*end, marker="*", color="red", markersize=12, zorder=8)
    ax.legend()
    wandb.log({
            "Room": wandb.Image(plt.gcf())
        })
    plt.close("all")

def test_triangulation(json_path: str):

    room, start, end = extract_vertices_and_segments(json_path)

    t = tr.triangulate(room, 'pn')
    print(t)

    D, T = generate_graphs(t)
    path = shortest_triangle_path(T, t["vertices"], t["triangles"], start, end)
    print(path)

    centroid_path = shortest_centroid_path(t["vertices"], t["triangles"], path)
    funnel = np.array(funnel_algorithm(centroid_path, path, t, start, end))

    print(f"Funnel points: {funnel} and its shape: {funnel.shape}")

    tr.compare(plt, room, t)
    ax = plt.gcf().axes[-1]
    plot_path(ax, t["vertices"], t["triangles"], path, start, end)
    ax.plot(funnel[:, 0], funnel[:, 1], color="magenta", linewidth=2,
            marker="o", markersize=4, zorder=7, label="funnel path")
    ax.plot(*start, marker="s", color="green", markersize=8, zorder=8)
    ax.plot(*end, marker="*", color="red", markersize=12, zorder=8)
    ax.legend()
    plt.show()
    

if __name__ == "__main__":
    test_triangulation("rooms/cross_room.json")

    
