import numpy as np
from message_types.msg_waypoints import msg_waypoints


class planRRT():
    def __init__(self):
        self.waypoints = msg_waypoints()
        self.segmentLength = 200 # standard length of path segments

    def planPath(self, wpp_start, wpp_end, map):

        # desired down position is down position of end node
        pd = -100 #wpp_end.item(2)

        # specify start and end nodes from wpp_start and wpp_end
        # format: N, E, D, cost, parentIndex, connectsToGoalFlag,
        start_node = np.array([wpp_start.item(0), wpp_start.item(1), pd, 0, 0, 0])
        end_node = np.array([wpp_end.item(0), wpp_end.item(1), pd, 0, 0, 0])

        # establish tree starting with the start node
        tree = start_node.reshape(1, -1)

        # check to see if start_node connects directly to end_node
        if ((np.linalg.norm(start_node[0:3] - end_node[0:3]) < self.segmentLength ) and not self.collision(start_node, end_node, map)):
            self.waypoints.ned = end_node[0:3]
        else:
            numPaths = 0
            while numPaths < 3:
                tree, flag = self.extendTree(tree, end_node, self.segmentLength, map, pd)
                numPaths = numPaths + flag

        # find path with minimum cost to end_node
        path = self.findMinimumPath(tree, end_node)
        waypoints_smoothed = self.smoothPath(path, map)
        self.waypoints.ned = np.hstack((waypoints_smoothed[:, :3].T, waypoints_smoothed[-1, :3].reshape(-1, 1)))
        # self.waypoints.ned = np.hstack((waypoints_smoothed[:, :3].T, np.array([[np.inf], [np.inf], [np.inf]])))
        self.waypoints.airspeed = np.ones(len(waypoints_smoothed)) * 25.0
        self.waypoints.num_waypoints = len(waypoints_smoothed)
        return self.waypoints

    def generateRandomNode(self, map, pd): #, chi):
        pn = np.random.uniform(0, map.city_width)
        pe = np.random.uniform(0, map.city_width)
        return np.array([pn, pe, pd])

    def collision(self, start_node, end_node, map):
        # Hayden's Vectorized approach
        collided = False
        buffer = 30.
        pts = self.pointsAlongPath(start_node, end_node)
        for i in range(len(pts)):
            col_n = np.abs(map.building_north - pts[i, 0]) < (map.building_width/2.0 + buffer)
            col_e = np.abs(map.building_east - pts[i, 1]) < (map.building_width/2.0 + buffer)
            arg_n = np.where(col_n)[0]
            arg_e = np.where(col_e)[0]
            if len(arg_n) == 0 and len(arg_e) == 0:
                continue
            if map.building_height[arg_n, arg_e] >= -pts[i, 2]:
                collided = True
        return collided


    def pointsAlongPath(self, start_node, end_node): #, Del):
        N = 100  # points between nodes
        q = (end_node[0:3] - start_node[0:3])/np.linalg.norm(end_node[0:3] - start_node[0:3])
        dist = np.linalg.norm(end_node[0:3] - start_node[0:3]) / N
        points = start_node[0:3].reshape(1, 3)
        for i in range(N):
            points = np.vstack((points, points[i, :] + q * dist))
        return points

    def downAtNE(self, map, n, e):
        return

    def extendTree(self, tree, end_node, segmentLength, map, pd):
        valid_addition = False
        while not valid_addition:
            p = self.generateRandomNode(map, pd)
            n_dist = (p.item(0) - tree[:, 0])**2 + (p.item(1) - tree[:, 1])**2 + (p.item(2) - tree[:, 2])**2
            parent = np.argmin(n_dist)
            n_closest = tree[parent, :3]
            q = (p - n_closest)/np.linalg.norm(p - n_closest)
            v_star = n_closest + q*segmentLength
            if self.collision(n_closest, v_star, map):
                continue
            else:
                cost = n_dist[parent] + tree[parent, 3]
                n_new = np.append(v_star, [parent, cost, 0])
                tree = np.vstack((tree, n_new))
                valid_addition = True
        if np.linalg.norm(end_node[0:3] - v_star) < segmentLength:
            flag = 1
            tree[-1, 5] = 1
        else:
            flag = 0
        return tree, flag

    def findMinimumPath(self, tree, end_node):
        term_pts = tree[np.where(tree[:, 5] == 1)]  # find all completion nodes
        path = np.array([term_pts[np.argmin(term_pts[:, 4])]])  # find min cost completion node
        finish = False
        while not finish:
            parent = path[0, 3]
            path = np.vstack((tree[int(parent)], path))
            if parent == 0:
                finish = True
                path = np.vstack((path, end_node))
        return path

    def smoothPath(self, path, map):
        w_s = np.array([path[0, :]])
        i = 0
        j = 1
        while j < len(path)-1:
            if self.collision(path[i, :], path[j+1, :], map):
                w_s = np.vstack([w_s, path[j, :]])
                i = j
            j += 1
        w_s = np.vstack([w_s, path[-1, :]])
        return w_s

