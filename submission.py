# coding=utf-8
"""
This file is your main submission that will be graded against. Only copy-paste
code on the relevant classes included here. Do not add any classes or functions
to this file that are not part of the classes that we want.
"""

import heapq
import os
import pickle
import math


class PriorityQueue(object):
    """
    A queue structure where each element is served in order of priority.

    Elements in the queue are popped based on the priority with higher priority
    elements being served before lower priority elements.  If two elements have
    the same priority, they will be served in the order they were added to the
    queue.

    Traditionally priority queues are implemented with heaps, but there are any
    number of implementation options.

    (Hint: take a look at the module heapq)

    Attributes:
        queue (list): Nodes added to the priority queue.
    """

    def __init__(self):
        """Initialize a new Priority Queue."""

        self.queue = []
        self.counter = 0

    def pop(self):
        """
        Pop top priority node from queue.

        Returns:
            The node with the highest priority.
        """

        # TODO: finish this function!
        top = heapq.heappop(self.queue)
        return [top[x] for x in range(len(top)) if x != len(top) - 2]

    def remove(self, node):
        """
        Remove a node from the queue.

        Hint: You might require this in ucs. However, you may
        choose not to use it or to define your own method.

        Args:
            node (tuple): The node to remove from the queue.
        """

        nodes_need_append = []
        for i in range(len(self.queue)):
            poped = heapq.heappop(self.queue)
            if poped[-1] == node:
                break
            else:
                nodes_need_append.append(poped)
        for node in nodes_need_append:
            heapq.heappush(self.queue, node)

    def __iter__(self):
        """Queue iterator."""

        return iter(sorted(self.queue))

    def __str__(self):
        """Priority Queue to string."""

        return 'PQ:%s' % self.queue

    def append(self, node):
        """
        Append a node to the queue.

        Args:
            node: Comparable Object to be added to the priority queue.
        """

        # TODO: finish this function!
        anc = [item for item in node]
        temp = anc[-1]
        anc[-1] = self.counter
        anc.append(temp)
        heapq.heappush(self.queue, anc)
        self.counter += 1

    def __contains__(self, key):
        """
        Containment Check operator for 'in'

        Args:
            key: The key to check for in the queue.

        Returns:
            True if key is found in queue, False otherwise.
        """

        return key in [n[-1] for n in self.queue]

    def __eq__(self, other):
        """
        Compare this Priority Queue with another Priority Queue.

        Args:
            other (PriorityQueue): Priority Queue to compare against.

        Returns:
            True if the two priority queues are equivalent.
        """

        return self.queue == other.queue

    def size(self):
        """
        Get the current size of the queue.

        Returns:
            Integer of number of items in queue.
        """

        return len(self.queue)

    def clear(self):
        """Reset queue to empty (no nodes)."""

        self.queue = []

    def top(self):
        """
        Get the top item in the queue.

        Returns:
            The first item stored in the queue.
        """

        return self.queue[0]


def breadth_first_search(graph, start, goal):
    """
    Warm-up exercise: Implement breadth-first-search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    q = PriorityQueue()
    seen = set()
    res = []
    if start == goal:
        return res

    init = [0, start]
    q.append(init)
    seen.add(start)
    edge = {}

    while q:
        prior, node = q.pop()
        if node == goal:
            return res

        for x in sorted(graph.neighbors(node)):
            if x not in seen:
                seen.add(x)
                edge[x] = node
                if x == goal:
                    res.append(goal)
                    cur = goal
                    while cur != start:
                        prev = edge[cur]
                        res.append(prev)
                        cur = prev
                    res = res[::-1]
                    return res
                else:
                    q.append([prior+1, x])


def uniform_cost_search(graph, start, goal):
    """
    Warm-up exercise: Implement uniform_cost_search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    q = PriorityQueue()
    res = []

    if start == goal:
        return res

    init = [0, start]
    q.append(init)
    edge = {}
    path = {}
    explored = set()
    while q:
        weight, node = q.pop()
        if node not in explored:
            if node == goal:
                res.append(goal)
                cur = goal
                while cur != start:
                    prev = edge[cur]
                    res.append(prev)
                    cur = prev
                res = res[::-1]
                return res

            explored.add(node)
            for x in graph.neighbors(node):
                w = weight + graph.get_edge_weight(node, x)
                if x not in path.keys():
                    path[x] = w
                    q.append([w,x])
                    edge[x] = node
                else:
                    if path[x] > w:
                        path[x] = w
                        q.append([w,x])
                        edge[x] = node

    return res


def null_heuristic(graph, v, goal):
    """
    Null heuristic used as a base line.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        0
    """

    return 0


def euclidean_dist_heuristic(graph, v, goal):
    """
    Warm-up exercise: Implement the euclidean distance heuristic.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Euclidean distance between `v` node and `goal` node
    """

    # TODO: finish this function!
    vPos = graph.nodes[v]['pos']
    gPos = graph.nodes[goal]['pos']
    squares = [(x - y) ** 2 for x, y in zip(vPos, gPos)]
    return sum(squares) ** .5


def a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """
    Warm-up exercise: Implement A* algorithm.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    q = PriorityQueue()
    res = []

    if start == goal:
        return res

    init = [heuristic(graph,start,goal), 0, start]
    q.append(init)
    explored = set()
    edge = {}
    path = {}

    while q:
        he, p, node = q.pop()
        if node not in explored:
            if node == goal:
                res.append(goal)
                cur = goal
                while cur != start:
                    prev = edge[cur]
                    res.append(prev)
                    cur = prev
                res = res[::-1]
                return res

            explored.add(node)
            for x in graph.neighbors(node):
                pa = p + graph.get_edge_weight(node, x)
                h = heuristic(graph,x,goal)
                f = pa + h
                if x not in path.keys():
                    path[x] = pa
                    q.append([f,pa,x])
                    edge[x] = node
                else:
                    if path[x] > pa:
                        path[x] = pa
                        q.append([f,pa,x])
                        edge[x] = node

    return res


def bidirectional_ucs(graph, start, goal):
    """
    Exercise 1: Bidirectional Search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    res = []

    if start == goal:
        return res
    qs = PriorityQueue()
    qg = PriorityQueue()
    edges = {}
    edgeg = {}
    paths = {}
    pathg = {}
    ress = []
    resg = []
    inits = [0,start]
    initg = [0,goal]
    qs.append(inits)
    qg.append(initg)
    mu = float('inf')
    frontier = None
    paths[start] = 0
    pathg[goal] = 0
    while qs and qg:
        weights, nodes = qs.pop()
        weightg, nodeg = qg.pop()
        if weights + weightg >= mu:
            ress.append(frontier)
            curs = frontier
            while curs != start:
                prevs = edges[curs]
                ress.append(prevs)
                curs = prevs
            ress = ress[::-1]

            curg = frontier
            while curg != goal:
                prevg = edgeg[curg]
                resg.append(prevg)
                curg = prevg

            res = ress + resg
            return res

        if nodes in pathg.keys():
            if weights + pathg[nodes] < mu:
                mu = weights + pathg[nodes]
                frontier = nodes

        for x in graph.neighbors(nodes):
            ws = weights + graph.get_edge_weight(nodes, x)
            if x not in paths.keys():
                paths[x] = ws
                qs.append([ws, x])
                edges[x] = nodes
            else:
                if paths[x] > ws:
                    paths[x] = ws
                    qs.append([ws,x])
                    edges[x] = nodes

        if nodeg in paths.keys():
            if weightg + paths[nodeg] < mu:
                mu = weightg + paths[nodeg]
                frontier = nodeg

        for x in graph.neighbors(nodeg):
            wg = weightg + graph.get_edge_weight(nodeg, x)
            if x not in pathg.keys():
                pathg[x] = wg
                qg.append([wg, x])
                edgeg[x] = nodeg
            else:
                if pathg[x] > wg:
                    pathg[x] = wg
                    qg.append([wg, x])
                    edgeg[x] = nodeg

    return res


def bidirectional_a_star(graph, start, goal,
                         heuristic=euclidean_dist_heuristic):
    """
    Exercise 2: Bidirectional A*.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    res = []

    if start == goal:
        return res

    qs = PriorityQueue()
    qg = PriorityQueue()
    exploreds = set()
    exploredg = set()
    edges = {}
    edgeg = {}
    paths = {}
    pathg = {}
    ress = []
    resg = []
    inits = [heuristic(graph, start, goal),0, start]
    initg = [0,0,goal]
    qs.append(inits)
    qg.append(initg)
    mu = float('inf')
    frontier = None
    paths[start] = heuristic(graph, start, goal)
    pathg[goal] = 0
    while qs and qg:
        hs, ps, nodes = qs.pop()
        hg, pg, nodeg = qg.pop()
        exploreds.add(nodes)
        exploredg.add(nodeg)
        if hs + hg > mu:
            ress.append(frontier)
            curs = frontier
            while curs != start:
                prevs = edges[curs]
                ress.append(prevs)
                curs = prevs
            ress = ress[::-1]

            curg = frontier
            while curg != goal:
                prevg = edgeg[curg]
                resg.append(prevg)
                curg = prevg

            res = ress + resg
            return res
    #
    #     if nodes in pathg.keys():
    #         if hs + pathg[nodes] < mu:
    #             mu = hs + pathg[nodes]
    #             frontier = nodes
    #
        pfs = 0.5 * (heuristic(graph, nodes, goal) - heuristic(graph, nodes, start))
        for x in graph.neighbors(nodes):
            pas = hs + graph.get_edge_weight(nodes, x)
            h = 0.5 * (heuristic(graph, x, goal) - heuristic(graph, x, start)) - pfs
            f = pas + h
            if x not in exploreds and x not in qs:
                paths[x] = f
                qs.append([f, pas, x])
                edges[x] = nodes
                if x not in paths.keys():
                    paths[x] = f
                    qs.append([f, pas, x])
                    edges[x] = nodes
            elif x in qs:
                if paths[x] > f:
                    paths[x] = f
                    qs.remove(x)
                    qs.append([f, pas, x])
                    edges[x] = nodes

            if x in exploredg:
                if paths[x] + pathg[x] < mu:
                    mu = paths[x] + pathg[x]
                    frontier = x

        pfg = 0.5 * (heuristic(graph, nodeg, start) - heuristic(graph, nodeg, goal))
        for x in graph.neighbors(nodeg):
            pag = hg + graph.get_edge_weight(nodeg, x)
            h = 0.5 * (heuristic(graph, x, start) - heuristic(graph, x, goal)) - pfg
            f = pag + h
            if x not in exploredg and x not in qg:
                pathg[x] = f
                qg.append([f, pag, x])
                edgeg[x] = nodeg
                if x not in pathg.keys():
                    pathg[x] = f
                    qg.append([f, pag, x])
                    edgeg[x] = nodeg
            elif x in qg:
                if pathg[x] > f:
                    pathg[x] = f
                    qg.remove(x)
                    qg.append([f, pag, x])
                    edgeg[x] = nodeg

            if x in exploreds:
                if paths[x] + pathg[x] < mu:
                    mu = paths[x] + pathg[x]
                    frontier = x

    return res


def tridirectional_search(graph, goals):
    """
    Exercise 3: Tridirectional UCS Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    # TODO: finish this function
    res = []
    if goals[0] == goals[1] == goals[2]:
        return res

    q1 = PriorityQueue()
    q2 = PriorityQueue()
    q3 = PriorityQueue()
    explored1 = set()
    explored2 = set()
    explored3 = set()
    edge1 = {}
    edge2 = {}
    edge3 = {}
    init1 = [0, goals[0]]
    init2 = [0, goals[1]]
    init3 = [0, goals[2]]
    q1.append(init1)
    q2.append(init2)
    q3.append(init3)
    path1 = {}
    path2 = {}
    path3 = {}
    path1[goals[0]] = 0
    path2[goals[1]] = 0
    path3[goals[2]] = 0
    mu12 = float('inf')
    mu23 = float('inf')
    mu31 = float('inf')
    frontier12 = None
    frontier23 = None
    frontier31 = None
    res12 = []
    res23 = []
    res31 = []
    while q1 and q2 and q3:
        w1, idx1, n1 = q1.top()
        w2, idx2, n2 = q2.top()
        w3, idx3, n3 = q3.top()
        if w1 + w2 >= mu12 and w2 + w3 >= mu23 and w3 + w1 >= mu31:
            break

        w1, n1 = q1.pop()
        w2, n2 = q2.pop()
        w3, n3 = q3.pop()
        explored1.add(n1)
        explored2.add(n2)
        explored3.add(n3)

        for x in sorted(graph.neighbors(n1)):
            we1 = w1 + graph.get_edge_weight(n1, x)
            if x not in explored1 and x not in q1:
                path1[x] = we1
                q1.append([we1, x])
                edge1[x] = n1
            elif x in q1:
                if path1[x] > we1:
                    path1[x] = we1
                    q1.remove(x)
                    q1.append([we1, x])
                    edge1[x] = n1

            if x in explored2:
                if path1[x] + path2[x] < mu12:
                    mu12 = path1[x] + path2[x]
                    frontier12 = x

            if x in explored3:
                if path1[x] + path3[x] < mu31:
                    mu31 = path1[x] + path3[x]
                    frontier31 = x

        for x in sorted(graph.neighbors(n2)):
            we2 = w2 + graph.get_edge_weight(n2, x)
            if x not in explored2 and x not in q2:
                path2[x] = we2
                q2.append([we2, x])
                edge2[x] = n2
            elif x in q2:
                if path2[x] > we2:
                    path2[x] = we2
                    q2.remove(x)
                    q2.append([we2, x])
                    edge2[x] = n2

            if x in explored1:
                if path1[x] + path2[x] < mu12:
                    mu12 = path1[x] + path2[x]
                    frontier12 = x

            if x in explored3:
                if path2[x] + path3[x] < mu23:
                    mu23 = path2[x] + path3[x]
                    frontier23 = x

        for x in sorted(graph.neighbors(n3)):
            we3 = w3 + graph.get_edge_weight(n3, x)
            if x not in explored3 and x not in q3:
                path3[x] = we3
                q3.append([we3, x])
                edge3[x] = n3
            elif x in q3:
                if path3[x] > we3:
                    path3[x] = we3
                    q3.remove(x)
                    q3.append([we3, x])
                    edge3[x] = n3

            if x in explored2:
                if path3[x] + path2[x] < mu23:
                    mu23 = path3[x] + path2[x]
                    frontier23 = x

            if x in explored1:
                if path1[x] + path3[x] < mu31:
                    mu31 = path1[x] + path3[x]
                    frontier31 = x

    def constructPath(re,frontier,edgea,edgeb,a,b):
        r1 = []
        r2 = []
        r1.append(frontier)
        cur1 = frontier
        while cur1 != a:
            prev1 = edgea[cur1]
            r1.append(prev1)
            cur1 = prev1
        r1 = r1[::-1]

        cur2 = frontier
        while cur2 != b:
            prev2 = edgeb[cur2]
            r2.append(prev2)
            cur2 = prev2

        re += r1 + r2

    constructPath(res12, frontier12, edge1, edge2, goals[0], goals[1])
    constructPath(res23, frontier23, edge2, edge3, goals[1], goals[2])
    constructPath(res31, frontier31, edge3, edge1, goals[2], goals[0])

    if mu12 >= mu23 and mu12 >= mu31:
        res = res23[:-1] + res31

    if mu23 >= mu31 and mu23 >= mu12:
        res = res31[:-1] + res12

    if mu31 >= mu23 and mu31 >= mu12:
        res = res12[:-1] + res23

    return res


def tridirectional_upgraded(graph, goals, heuristic=euclidean_dist_heuristic, landmarks=None):
    """
    Exercise 4: Upgraded Tridirectional Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.
        landmarks: Iterable containing landmarks pre-computed in compute_landmarks()
            Default: None

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    # TODO: finish this function
    res = []
    if goals[0] == goals[1] == goals[2]:
        return res

    q1 = PriorityQueue()
    q2 = PriorityQueue()
    q3 = PriorityQueue()
    explored1 = set()
    explored2 = set()
    explored3 = set()
    edge1 = {}
    edge2 = {}
    edge3 = {}
    init1 = [min(heuristic(graph,goals[0],goals[1]), heuristic(graph,goals[0],goals[2])),0, goals[0]]
    init2 = [min(heuristic(graph,goals[1],goals[0]), heuristic(graph,goals[1],goals[2])),0, goals[1]]
    init3 = [min(heuristic(graph,goals[2],goals[1]), heuristic(graph,goals[2],goals[0])),0, goals[2]]
    q1.append(init1)
    q2.append(init2)
    q3.append(init3)
    path1 = {}
    path2 = {}
    path3 = {}
    path1[goals[0]] = min(heuristic(graph,goals[0],goals[1]), heuristic(graph,goals[0],goals[2]))
    path2[goals[1]] = min(heuristic(graph,goals[1],goals[0]), heuristic(graph,goals[1],goals[2]))
    path3[goals[2]] = min(heuristic(graph,goals[2],goals[1]), heuristic(graph,goals[2],goals[0]))
    mu12 = float('inf')
    mu23 = float('inf')
    mu31 = float('inf')
    frontier12 = None
    frontier23 = None
    frontier31 = None
    res12 = []
    res23 = []
    res31 = []
    while q1 and q2 and q3:
        h1, p1, n1 = q1.pop()
        h2, p2, n2 = q2.pop()
        h3, p3, n3 = q3.pop()
        explored1.add(n1)
        explored2.add(n2)
        explored3.add(n3)
        if h1 + h2 >= mu12 and h2 + h3 >= mu23 and h3 + h1 >= mu31:
            break

        for x in sorted(graph.neighbors(n1)):
            pa1 = h1 + graph.get_edge_weight(n1, x)
            f = pa1 + min(heuristic(graph,x,goals[1]), heuristic(graph,x,goals[2])) - min(heuristic(graph,n1,goals[1]), heuristic(graph,n1,goals[2]))
            if x not in explored1 and x not in q1:
                path1[x] = f
                q1.append([f, pa1, x])
                edge1[x] = n1
            elif x in q1:
                if path1[x] > f:
                    path1[x] = f
                    q1.remove(x)
                    q1.append([f, pa1, x])
                    edge1[x] = n1

            if x in explored2:
                if path1[x] + path2[x] < mu12:
                    mu12 = path1[x] + path2[x]
                    frontier12 = x

            if x in explored3:
                if path1[x] + path3[x] < mu31:
                    mu31 = path1[x] + path3[x]
                    frontier31 = x

        for x in sorted(graph.neighbors(n2)):
            pa2 = h2 + graph.get_edge_weight(n2, x)
            f = pa2 + min(heuristic(graph,x,goals[0]), heuristic(graph,x,goals[2])) - min(heuristic(graph,n2,goals[0]), heuristic(graph,n2,goals[2]))
            if x not in explored2 and x not in q2:
                path2[x] = f
                q2.append([f, pa2, x])
                edge2[x] = n2
            elif x in q2:
                if path2[x] > f:
                    path2[x] = f
                    q2.remove(x)
                    q2.append([f, pa2, x])
                    edge2[x] = n2

            if x in explored1:
                if path1[x] + path2[x] < mu12:
                    mu12 = path1[x] + path2[x]
                    frontier12 = x

            if x in explored3:
                if path2[x] + path3[x] < mu23:
                    mu23 = path2[x] + path3[x]
                    frontier23 = x

        for x in sorted(graph.neighbors(n3)):
            pa3 = h3 + graph.get_edge_weight(n3, x)
            f = pa3 + min(heuristic(graph,x,goals[0]), heuristic(graph,x,goals[1])) - min(heuristic(graph,n3,goals[0]), heuristic(graph,n3,goals[1]))
            if x not in explored3 and x not in q3:
                path3[x] = f
                q3.append([f, pa3, x])
                edge3[x] = n3
            elif x in q3:
                if path3[x] > f:
                    path3[x] = f
                    q3.remove(x)
                    q3.append([f, pa3, x])
                    edge3[x] = n3

            if x in explored2:
                if path3[x] + path2[x] < mu23:
                    mu23 = path3[x] + path2[x]
                    frontier23 = x

            if x in explored1:
                if path1[x] + path3[x] < mu31:
                    mu31 = path1[x] + path3[x]
                    frontier31 = x

    def constructPath(re, frontier, edgea, edgeb, a, b):
        r1 = []
        r2 = []
        r1.append(frontier)
        cur1 = frontier
        while cur1 != a:
            prev1 = edgea[cur1]
            r1.append(prev1)
            cur1 = prev1
        r1 = r1[::-1]

        cur2 = frontier
        while cur2 != b:
            prev2 = edgeb[cur2]
            r2.append(prev2)
            cur2 = prev2

        re += r1 + r2

    constructPath(res12, frontier12, edge1, edge2, goals[0], goals[1])
    constructPath(res23, frontier23, edge2, edge3, goals[1], goals[2])
    constructPath(res31, frontier31, edge3, edge1, goals[2], goals[0])

    if mu12 >= mu23 and mu12 >= mu31:
        res = res23[:-1] + res31

    if mu23 >= mu31 and mu23 >= mu12:
        res = res31[:-1] + res12

    if mu31 >= mu23 and mu31 >= mu12:
        res = res12[:-1] + res23

    return res


def return_your_name():
    """Return your name from this function"""
    # TODO: finish this function
    name = 'Yanxiang Zhou'
    return name


def compute_landmarks(graph):
    """
    Feel free to implement this method for computing landmarks. We will call
    tridirectional_upgraded() with the object returned from this function.

    Args:
        graph (ExplorableGraph): Undirected graph to search.

    Returns:
    List with not more than 4 computed landmarks.
    """
    return None


def custom_heuristic(graph, v, goal):
    """
       Feel free to use this method to try and work with different heuristics and come up with a better search algorithm.
       Args:
           graph (ExplorableGraph): Undirected graph to search.
           v (str): Key for the node to calculate from.
           goal (str): Key for the end node to calculate to.
       Returns:
           Custom heuristic distance between `v` node and `goal` node
       """
    pass


# Extra Credit: Your best search method for the race
def custom_search(graph, start, goal, data=None):
    """
    Race!: Implement your best search algorithm here to compete against the
    other student agents.

    If you implement this function and submit your code to Gradescope, you'll be
    registered for the Race!

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        data :  Data used in the custom search.
            Will be passed your data from load_data(graph).
            Default: None.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    raise NotImplementedError


def load_data(graph, time_left):
    """
    Feel free to implement this method. We'll call it only once
    at the beginning of the Race, and we'll pass the output to your custom_search function.
    graph: a networkx graph
    time_left: function you can call to keep track of your remaining time.
        usage: time_left() returns the time left in milliseconds.
        the max time will be 10 minutes.

    * To get a list of nodes, use graph.nodes()
    * To get node neighbors, use graph.neighbors(node)
    * To get edge weight, use graph.get_edge_weight(node1, node2)
    """

    # nodes = graph.nodes()
    return None


def haversine_dist_heuristic(graph, v, goal):
    """
    Note: This provided heuristic is for the Atlanta race.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Haversine distance between `v` node and `goal` node
    """

    # Load latitude and longitude coordinates in radians:
    vLatLong = (math.radians(graph.nodes[v]["pos"][0]), math.radians(graph.nodes[v]["pos"][1]))
    goalLatLong = (math.radians(graph.nodes[goal]["pos"][0]), math.radians(graph.nodes[goal]["pos"][1]))

    # Now we want to execute portions of the formula:
    constOutFront = 2 * 6371  # Radius of Earth is 6,371 kilometers
    term1InSqrt = (math.sin((goalLatLong[0] - vLatLong[0]) / 2)) ** 2  # First term inside sqrt
    term2InSqrt = math.cos(vLatLong[0]) * math.cos(goalLatLong[0]) * (
                (math.sin((goalLatLong[1] - vLatLong[1]) / 2)) ** 2)  # Second term
    return constOutFront * math.asin(math.sqrt(term1InSqrt + term2InSqrt))  # Straight application of formula
