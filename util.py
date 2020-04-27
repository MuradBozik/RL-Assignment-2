import os
import time
from datetime import datetime, timedelta
import numpy as np
from hex_skeleton import HexBoard, Hex
from collections import deque
from node import Node
import queue  # TODO: Remove before finish
import graphviz as gv  # TODO: Remove before finish

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'  # TODO: Remove before finish


class UTIL:
    def __init__(self, infinity, maximizer, minimizer):
        self.INF = infinity
        self.MAXIMIZER = maximizer
        self.MINIMIZER = minimizer

    # region AlphaBetaSearch
    def alphaBetaSearchDijkstra(self, node, depth, alpha, beta, isMaximizer):
        node.searched = True

        if depth == 0 or node.type == 'LEAF':
            # calculate the value
            if isMaximizer:
                node.value = self.dijkstraEval(node.board, self.MAXIMIZER)
            else:
                node.value = self.dijkstraEval(node.board, self.MINIMIZER)
            return node.value

        # If we already get this nodes children then this section will be skipped
        if len(node.children) <= 0:
            # get children of this node
            children = []
            moves = HexBoard.getMoveList(node)

            # If there is no possible move then this is a leaf node
            if len(moves) == 0:
                if isMaximizer:
                    node.value = self.dijkstraEval(node.board, self.MAXIMIZER)
                else:
                    node.value = self.dijkstraEval(node.board, self.MINIMIZER)
                return node.value
            else:
                for move in moves:
                    b = node.board.copy()
                    if isMaximizer:
                        b[move] = self.MAXIMIZER
                    else:
                        b[move] = self.MINIMIZER

                    child_type = 'MIN'
                    if node.type == 'MIN':
                        child_type = 'MAX'

                    child = Node(node_type=child_type, board_state=b, parent=node)
                    children.append(child)

            node.children = children

        # Which one is the first player?
        if isMaximizer:
            bestVal = -self.INF
            for n in node.children:
                bestVal = max(bestVal, self.alphaBetaSearchDijkstra(n, depth - 1, alpha, beta, False))
                alpha = max(alpha, bestVal)  # Updating alpha
                if bestVal >= beta:
                    break  # beta cutoff, a>=b
        else:
            bestVal = self.INF
            for n in node.children:
                bestVal = min(bestVal, self.alphaBetaSearchDijkstra(n, depth - 1, alpha, beta, True))
                beta = min(beta, bestVal)  # Updating beta
                if alpha >= bestVal:
                    break  # alpha cutoff, a>=b

        node.value = bestVal

        return bestVal

    @staticmethod
    def getBestMove(node, bestVal):
        bestChildren = []
        for child in node.children:
            if child.value == bestVal:
                bestChildren.append(child)

        if len(bestChildren) > 0:
            index = np.random.randint(0, len(bestChildren))
            bestNode = bestChildren[index]
        else:
            # print("Error, best Move couldn't find. Randomly selects the move!")
            moves = HexBoard.getMoveList(node)
            index = np.random.randint(0, len(moves))
            return moves[index]

        b_size = node.board.shape[0]

        for i in range(b_size):
            for j in range(b_size):
                if node.board[i, j] != bestNode.board[i, j]:
                    return i, j

    @staticmethod
    def updateNode(node, board):
        """ After making a move root node should change to the correct child
            If there is no child and a random move has been made, then returns new created node """
        if len(node.children) > 0:
            for child in node.children:
                if np.array_equal(child.board, board.board):
                    return child

        node_type = 'MAX'
        if node.type == 'MAX' or (node.parent_type == 'MIN'):
            node_type = 'MIN'
        return Node(node_type=node_type, board_state=board.board, parent=node)

    def iterativeDeepening(self, node, maximizer, processtime=1, maxdepth = 1):
        depth = 1
        end_time = datetime.now() + timedelta(seconds=processtime)
        bestScore = 0  # just initialized not to get an error
        # now = time.time()
        while datetime.now() < end_time and depth <= maxdepth:
            # begin = time.time()
            # Use line below if you want to press a key to stop iterative deepening
            # while datetime.now() < end_time and not keyboard.is_pressed('space'):
            bestScore = self.alphaBetaSearchDijkstra(node, depth, -self.INF, self.INF, isMaximizer=maximizer)
            # print(f"Depth {depth} completed (ID-tt)!: It took {time.time() - begin}s")
            depth = depth + 1
        # print(f"Iteration completed (ID-tt)!: It took {time.time() - now}s")
        return bestScore
    # endregion AlphaBetaSearch


    # region Dijsktra
    def updateWeights(self, color, hexboard):
        hexes = hexboard.hexes.copy()
        if color == hexboard.RED:
            for hx in hexes:
                if hexboard.getColor(hx.position) == hexboard.RED:
                    hx.weight = 0
                elif hexboard.getColor(hx.position) == hexboard.BLUE:
                    hx.weight = 99999999
                else:
                    hx.weight = 1
        elif color == hexboard.BLUE:
            for hx in hexes:
                if hexboard.getColor(hx.position) == hexboard.BLUE:
                    hx.weight = 0
                elif hexboard.getColor(hx.position) == hexboard.RED:
                    hx.weight = 99999999
                else:
                    hx.weight = 1
        else:
            print("There is a problem")
        return hexes

    def getMin(self, hexes):
        min_hex = Hex(position=(-3, -3), neighbors=None, weight=99999999, distance=99999999)
        for hx in hexes:
            if hx.distance < min_hex.distance:
                min_hex = hx
        return min_hex

    def getLastRow(self, hexboard, color, hexes):
        for i in range(hexboard.size):
            for hx in hexes:
                if color == hexboard.RED and hx.position == (hexboard.size - 1, i):
                    hx.neighbors.append((-2, -2))
                elif color == hexboard.BLUE and hx.position == (i, hexboard.size - 1):
                    hx.neighbors.append((-2, -2))
        return hexes

    def resetHexes(self, hexboard):
        for hx in hexboard.hexes:
            hx.weight = 1
            hx.distance = 99999999
            hx.neighbors = hexboard.getNeighbors(hx.position)

    def dijkstra(self, hexboard, color):
        self.resetHexes(hexboard)

        start = None
        dest = None

        if color == hexboard.RED:
            Rs_neighbors = []
            Re_neighbors = []
            for i in range(hexboard.size):
                Rs_neighbors.append(hexboard.getHex((0, i)).position)
                Re_neighbors.append(hexboard.getHex((hexboard.size - 1, i)).position)
            #  Creating outer hexes
            # Set the distance to zero for our initial node 
            Rs = Hex(position=(-1, -1), neighbors=Rs_neighbors, weight=0, distance=0)
            Re = Hex(position=(-2, -2), neighbors=Re_neighbors, weight=0, distance=99999999)

            start = Rs
            dest = Re
        elif color == hexboard.BLUE:
            Bs_neighbors = []
            Be_neighbors = []
            for i in range(hexboard.size):
                Bs_neighbors.append(hexboard.getHex((i, 0)).position)
                Be_neighbors.append(hexboard.getHex((i, hexboard.size - 1)).position)
            #  Creating outer hexes
            # Set the distance to zero for our initial node 
            Bs = Hex(position=(-1, -1), neighbors=Bs_neighbors, weight=0, distance=0)
            Be = Hex(position=(-2, -2), neighbors=Be_neighbors, weight=0, distance=99999999)

            start = Bs
            dest = Be

        # 1. Mark all nodes unvisited and store them. 
        updatedWeights_nodes = self.updateWeights(color, hexboard)
        unvisited_nodes = self.getLastRow(hexboard, color, updatedWeights_nodes)
        unvisited_nodes.extend([start, dest])

        previous_vertices = {node: None for node in unvisited_nodes}

        while unvisited_nodes:
            # 3. Select the unvisited node with the smallest distance, 
            # it's current node now.
            current_node = self.getMin(unvisited_nodes)
            # 6. Stop, if the smallest distance 
            # among the unvisited nodes is infinity.
            if current_node.distance == 99999999:
                break

            # 4. Find unvisited neighbors for the current node 
            # and calculate their distances through the current node.
            for neighbor_position in current_node.neighbors:
                if neighbor_position == (-2, -2):
                    neighbor = dest
                else:
                    neighbor = hexboard.getHex(neighbor_position)

                alternative_route = current_node.distance + neighbor.weight

                # Compare the newly calculated distance to the assigned 
                # and save the smaller one.

                if alternative_route < neighbor.distance:
                    neighbor.distance = alternative_route
                    previous_vertices[neighbor] = current_node

            # 5. Mark the current node as visited 
            # and remove it from the unvisited set.
            unvisited_nodes.remove(current_node)

        path, current_node = deque(), dest

        while previous_vertices[current_node] is not None:
            path.appendleft(current_node.position)
            current_node = previous_vertices[current_node]
        if path:
            path.appendleft(current_node.position)

        if len(path) > 0:
            path.remove((-1, -1))
            path.remove((-2, -2))
        return path

    def getFreeHexes(self, hexboard, path):
        freeCounts = 0
        for position in path:
            if hexboard.board[position] == hexboard.EMPTY:
                freeCounts = freeCounts + 1
        return freeCounts

    def dijkstraEval(self, board_state, maximizer):
        h = HexBoard(board_state.shape[0])
        h.board = board_state

        red_shortestPath = self.dijkstra(hexboard=h, color=h.RED)
        blue_shortestPath = self.dijkstra(hexboard=h, color=h.BLUE)

        freeReds = self.getFreeHexes(hexboard=h, path=red_shortestPath)
        freeBlues = self.getFreeHexes(hexboard=h, path=blue_shortestPath)

        if maximizer == h.RED:
            heuristic_score = freeBlues - freeReds
        else:
            heuristic_score = freeReds - freeBlues

        return heuristic_score
    # endregion Dijkstra


    # region Evaluation
    # endregion Evaluation


    # region Visualization TODO: Remove before finish

    def getSearchedNum(self, nodes):
        searched_num = 0
        for node in nodes:
            if node['searched']:
                searched_num = searched_num + 1
        return searched_num

    def visualizeTree(self, root, fname):
        """
        Visualizes tree from the root node
        """

        g = gv.Digraph(fname, filename=fname)
        g.format = 'png'
        nodes, edges = self.getAllnodesAndEdges(root)
        # Default values
        node_shape = 'box'
        pen_color = 'black'
        node_style = ''
        color = ''

        # if root['searched']:  # If the root is searched then search algortihm worked
        #    searched = True

        for node in nodes:
            if node.type == 'MAX':
                node_shape = 'box'
            elif node.type == 'MIN':
                node_shape = 'circle'
            else:
                if node.parent_type == 'MIN':
                    node_shape = 'box'
                else:
                    node_shape = 'circle'

            g.attr('node', shape=node_shape, pencolor=pen_color, style=node_style, color=color)

            node_label = "?"
            node_xlabel = ""
            try:
                node_label = str(f"{node.board}")
                node_xlabel = str(f"n:{node.visit}, w:{node.wins}")
            except KeyError:
                node_label = "?"
                node_xlabel = ""

            g.node(node['id'], label=node_label, xlabel=node_xlabel)

        for node1, node2 in edges:
            g.edge(node1['id'], node2['id'])

        # Styling
        # penwidth='4'
        g.edge_attr.update(arrowhead='none')

        g.render(view=True, cleanup=True, format='png')

    def getAllnodesAndEdges(self, node):
        """
        'Node' represents a board state
        'Edge' represents the connection between two nodes.

        Function returns two parameter:
          - all children and node itself as a list
          - pairs of nodes (parent, child)
        """

        allnodes = list()
        alledges = list()

        q = queue.Queue()
        q.put(node)

        while not q.empty():
            n = q.get()
            for c in n.children:
                allnodes.append(c)
                alledges.append((n, c))
                if c in node.children:
                    q.put(c)

        # Last, add node itself
        allnodes.append(node)
        return allnodes, alledges

    # endregion Visualization
