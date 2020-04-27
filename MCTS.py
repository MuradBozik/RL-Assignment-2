from hex_skeleton import HexBoard
import numpy as np
from copy import deepcopy
from random import choice
from node import Node
import time
from datetime import datetime, timedelta


class MCTS:
    def __init__(self, game, cp=0.8, n=100):
        """ We want to keep track of the game, that's why we parametrized it. """
        self.game = game
        self.Cp = cp  # UCT exploration/exploitation parameter
        self.N = n  # Number of simulations

    def search(self, root, itermax, delta, isMaximizer):
        """Return the best moves based on MCTS"""
        end_time = datetime.now() + timedelta(seconds=delta)
        # now = time.time()
        while datetime.now() < end_time and itermax > 0:
            # We don't want to change the game, we will turn to it in each iteration
            game_state = deepcopy(self.game)
            # we will change the root (will be Expanded)
            node = root

            path = [root.board.tobytes()]

            # Select
            # node is fully expanded and non-terminal
            while (node.untriedMoves == []) and (node.children != []) and (not game_state.isTerminal()):
                node = self.UCTSelectChild(node, isMaximizer)
                m = HexBoard.getMove(game_state.board, node.board)
                game_state = HexBoard.makeMove(m, game_state)
                path.append(node.board.tobytes())

            # Expand
            if (node.untriedMoves != []) and (not game_state.isTerminal()):
                # node is expanded and updated with child
                node = self.Expand(node, game_state)
                path.append(node.board.tobytes())

            # Playouts
            for p in range(self.N):
                # for each playout we want to return to same game_state
                _game = deepcopy(game_state)
                while not _game.isTerminal():
                    move = choice(HexBoard.getFreeMoves(_game.board))
                    _game = HexBoard.makeMove(move, _game)

                # This works just once for a particular node
                if node.wins == np.inf:
                    node.wins = 0

                if node.loss == np.inf:
                    node.loss = 0

                if _game.checkWin(_game.maximizer):
                    node.wins += 1
                else:
                    node.loss += 1

                # print(f'Playout {p} is done!')

            # Backpropagate

            # We are removing current node from path
            path.pop()

            while node is not None:
                # backpropagate works from the current node to the root node
                if len(path) > 0:
                    parent = node.getParent(path.pop())
                    if parent.wins == np.inf:
                        parent.wins = 0
                        parent.loss = 0
                    parent.wins += node.wins
                    parent.loss += node.loss
                else:
                    parent = None

                node.visit += 1
                node = parent
            itermax -= 1
        # print(f"Iteration completed!: It took {time.time() - now}s")
        sortedList = sorted(root.children, key=lambda c: c.visit)

        # return the move that was most visited
        return HexBoard.getMove(root.board, sortedList[-1].board), root

    def UCTSelectChild(self, node, isMaximizer):
        """ If Player is maximizer then it selects best child
            But if player is minimizer then it selects worst child
        """
        n = node.visit
        Cp = self.Cp
        bestchild = None
        bestscore = -1

        for child in node.children:
            wj = child.wins if isMaximizer else child.loss
            nj = child.visit
            if nj == 0 or n == 0:
                uct = np.inf
            else:
                uct = wj / nj + Cp * np.sqrt(np.log(n) / nj)

            if uct > bestscore:
                bestscore = uct
                bestchild = child
        return bestchild

    def Expand(self, node, game_state):
        """
        Function: Adds a random child to node,
                  Updates game_state and node's untriedMoves
        Returns : The created child
        """
        untriedMoves = node.untriedMoves
        m = choice(untriedMoves)
        # node's untriedMoves updated here
        node.untriedMoves.remove(m)
        # game_state updated here
        game_state = HexBoard.makeMove(m, game_state)

        # Check the child, if it is already appended return it otherwise create that child
        child = node.getChild(game_state)
        if child is not None:
            return child

        child_type = 'MIN'
        if node.type == 'MIN':
            child_type = 'MAX'

        child = Node(node_type=child_type, board_state=game_state.board, parent=node)

        # node's children updated here
        node.children.append(child)

        return child
