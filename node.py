import numpy as np
import uuid
from hex_skeleton import HexBoard


class Node:
    def __init__(self, node_type, board_state, parent):
        self.id = str(uuid.uuid4())  # to make it unique
        self.name = self.id[-3:]  # to make it simple during visualizations (for the test purposes)
        self.type = node_type  # for visualization
        self.board = board_state  # for MCTS

        untriedMoves = HexBoard.getFreeMoves(board_state)
        self.untriedMoves = untriedMoves  # for MCTS
        self.children = []  # for MCTS
        if parent is not None:
            self.parents = [parent]
            self.parent_type = parent.type  # for visualization
        else:
            self.parents = []  # for MCTS
            self.parent_type = None  # for visualization

        self.searched = False
        self.value = None
        self.visit = 0  # for MCTS UCT-Selection and finding the best move
        self.wins = np.inf  # for MCTS UCT-Selection
        self.loss = np.inf

    def getChild(self, game):
        """ Returns the child according to board state """
        for child in self.children:
            if np.array_equal(child.board, game.board):
                return child
        return None

    def getParent(self, key):
        for parent in self.parents:
            if parent.board.tobytes() == key:
                return parent
        return None
