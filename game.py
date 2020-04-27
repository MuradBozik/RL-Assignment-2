import time

from hex_skeleton import HexBoard
from node import Node
from util import UTIL
from MCTS import MCTS
import os
import errno
import matplotlib.pyplot as plt
import numpy as np
from trueskill import rate_1vs1, Rating
import pickle


class InterfaceGame:
    def __init__(self, boardsize=None, experiment_number=None, starter=None):
        self.boardsize = boardsize
        self.experiment_number = experiment_number
        self.experiment_name = None
        self.starter = starter
        self.game = None
        self.rating_p1 = Rating()
        self.rating_p2 = Rating()
        self.all_ratings_p1 = [self.rating_p1]
        self.all_ratings_p2 = [self.rating_p2]
        if (boardsize is None) or (experiment_number is None) or (starter is None):
            self.askSettings()
        else:
            self.game = self.initGame(boardsize, starter)

    def askSettings(self):
        # BLUE = 1, # RED = 2
        self.boardsize = int(input("Enter the board size: "))
        answer = input("(E)xperiments or (M)anual: ").lower()
        if answer == 'e':
            self.experiment_number = int(input("Enter the amount experiments: "))
            self.experiment_name = input("Note: Experiment results will be saved inside the folder with this name\n"
                                         "Enter the name for experiment: ")
        self.starter = str(input("Start (R)ed or (B)lue: ")).lower()
        if self.starter == 'r':
            self.starter = 2
        elif self.starter == 'b':
            self.starter = 1
        else:
            print("Wrong input, terminating")
            exit()

        self.game = self.initGame(self.boardsize, self.starter)

        if answer == 'm':
            self.PlayervsComputer()
        else:
            firstplayer = input('(1):Mcts   (2):ID-TT alphabeta \n' +
                                'Select First Player 1, or 2...\n').strip()
            firstParams = self.getParams(firstplayer)
            secondplayer = input('(1):Mcts   (2):ID-TT alphabeta \n' +
                                 'Select Second Player 1, or 2...\n')
            secondParams = self.getParams(secondplayer)

            self.ComputervsComputer(firstplayer, firstParams, secondplayer, secondParams)

    @staticmethod
    def getParams(selection):
        if selection == '1':
            answer = input('Select following HyperParameter values CP, N, itermax, process time (seconds):\n' +
                           'Example: 0.8 100 25 20\n')
        else:
            answer = input('Select process time (seconds), maxdepth for Iterative Deepening (min = 1):\n' +
                           'Example: 20 3 \n')
        params = answer.split()
        hyperparams = []
        for i, val in enumerate(params):
            val = float(val) if i == 0 else int(val)
            hyperparams.append(val)
        return hyperparams

    @staticmethod
    def clearOutput():
        os.system('cls') if os.name == 'nt' else os.system('clear')

    def getReady(self, game):
        player = game.minimizer if (game.turn % 2) == 0 else game.maximizer
        playerName = '(Blue Player)' if player == 1 else '(Red Player)'

        while True:
            print(playerName + " Choose the coordinate to place your color: ")
            ans = input("Example: 2 c : ")
            move = self.getCoords(ans)
            if move[0] >= game.size or move[1] >= game.size:
                print('The coordinates you entered are out of bounds. Try again.')
            elif not game.isEmpty(move):
                print('Please select an empty coordinate. Try again.')
            else:
                break
        return move

    @staticmethod
    def getCoords(answer):
        coords = answer.split()
        coordinates = list()
        for coord in coords:
            coord = coord.strip()
        if len(coords) == 2:
            coordinates.append(int(coords[0]))
            coordinates.append(ord(coords[1]) - ord('a'))
        elif len(coords) == 1 and len(coords[0]) == 2:
            coordinates.append(int(coords[0][0]))
            coordinates.append(ord(coords[0][1]) - ord('a'))
        return tuple(coordinates)

    @staticmethod
    def initGame(b_size, starter):
        # starter: BLUE = 1, # RED = 2
        return HexBoard(b_size, HexBoard.getOppositeColor(starter), starter)

    # TODO: There is a problem in here doesn't plot correctly
    def updateRatings(self, color_blue=True):
        if self.starter == HexBoard.BLUE and color_blue:
            self.rating_p1, self.rating_p2 = rate_1vs1(self.rating_p1, self.rating_p2)
        else:
            self.rating_p2, self.rating_p1 = rate_1vs1(self.rating_p2, self.rating_p1)
        self.all_ratings_p1.append(self.rating_p1)
        self.all_ratings_p2.append(self.rating_p2)

    # TODO: Check if which player is which color
    def saveRatings(self):
        """Saves all ratings of players in player1 and player2 pickle file"""
        foldername = "{}/".format(self.experiment_name)
        if not os.path.exists(os.path.dirname(foldername)):
            try:
                os.makedirs(os.path.dirname(foldername))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        with open(foldername + 'ratings' + str(1), 'wb') as f:
            pickle.dump(self.all_ratings_p1, f)
        with open(foldername + 'ratings' + str(2), 'wb') as f:
            pickle.dump(self.all_ratings_p2, f)

    def plotElo(self):
        if self.starter == HexBoard.BLUE:
            y1, y2 = self.all_ratings_p1, self.all_ratings_p2
        else:
            y2, y1 = self.all_ratings_p1, self.all_ratings_p2
        x = [x for x in range(len(y1))]
        fig, ax = plt.subplots()
        ax.plot(x, y1, color='blue')
        ax.plot(x, y2, color='red')
        ax.set(xlabel='game (n)', ylabel='Elo (mu)')
        ax.legend(['BLUE', 'RED'])
        ax.grid()
        fig.savefig("{}/".format(self.experiment_name) + 'fig.png')
        plt.show()

    # TODO: Add stats usefull for us
    def matchStatistics(self, blue_wins, red_wins, totaltime, firstplayer, firstParams, secondplayer, secondParams):
        if self.starter == HexBoard.RED:
            P1 = 'RED'
            P2 = 'BLUE'
        else:
            P1 = 'BLUE'
            P2 = 'RED'
        if firstplayer == '1':
            P1_type = 'MCTS'
            params1 = f"HyperParameters CP:{firstParams[0]}, N:{firstParams[1]}, itermax:{firstParams[2]}, process " \
                      f"time (seconds):{firstParams[3]}\n"
        else:
            P1_type = 'IDTT'
            params1 = f"HyperParameters process time (seconds):{firstParams[0]}, maxdepth:{firstParams[1]}\n"

        if secondplayer == '1':
            P2_type = 'MCTS'
            params2 = f"HyperParameters CP:{secondParams[0]}, N:{secondParams[1]}, itermax:{secondParams[2]}, process " \
                      f"time (seconds):{secondParams[3]}\n"
        else:
            P2_type = 'IDTT'
            params2 = f"HyperParameters process time (seconds):{secondParams[0]}, maxdepth:{secondParams[1]}\n"

        with open("{}/".format(self.experiment_name) + "output.txt", "w") as f:
            f.write("------- Result of experiment -------\n")
            f.write(f"Board size: {self.boardsize}\n")
            f.write(f"Number of experiment: {self.experiment_number}\n")
            f.write(f'First Player was {P1} ({P1_type})!\n')
            f.write(params1)
            f.write(f'Second Player was {P2} ({P2_type})!\n')
            f.write(params2)
            f.write(f"Win count of Red: {red_wins}\n")
            f.write(f"Win count of Blue: {blue_wins}\n")
            f.write(f'Rating of {P1} Player: {self.rating_p1}\n')
            f.write(f'Rating of {P2} Player: {self.rating_p2}\n')
            f.write(f"Total time: {np.round(totaltime, 2)} secs\n")

    def ComputervsComputer(self, firstplayer, firstParams, secondplayer, secondParams):
        t0 = time.time()
        experiment_counter = self.experiment_number
        red_wins = 0
        blue_wins = 0

        IterativeDeepeningTranspositionTable = dict()  # We will store best moves for iterative deepening in here

        while experiment_counter != 0:
            game = self.initGame(self.boardsize, self.starter)
            self.clearOutput()
            game.print()

            node = Node(node_type='MIN', board_state=game.board, parent=None)  # initialize node
            util = UTIL(infinity=np.inf, maximizer=game.maximizer, minimizer=game.minimizer)  # initialize util class
            params = {'game': game,
                      'node': node
                      }

            def MCTS_Player(hyperparameters, isMaximizer):
                # First Computer's turn (MCTS)
                mcts_agent = MCTS(game=game, cp=hyperparameters[0], n=hyperparameters[1])  # initialize mcts agent
                move, params['node'] = mcts_agent.search(params['node'], hyperparameters[2], hyperparameters[3],
                                                         isMaximizer)
                params['game'] = HexBoard.makeMove(move, params['game'])
                params['node'] = util.updateNode(params['node'], params['game'])

            def IDTT_Player(hyperparameters, isMaximizer):
                # iterative deepening with 4 depth-Dijkstra
                boardState = params['node'].board.copy()
                try:
                    move = IterativeDeepeningTranspositionTable[boardState.tobytes()]
                except KeyError:
                    best_value = util.iterativeDeepening(params['node'], isMaximizer, hyperparameters[0],
                                                         hyperparameters[1])
                    move = util.getBestMove(params['node'], best_value)
                    IterativeDeepeningTranspositionTable[boardState.tobytes()] = move
                params['game'] = HexBoard.makeMove(move, params['game'])
                params['node'] = util.updateNode(params['node'], params['game'])

            FirstPlayer = MCTS_Player if firstplayer == '1' else IDTT_Player
            SecondPlayer = MCTS_Player if secondplayer == '1' else IDTT_Player

            while not game.isGameOver():
                if (game.turn % 2) == 0:
                    # This is minimizer
                    p1 = 'Red' if self.starter == 2 else 'Blue'
                    print(f"First ({p1}) Player is thinking! Remaining Experiment: {experiment_counter}")
                    FirstPlayer(firstParams, isMaximizer=False)
                else:
                    # This player is maximizer
                    p2 = 'Blue' if self.starter == 2 else 'Red'
                    print(f"Second ({p2}) Player is thinking! Remaining Experiment: {experiment_counter}")
                    SecondPlayer(secondParams, isMaximizer=True)
                self.clearOutput()
                game.print()

                if game.isGameOver():
                    if game.checkWin(game.BLUE):
                        self.updateRatings()
                        blue_wins += 1
                        print("!!! Blue Player Won !!!")
                    elif game.checkWin(game.RED):
                        self.updateRatings(False)
                        red_wins += 1
                        print("!!! Red Player Won !!!")

            experiment_counter -= 1
        t1 = time.time()
        totalTime = t1 - t0
        self.saveRatings()
        self.plotElo()
        self.matchStatistics(blue_wins, red_wins, totalTime, firstplayer, firstParams, secondplayer, secondParams)

    def PlayervsComputer(self):
        game = self.game
        self.clearOutput()
        game.print()

        node = Node(node_type='MIN', board_state=game.board, parent=None)  # 1 - initialize node
        mcts_agent = MCTS(game=game, cp=0.8, n=100)  # 3 - initialize mcts agent
        util = UTIL(infinity=np.inf, maximizer=game.maximizer, minimizer=game.minimizer)  # 4 - initialize util class

        while not game.isGameOver():
            if (game.turn % 2) == 0:
                move = self.getReady(game)
                game = HexBoard.makeMove(move, game)
                node = util.updateNode(node, game)
            else:
                print("Computer is thinking!!!")
                itermax = 100  # maximum iteration for search
                move, node = mcts_agent.search(node, itermax, delta=10, isMaximizer=True)
                print(f'best move: {move}')
                game = HexBoard.makeMove(move, game)
                node = util.updateNode(node, game)
            self.clearOutput()
            game.print()

            if game.isGameOver():
                if game.checkWin(game.BLUE):
                    print("!!! Blue Player Won !!!")
                elif game.checkWin(game.RED):
                    print("!!! Red Player Won !!!")


if __name__ == '__main__':
    interface = InterfaceGame()
