# MIT License
#
# Copyright (c) 2018 Blanyal D'Souza
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

# Adapted from https://github.com/blanyal/alpha-zero

from config import CFG
from game import TicTacToeGame
from mcts import MonteCarloTreeSearch, TreeNode
from neural_net import NeuralNetworkWrapper


class HumanPlay(object):
    """Class with functions for a Human vs an AI game.

    Attributes:
        game: An object containing the game state.
        net: An object containing the neural network.
    """

    def __init__(self, game, net):
        """Initializes HumanPlay with the board state and neural network."""
        self.game = game
        self.net = net

    def play(self):
        """Function to play a game vs the AI."""
        print("Start Human vs AI\n")

        mcts = MonteCarloTreeSearch(self.net)
        game = self.game.clone()  # Create a fresh clone for each game.
        game_over = False
        value = 0
        node = TreeNode()

        print("Enter your move in the form: row, column. Eg: 1,1")
        go_first = input("Do you want to go first: y/n?")

        if go_first.lower().strip() == 'y':
            print("You play as X")
            human_value = 1
            game.print_board()
        else:
            print("You play as O")
            human_value = -1

        # Keep playing until the game is in a terminal state.
        while not game_over:
            # MCTS simulations to get the best child node.
            # If player_to_eval is 1 play as the Human.
            # Else play as the AI.
            if game.current_player == human_value:
                action = input("Enter your move: ")
                if isinstance(action, str):
                    action = [int(n, 10) for n in action.split(",")]
                    action = (1, action[0], action[1])

                best_child = TreeNode()
                best_child.action = action
            else:
                best_child = mcts.search(game, node, CFG.temp_final)

            action = best_child.action
            game.play_action(action)  # Play the child node's action.

            game.print_board()

            game_over, value = game.check_game_over(game.current_player)

            best_child.parent = None
            node = best_child  # Make the child node the root node.

        if value == human_value * game.current_player:
            print("You won!")
        elif value == -human_value * game.current_player:
            print("You lost.")
        else:
            print("Draw Match")
        print("\n")


if __name__ == '__main__':
    # Initialize the game object with the chosen game.
    game = TicTacToeGame()
    net = NeuralNetworkWrapper(game)
    net.load_best_model()

    human_play = HumanPlay(game, net)
    human_play.play()
