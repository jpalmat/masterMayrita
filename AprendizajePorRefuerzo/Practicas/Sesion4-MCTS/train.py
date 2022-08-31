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

"""Class to train the Neural Network."""
import numpy as np
import torch

from config import CFG
from mcts import MonteCarloTreeSearch, TreeNode
from neural_net import NeuralNetworkWrapper
from copy import deepcopy


class Train(object):
    """Class with functions to train the Neural Network using MCTS."""

    def __init__(self, game, model):
        self.game = game
        self.model = model
        self.eval_model = NeuralNetworkWrapper(game)

    def start(self):
        """Main training loop."""
        for i in range(CFG.num_iterations):
            print("Iteration", i + 1)

            training_data = []  # list to store self play states, pis and vs
            for j in range(CFG.num_games):
                print("Start Training Self-Play Game", j + 1)
                game = self.game.clone()  # Create a fresh clone for each game.
                self.play_game(game, training_data)

            # Save the current neural network model.
            torch.save(self.model.model.state_dict(), "tic_tac_toe.pth")

            # Load the recently saved model into the evaluator network.
            self.eval_model.model.load_state_dict(self.model.model.state_dict())

            # Train the network using self play values.
            self.model.train(training_data)

            # Initialize MonteCarloTreeSearch objects for both networks.
            current_mcts = MonteCarloTreeSearch(self.model)
            eval_mcts = MonteCarloTreeSearch(self.eval_model)

            evaluator = Evaluate(current_mcts=current_mcts, eval_mcts=eval_mcts, game=self.game)
            wins, losses = evaluator.evaluate()

            print(f"wins: {wins} | losses: {losses}")

            if CFG.num_games == 0:
                win_rate = 0
            else:
                try:
                    win_rate = wins / (wins + losses)
                except:
                    win_rate = 0
            print("win rate:", win_rate)

            if win_rate > CFG.eval_win_rate:
                # Save current model as the best model.
                print("New model saved as best model.")
                torch.save(self.model.model.state_dict(), "tic_tac_toe_best.pth")
            else:
                print("New model discarded and previous model loaded.")
                # Discard current model and use previous best model.
                try:
                    self.model.model.load_state_dict(torch.load("tic_tac_toe_best.pth"))
                except:
                    # If the best model is not available yet, load the last one
                    self.model.model.load_state_dict(torch.load("tic_tac_toe.pth"))

    def play_game(self, game, training_data):
        """Loop for each self-play game.

        Runs MCTS for each game state and plays a move based on the MCTS output.
        Stops when the game is over and prints out a winner.

        Args:
            game: An object containing the game state.
            training_data: A list to store self play states, pis and vs.
        """
        mcts = MonteCarloTreeSearch(self.model)

        game_over = False
        value = 0
        self_play_data = []
        count = 0

        node = TreeNode()

        # Keep playing until the game is in a terminal state.
        while not game_over:
            # MCTS simulations to get the best child node.
            if count < CFG.temp_thresh:
                best_child = mcts.search(game, node, CFG.temp_init)
            else:
                best_child = mcts.search(game, node, CFG.temp_final)

            # Store state, prob and v for training.
            self_play_data.append([deepcopy(game.state), deepcopy(best_child.parent.child_psas), 0])

            action = best_child.action
            game.play_action(action)  # Play the child node's action.
            count += 1

            game_over, value = game.check_game_over(game.current_player)

            best_child.parent = None
            node = best_child  # Make the child node the root node.

        # Update v as the value of the game result.
        for game_state in self_play_data:
            value = -value
            game_state[2] = value
            
            state = game_state[0]
            psa_vector = np.reshape(game_state[1], (game.row, game.column))

            # Augment data by rotating and flipping the game state.
            for i in range(4):
                training_data.append([
                    np.rot90(state, i),
                    np.rot90(psa_vector, i).flatten(),
                    game_state[2]])
                training_data.append([
                    np.fliplr(np.rot90(state, i)),
                    np.fliplr(np.rot90(psa_vector, i)).flatten(),
                    game_state[2]])


class Evaluate(object):
    """Represents the Policy and Value Resnet.

    Attributes:
        current_mcts: An object for the current network's MCTS.
        eval_mcts: An object for the evaluation network's MCTS.
        game: An object containing the game state.
    """

    def __init__(self, current_mcts, eval_mcts, game):
        """Initializes Evaluate with the both network's MCTS and game state."""
        self.current_mcts = current_mcts
        self.eval_mcts = eval_mcts
        self.game = game

    def evaluate(self):
        """Play self-play games between the two networks and record game stats.

        Returns:
            Wins and losses count from the perspective of the current network.
        """
        wins = 0
        losses = 0

        # Self-play loop
        for i in range(CFG.num_eval_games):
            print("Start Evaluation Self-Play Game:", i, "\n")

            game = self.game.clone()  # Create a fresh clone for each game.
            game_over = False
            value = 0
            node = TreeNode()

            player = game.current_player

            # Keep playing until the game is in a terminal state.
            while not game_over:
                # MCTS simulations to get the best child node.
                # If player_to_eval is 1 play using the current network
                # Else play using the evaluation network.
                if game.current_player == 1:
                    best_child = self.current_mcts.search(game, node,
                                                          CFG.temp_final)
                else:
                    best_child = self.eval_mcts.search(game, node,
                                                       CFG.temp_final)

                action = best_child.action
                game.play_action(action)  # Play the child node's action.

                game.print_board()

                game_over, value = game.check_game_over(player)

                best_child.parent = None
                node = best_child  # Make the child node the root node.

            if value == 1:
                print("win")
                wins += 1
            elif value == -1:
                print("loss")
                losses += 1
            else:
                print("draw")
            print("\n")

        return wins, losses
