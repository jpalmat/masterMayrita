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

from copy import deepcopy

import numpy as np


class TicTacToeGame(object):
    """Represents the game board and its logic.

    Attributes:
        row: An integer indicating the length of the board row.
        column: An integer indicating the length of the board column.
        current_player: An integer to keep track of the current player.
        state: A list which stores the game state in matrix form.
        action_size: An integer indicating the total number of board squares.
    """

    def __init__(self):
        """Initializes TicTacToeGame with the initial board state."""
        self.row = 3
        self.column = 3
        self.current_player = 1
        self.state = []
        self.action_size = self.row * self.column

        # Create a n x n matrix to represent the board
        for i in range(self.row):
            self.state.append([0 * j for j in range(self.column)])

        self.state = np.array(self.state)

    def clone(self):
        """Creates a deep clone of the game object.

        Returns:
            the cloned game object.
        """
        game_clone = TicTacToeGame()
        game_clone.state = deepcopy(self.state)
        game_clone.current_player = self.current_player
        return game_clone

    def play_action(self, action):
        """Plays an action on the game board.

        Args:
            action: A tuple in the form of (row, column).
        """
        x = action[1]
        y = action[2]

        self.state[x][y] = self.current_player
        self.current_player = -self.current_player

    def get_valid_moves(self, current_player):
        """Returns a list of moves along with their validity.

        Searches the board for zeros(0). 0 represents an empty square.

        Returns:
            A list containing moves in the form of (validity, row, column).
        """
        valid_moves = []

        for x in range(self.row):
            for y in range(self.column):
                if self.state[x][y] == 0:
                    valid_moves.append((1, x, y))
                else:
                    valid_moves.append((0, None, None))

        return np.array(valid_moves)

    def check_game_over(self, current_player):
        """Checks if the game is over and return a possible winner.

        There are 3 possible scenarios.
            a) The game is over and we have a winner.
            b) The game is over but it is a draw.
            c) The game is not over.

        Args:
            current_player: An integer representing the current player.

        Returns:
            A bool representing the game over state.
            An integer action value. (win: 1, loss: -1, draw: 0
        """

        player_a = current_player
        player_b = -current_player

        # Check for horizontal marks
        for x in range(self.row):
            player_a_count = 0
            player_b_count = 0
            for y in range(self.column):
                if self.state[x][y] == player_a:
                    player_a_count += 1
                elif self.state[x][y] == player_b:
                    player_b_count += 1
            if player_a_count == self.row:
                return True, 1
            elif player_b_count == self.row:
                return True, -1

        # Check for vertical marks
        for x in range(self.row):
            player_a_count = 0
            player_b_count = 0
            for y in range(self.column):
                if self.state[y][x] == player_a:
                    player_a_count += 1
                elif self.state[y][x] == player_b:
                    player_b_count += 1
            if player_a_count == self.row:
                return True, 1
            elif player_b_count == self.row:
                return True, -1

        # Check for major diagonal marks
        player_a_count = 0
        player_b_count = 0
        for x in range(self.row):
            if self.state[x][x] == player_a:
                player_a_count += 1
            elif self.state[x][x] == player_b:
                player_b_count += 1

        if player_a_count == self.row:
            return True, 1
        elif player_b_count == self.row:
            return True, -1

        # Check for minor diagonal marks
        player_a_count = 0
        player_b_count = 0
        for y in range(self.row - 1, -1, -1):
            x = 2 - y
            if self.state[x][y] == player_a:
                player_a_count += 1
            elif self.state[x][y] == player_b:
                player_b_count += 1

        if player_a_count == self.row:
            return True, 1
        elif player_b_count == self.row:
            return True, -1

        # There are still moves left so the game is not over
        valid_moves = self.get_valid_moves(current_player)

        for move in valid_moves:
            if move[0] is 1:
                return False, 0

        # If there are no moves left the game is over without a winner
        return True, 0

    def print_board(self):
        """Prints the board state."""
        print("   0    1    2")
        for x in range(self.row):
            print(x, end='')
            for y in range(self.column):
                if self.state[x][y] == 0:
                    print('  -  ', end='')
                elif self.state[x][y] == 1:
                    print('  X  ', end='')
                elif self.state[x][y] == -1:
                    print('  O  ', end='')
            print('\n')
        print('\n')
