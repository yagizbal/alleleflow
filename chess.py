import numpy as np


def create_chessboard(queen_positions):
    chessboard = np.zeros((8, 8), dtype=int)
    for pos in queen_positions:
        row = pos // 8  #row number
        col = pos % 8  #column number
        chessboard[row][col] = 1
    return chessboard

def fitness(chessboard):
    size = 8
    total_pairs = size * (size - 1) // 2

    # Get positions of all queens
    queens = np.array(np.where(chessboard == 1)).T
    board_to_return = np.zeros((8, 8), dtype=int)

    # Iterate over each queen
    for queen in queens:
        x, y = queen
        # Reset number of attacking pairs for the current queen
        attacking_pairs = 0  

        # Check horizontal and vertical
        attacking_pairs += np.count_nonzero(chessboard[x, :]) - 1  # row
        attacking_pairs += np.count_nonzero(chessboard[:, y]) - 1  # column

        # Check diagonals
        diag_tl_br = np.diag(chessboard, y - x)
        diag_bl_tr = np.diag(np.fliplr(chessboard), (size - y - 1) - x)

        if np.count_nonzero(diag_tl_br) > 1:
            attacking_pairs += 1

        if np.count_nonzero(diag_bl_tr) > 1:
            attacking_pairs += 1

        board_to_return[x, y] = attacking_pairs

    # Compute the total number of attacking pairs
    total_attacking_pairs = np.sum(board_to_return)
    non_attacking_pairs = total_pairs - total_attacking_pairs  # Compute the fitness

    return non_attacking_pairs, board_to_return