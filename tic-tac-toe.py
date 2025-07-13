import tkinter as tk
from functools import partial
import math

class TicTacToeGUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Tic-Tac-Toe")
        self.board = [' '] * 9
        self.buttons = []
        self.game_over = False

        # Mode control
        self.ai_starts = False
        self.user_symbol = 'X'
        self.ai_symbol = 'O'
        self.current_symbol = 'X'

        self._build_ui()
        self.window.mainloop()

    def _build_ui(self):
        # Create 3x3 grid buttons
        frame = tk.Frame(self.window)
        frame.pack(padx=10, pady=10)
        for i in range(9):
            btn = tk.Button(frame, text=' ', font=('Arial', 24), width=4, height=2,
                            command=partial(self.on_click, i))
            btn.grid(row=i//3, column=i%3)
            self.buttons.append(btn)

        # Control buttons
        ctrl = tk.Frame(self.window)
        ctrl.pack(pady=5)
        tk.Button(ctrl, text="Player First",
                  command=partial(self.reset, False)).pack(side='left', padx=5)
        tk.Button(ctrl, text="Computer First",
                  command=partial(self.reset, True)).pack(side='left', padx=5)
        tk.Button(ctrl, text="Reset",
                  command=lambda: self.reset(self.ai_starts)).pack(side='left', padx=5)

    def reset(self, ai_starts):
        self.ai_starts = ai_starts
        # Assign markers
        if ai_starts:
            self.ai_symbol, self.user_symbol = 'X', 'O'
        else:
            self.user_symbol, self.ai_symbol = 'X', 'O'
        self.board = [' '] * 9
        self.current_symbol = 'X'
        self.game_over = False

        # Clear buttons
        for btn in self.buttons:
            btn.config(text=' ', state='normal')

        # If computer starts, make its move
        if self.current_symbol == self.ai_symbol:
            self._ai_move()

    def on_click(self, idx):
        if self.game_over:
            return
        if self.board[idx] != ' ':
            return
        if self.current_symbol != self.user_symbol:
            return

        # Human move
        self._make_move(idx, self.user_symbol)

        # If game continues, let AI play
        if not self.game_over and self.current_symbol == self.ai_symbol:
            self.window.after(200, self._ai_move)

    def _make_move(self, idx, symbol):
        self.board[idx] = symbol
        self.buttons[idx].config(text=symbol)
        winner = self._check_winner()
        if winner or all(cell!=' ' for cell in self.board):
            self._end_game(winner)
        else:
            # Switch turn
            self.current_symbol = 'O' if symbol == 'X' else 'X'

    def _end_game(self, winner):
        self.game_over = True
        msg = "Draw!" if not winner else f"{winner} wins!"
        popup = tk.Toplevel(self.window)
        tk.Label(popup, text=msg, font=('Arial', 18)).pack(pady=10)
        tk.Button(popup, text="OK", command=popup.destroy).pack(pady=5)

    def _check_winner(self):
        wins = [
            (0,1,2),(3,4,5),(6,7,8),
            (0,3,6),(1,4,7),(2,5,8),
            (0,4,8),(2,4,6)
        ]
        for a, b, c in wins:
            if self.board[a] == self.board[b] == self.board[c] != ' ':
                return self.board[a]
        return None

    def _ai_move(self):
        move = self._minimax(self.board[:], self.ai_symbol)['index']
        self._make_move(move, self.ai_symbol)

    def _minimax(self, board, player):
        winner = self._check_winner_board(board)
        if winner == self.ai_symbol:
            return {'score': 1}
        if winner == self.user_symbol:
            return {'score': -1}
        if all(cell!=' ' for cell in board):
            return {'score': 0}

        moves = []
        for i in range(9):
            if board[i] == ' ':
                board[i] = player
                score = self._minimax(board,
                    self.user_symbol if player == self.ai_symbol else self.ai_symbol
                )['score']
                moves.append({'index': i, 'score': score})
                board[i] = ' '

        # Choose best move for current player
        if player == self.ai_symbol:
            best = max(moves, key=lambda x: x['score'])
        else:
            best = min(moves, key=lambda x: x['score'])
        return best

    def _check_winner_board(self, b):
        wins = [
            (0,1,2),(3,4,5),(6,7,8),
            (0,3,6),(1,4,7),(2,5,8),
            (0,4,8),(2,4,6)
        ]
        for a, b_, c in wins:
            if b[a] == b[b_] == b[c] != ' ':
                return b[a]
        return None

if __name__ == "__main__":
    TicTacToeGUI()