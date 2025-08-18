import sys
import chess
from engine.chess_env import ChessEnv
from engine.agent import Agent

def main():
    print("[*] ai-chess-ganglion CLI — you play White by default. Enter moves in UCI (e2e4).")
    env = ChessEnv()
    ai = Agent(weights_path=None, temperature=0.9)

    while not env.is_game_over():
        print(env.board)
        mv = input("Your move (uci): ").strip()
        if mv.lower() in ("quit","exit"):
            print("Goodbye.")
            return
        move = env.move_from_uci(mv)
        if move is None:
            print("Illegal or malformed; try again.")
            continue
        env.push(move)
        if env.is_game_over():
            break
        # AI move
        am = ai.select_move(env.board)
        print(f"[AI] {am.uci()}")
        env.push(am)

    print(env.board)
    print("Game over:", env.result())

if __name__ == "__main__":
    main()
