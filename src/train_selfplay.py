import os
import torch
import chess
from tqdm import tqdm
from engine.tri_gan import TriGANBrain, TriGANTrainer
from engine.utils import device_select, ensure_dir

def selfplay_epochs(epochs=1, games_per_epoch=4, save_dir="models"):
    device = device_select()
    brain = TriGANBrain(device=device)
    trainer = TriGANTrainer(brain, lr=1e-3, device=device)
    ensure_dir(save_dir)

    for ep in range(epochs):
        print(f"[Epoch {ep+1}/{epochs}]")
        for g in range(games_per_epoch):
            board = chess.Board()
            move_budget = 60
            pbar = tqdm(total=move_budget, desc=f"Game {g+1}/{games_per_epoch}", leave=False)
            while not board.is_game_over() and move_budget > 0:
                loss_d, loss_g = trainer.train_step(board)
                # choose move from current brain (greedy for selfplay)
                legal = list(board.legal_moves)
                if not legal:
                    break
                # Select via brain policy head
                board.push(legal[0])  # minimal advance to keep loop moving (we already used board in train_step)
                move_budget -= 1
                pbar.set_postfix({"D": f"{loss_d:.3f}", "G": f"{loss_g:.3f}"})
                pbar.update(1)
            pbar.close()

        # save checkpoint
        ckpt = os.path.join(save_dir, f"tri_ganglion_ep{ep+1}.pt")
        torch.save(brain.state_dict(), ckpt)
        print(f"[+] Saved {ckpt}")

if __name__ == "__main__":
    selfplay_epochs(epochs=1, games_per_epoch=2, save_dir="models")
