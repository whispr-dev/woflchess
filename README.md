ai-chess-ganglion/README.md
markdown
Copy
Edit
# AI Chess Ganglion — Triadic-GAN CNN+RNN+LSTM over a Cellular Automaton Bus

This project is an experimental chess engine whose "brain" is:
- Three generators: **CNN**, **RNN (GRU)**, and **LSTM**.
- A single **clocked ganglion** that advances a **toroidal cellular automaton (CA)** bus each tick.
- The CA bus uses a **shift-register** wrap on selected rows; multi-channel state supports **signal bleed (crosstalk)**.
- A **discriminator** adversarially trains the generators to produce oracle-like moves (Stockfish if available, else a heuristic oracle).

You can **play it now** (legal moves, stochastic policy), and you can **train** the tri-GAN with self-play.

## Install

```bash
# Linux/macOS (WSL ok) — Python 3.10+
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
(Optional) To use Stockfish as the oracle:

Install a Stockfish binary and ensure it’s in your PATH.

pip install stockfish

Play the engine
bash
Copy
Edit
# from project root
python -m src.play_cli
Enter UCI moves like e2e4. The AI responds with its move.

Train (self-play, tri-GAN)
bash
Copy
Edit
python -m src.train_selfplay
Checkpoints are saved to models/tri_ganglion_ep*.pt. To play with a saved model:

bash
Copy
Edit
python -m src.play_cli  # edit Agent(weights_path="models/tri_ganglion_ep1.pt") if desired
Architecture Sketch
Encoding: Board → 18 planes (pieces, STM, castling, ep). Move history → tokens.

Generators:

CNNNet: spatial features from board planes.

RNNNet and LSTMNet: temporal features from history.

Ganglion + CA Bus:

Each latent is injected on separate CA channels.

Ganglion tick: shift one row (wrap-around), CA update (Moore neighborhood), then crosstalk/bleed.

Read a patch and fuse with the net latents using a bleed-weighted blend.

Policy Head:

Scores candidate legal moves via a simple interaction with fused latent.

Discriminator:

Judges (board, fused) embeddings; trained to output high for oracle-like distributions.

Tri-GAN Training:

D learns to distinguish oracle vs generated.

Generators learn to both fool D and prefer oracle’s move via a margin loss.

Notes
This is research-grade scaffolding: the engine plays legally now; strength improves as you iterate training, move-conditioned candidate features, and oracle quality.

The CA rule is simple (sigmoid(alpha*state + beta*neighbor_sum)) and the row-shift implements the shift-register feel. Both are easy to mutate/learn later.

Crosstalk is controlled by bleed in Ganglion(CABus(..., bleed=...)).

Tuning
bleed (signal crosstalk): increase to strengthen channel coupling.

Temperature in Agent(...): controls move stochasticity.

Replace _candidate_latents with richer move-conditioned features (e.g., encode move deltas, attacked squares, etc.) for stronger policies.

License
MIT (experimental use encouraged).

yaml
Copy
Edit

---

## 3) Run it

Windows PowerShell (or WSL), paths assumed you cloned to `D:\code\ai-chess-ganglion`:

**PowerShell**
```powershell
cd D:\code\ai-chess-ganglion
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.play_cli
WSL/Linux

bash
Copy
Edit
cd ~/code/ai-chess-ganglion
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.play_cli
4) What’s next (quick wins)
Make candidate features move-aware (encode from/to/promo and local board patch) before scoring.

Add a value head for win/draw/loss to stabilize training.

Swap CA rule to a learned convolution per channel (still toroidal) and let the ganglion set the step size per tick.

Replace the simple discriminator with a pairwise ranker over oracle vs generated pairs.