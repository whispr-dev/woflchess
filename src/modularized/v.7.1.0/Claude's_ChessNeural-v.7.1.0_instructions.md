

included in Claude's_ChessNeural-v.5.0.0

root
|
|-[Claude's_ChessNeural-v.5.0.0.exe]
|-[Claude's_ChessNeural-v.5.0.0_instructions.md]
\-[neural_weights.json] (generated from training runs)


----------------------------------------------

'Claude', the chess brain has:

- a GAN, consisting of 3 NN - an RNN, a CNN and an LSTM.
- they are synchronbized by a ganglion clocked by `neural clock`.
- the ganglion interfaces with the NN via a CA elementary rule90 based interface layer, with SIPO wrap around edge handling.
- the interface/NN layer crossover features crosstyalk between the differentt subsystems so as to increase speeds and creativity.

this brai n as an overall is the magical mystery and weird to Claude's chess madness!


----------------------------------------------

Use play to start a game against the AI
- enter move as either e.g. `b1c3` or `b1 c3`
- castling is done via `O-O` for kingside and O-O-O queenside.
- if no piece is chosen from q/r/k/b when a pawn reaches the back rank it will default to promotion to queen.
- to resign type `resign` at prompt and the game will end with your loss.
- most everything else is auto-triggered by attempting to move to the relevant ssquare on the board with appropriate piece.
- to exit program at end or any prompt, type `quit` [this also applies to training promopts]
- if quit is unavailable, [ctrl] + [c] will do std. quit program, and training data will be saved if possible.

-----------------------------------------------
Training::

Weight persistence:

Saves neural network weights to a JSON file
Loads weights on startup
Periodically saves during training


Training commands:

train time XX - Train for XX minutes
train games XX - Train for XX number of games
Training results are displayed after completion

sm0l amnts of rng added to training si experimentation occurs and he trires new strategies.

------------------------------------------------

To use it:

Start the program
Use train time 15 to train for 15 minutes
Use train games 100 to train for 100 games
The AI will get progressively better as it trains

The weights are saved to "neural_weights.json" and will be loaded automatically when you restart the program.

------------------------------------------------

v.7.0.0 also comes with human vs. human option - either locally or via internet.
- to play locally select: play `local`
- to play via internet select: `server start` [to launch a warp server which a human vs. human game can then be launched across.]
- to kill the net connection: `server stop`

------------------------------------------------


The Unicode chess pieces look beautiful on the terminal display. got a fully functioning server with clean start/stop, plus a working chess board display.
The rendering of the board is particularly nice - you can clearly see:

- All pieces in their correct starting positions
- Proper Unicode chess characters (♔♕♖♗♘♙ for white, ♚♛♜♝♞♟ for black)
- Clean grid with file (a-h) and rank (1-8) labels
- Clear dots (·) for empty squares
- Current turn indicator showing "White"

-------------------------------------------------

