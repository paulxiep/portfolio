## Catan

---
#### Showcase (For recruiters looking for highlights)

 - [catan_units/hex_geometry.py](catan_units/hex_geometry.py) (can be visualized with catan_app.py): Not using the standard cube or trapezoidal coordinate system for hex coordinates, I daresay my system is more intuitive to humans and while code is not easy to read and debug, it'll scale to any hex board size.

---

Catan or Settlers of Catan is a popular classic board game. I only just in February 2023 started playing it. I found some academic papers on Catan strategy and algorithm and I wanted to see how well I can make one too.

Currently the project is in Phase 3. As it is in the early part of a phase, the code won't be clean. This is the same as when forking a new branch for new feature implementation, but this is a portfolio, not production, so I just develop in the main branch so you can see new updates sooner.

1st phase: basic engine build

    - Underlying engine hex geometry, topographic connections, and generic hex board
    - CatanHexBoard with CatanHex, CatanEdge, CatanCorner
    - tkinter Canvas drawing of the board
    - tkinter app for play session
    - very basic AI that can play the game
    - written in 4 days, but was a spaghetti mess
    - still accessible in old_app.py, though I'm unsure if I've since made some dependencies incompatible

2nd phase: pure ML AI training and experimentation

    - Refactored all the engine and class structures
    - Separated CatanSession and CatanApp, separated AI from the app
    - Added CatanSimulation as another subclass of CatanSession, focused on AI game simulation, without interactive UI
    - Added all necessary loggings of game state data
    - log_data_extractor to convert game log to ML training data
    - catboost_training as an example of training script. The actual training was done in jupyter notebook for better manual interaction.
    - Experimented with pure ML AI for 3 sub-ais, focusing on road placement ai
    - Pure ML AI of this stage was implemented by combining game state dataframe with eligible potential move and predicting what move would most likely result in a win.
    - Concluded that pure ML AI has potential, but to generate quality data efficiently, a better basis (rule-based) AI is needed
    - Only random AI (one aspect at a time, for example, if I want to train a road AI, I put all other AIs as basis, and the road as random), only the random moves that managed to win within 25 turns were used as data, the other games were discarded. Since the basis AI is still weak, it results in too many discarded games.
    - Also since I've made some AIs score-based instead of hard if-else logic, I also introduced many scaling factors that can be tuned. The next ML training will focus on tuning these scaling factors for the (new) basis AIs.

3rd phase: will first focus on improving rule-based AI, followed by scaling factors tuning with simulation data.

    - 3 sub ais now have basis_v2 implementation. More will come later in the phase
    - scaling factor tuning code was implemented and ran for 4 settlement weight scaling factors.
    - longest road has been implemented, with flaws. It will be perfected later in the pipeline.
    - AI with basis_v2 vs 3 basis AI was able to win 50% of the time (baseline is 25% in a 4 player game)
    - basis_v2 AI with new scalings vs 3 basis_v2 with old scalings was able to win 33% of the time.

Current project structure

    - hex_geometry.py: contains Geometry base class, and then Edge, Corner, Hex, and HexBoard class for generic purpose hex board. The main purpose of this layer is the implementation of edge-corner-hex connection in hexagonal space.
    - catan_hex.py: contains CatanEdge, CatanCorner, CatanHex, and CatanHexBoard, which adds Catan data such as resource, number, settlement, etc. to base class.
    - catan_player.py: contains necessary player information such as number of tokens left, resources, development cards, etc. It is implemented as a dataclass.
    - catan_ai.py: is implemented as a subclass of CatanPlayer to add 'ai personality' and ai call for automatic play. The CatanAI itself is composed of many CatanSubAI to be called for different decisions.
    - ai_utils.py: contain some utility functions that are called by CatanSubAI
    - catan_canvas.py: contains tkinter Canvas drawing implementation of CatanHexBoard
    - catan_session.py: is the main game engine that stores necessary data and implements game mechanisms. It is not used on its own but will be subclassed.
    - catan_app.py: is the main interactive app and a subclass of both CatanSession and Tk.
    - catan_simulation.py: is another subclass of catan session that'll play the whole AI game with one click and log the necessary information for AI training.
    - scalings_tuning.py: for tuning scaling factors introduced at the end of phase 2    

The following files are currently not maintained

    - simulate.py: entry point for catan_simulation
    - log_data_extractor.py: is used to convert game logs into data frames for ML training
    - create_ai_data.py: contains some functions called by log_data_extractor to generate training data specific to each CatanSubAI
    - catboost_train.py: is simply a demo code. I do the actual training in jupyter notebook.
    - old_app.py: the app from the 1st stage
    - old_canvas.py: the canvas for the 1st stage

Unimplemented game features

    - manual discard on a 7-dice roll (currently random)
    - manual selection of whom to rob on a robber move (currently random)
    - longest road calculation is still wrong in certain conditions

<developed in Python 3.8>
