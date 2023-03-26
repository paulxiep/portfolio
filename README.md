# Portfolio

Even though this repository is called 'portfolio', it has evolved to be more like a playground for my experiments.

Skills aimed to demonstrate/practice with each project are:

    1. catan: Advanced object-oriented programming and software design, practical AI programming.
    2. cities_skylines_transport_ridership: Basic data visualization using Seaborn.
    3. duck_detection: Basic object detection and json data manipulation.
    4. geospatial: Geospatial Data Science, still only a practice ground. I have yet to find a dataset I like to work with and make something fruitful out of it.
    5. 7 wonders: Same as catan but with focus on reinforcement learning.

Currently the most developed project is 'Catan'. The rest are 1-day projects (2, 3) or practice range (4). 7 wonders is still in very early stage.

Information about Catan development phases can be found below,
and more info can be found on the project's own readme.

### Catan
(copied from Catan subfolder readme)

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