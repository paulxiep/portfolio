# Portfolio

Even though this repository is called 'portfolio', it has evolved to be more like a playground for my experiments.

Currently the most developed project is 'Catan'. 

Information about development phases can be found below, 
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

3rd phase: yet to come, will first focus on improving rule-based AI, followed by scaling factors tuning with simulation data.