# 7 Wonders

**training log of 1st phase can be found [phase_1_training_log_deep_q.md](phase_1_training_log_deep_q.md)**

**The AI can be accessed via the streamlit [app](https://7-wonders-ai.streamlit.app/)**

---

#### Showcase (For recruiters looking for highlights)

1. Effect.from_dict() in [effects.py](effects.py): Single entry point method to generate subclasses of Effect from json input as obtained from v2_cards.json. Eliminates the need for repetitive code/specification by parsing the json which I obtained from other's repo to dataclass.
   
2. Custom distributed TensorFlow training in [dq_trainer.py](dq_trainer.py). With access to multiple GPUs, this would speed up training greatly.
---

7 Wonders is another (and 2nd) popular classic board game that I've decided to include in my repository. I started playing it in Mar 2023.

The project has reached the end of phase 1, with some planned features pushed to future updates.

This project used Gymnasium's library for proper reinforcement learning environment from the get-go, even if I haven't used most of its features yet.

The game elements consist of cards and boards, and the game play revolves around implementing card effects and card purchase. 

I'll see if I can use UI other than tkinter. That'll come in phase 2.

I've obtained card list for the 1st edition from 
[1st edition card list](https://github.com/joffrey-bion/seven-wonders/blob/main/sw-engine/src/main/resources/org/luxons/sevenwonders/engine/data/cards.json)

Then I edited the json to change it to 2nd edition based on information from 
[2nd edition changes information](https://boardgamegeek.com/thread/2491704/changes-old-edition-or-some-them)

Wonder json and change information were obtained from the same source.

1st phase: basic engine build, Deep-Q learning of card selection

    - Created base elements classes and Game class as a Gymnasium Env class.
    - Defined action and observation space.
    - Implemented Effect on Board
    - Implemented gameplay and DQPlayer, enabling simulation of gameplay and results
    - Implemented Deep-Q learning on card + action phase of turns.
    - Payment phase (if action is build) greedily minimizes cost to self.
    - DQTrainer for training Deep-Q model, with optional distributed training and Monte-Carlo loss update.

2nd phase: interface and app for human players to join the play, potentially Policy Gradient.

    - not to be done in short-term plan yet

Structure of the current project:

    - elements.py: Stage, Card, Wonder, Board, and Player class as dataclasses. DQPlayer is also here for now.
    - game.py: Game class as base game session class, will be subclassed into app as in Catan project
    - effects.py: Effect class and subclasses for implementing card/wonder effects
    - src/utils.py: squash_idle function to squash idle turns in memory, used for preparing training data
    - src/model.py: NN model inside the DQPlayer
    - src/constants.py: for storing various game constants and dicts
    - dq_trainer.py: Class for implementing training loop of DQPlayer.
