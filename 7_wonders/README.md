## 7 Wonders

7 Wonders is another (and 2nd) popular classic board game that I've decided to include in my repository. I started playing it in Mar 2023.

Currently the project is in (very early stage of) Phase 1. I'm building the basic game engine.

The game elements consist of cards and boards, then the game play will involve around implementing card effects and card purchase. 

I'll see if I can use UI other than tkinter.

I've obtained card list for the 1st edition from 
[1st edition card list](https://github.com/joffrey-bion/seven-wonders/blob/main/sw-engine/src/main/resources/org/luxons/sevenwonders/engine/data/cards.json)

Then I edited the json to change it to 2nd edition based on information from 
[2nd edition changes information](https://boardgamegeek.com/thread/2491704/changes-old-edition-or-some-them)

1st phase: basic engine build (more to come)

    - Created Card class and Game class, Game class can initialize card deck based on v2_cards.json

Structure of the current project (again, it's in very early stage):

    - card.py: Card class as a dataclass
    - game.py: Game class as base game session class, will be subclassed into app and simulation as in Catan project