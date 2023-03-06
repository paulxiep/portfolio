## Catan

Catan or Settlers of Catan is a popular classic board game. I only just in February 2023 started playing it. I found some academic papers on Catan strategy and algorithm and I wanted to see how well I can make one too.

I've written the underlying engine of generic hex board and catan hex board with hexes, edges, and corners, and their topographic information. Then I wrote a basic visualization an interactive UI in tkinter of the board and the app for play session.

A very basic AI with no forward thinking was then implemented.

So all the engine, the app, and basic AI were written in 4 days, albeit as a spaghetti mess.

With the 2nd major update I've refactored all the code and class structures. I broke the interactive app with it, but now I'm focusing on simulation session with AI players so I can generate data to train a basic reinforcement learning AI. The interactive app will be fixed in due time after some AI training is done. (but the working version with old code is still kept in old_app.py)

So now I have this kind of structure.

- hex_geometry.py: contains Geometry base class, and then Edge, Corner, Hex, and HexBoard class for generic purpose hex board. The main purpose of this layer is the implementation of edge-corner-hex connection in hexagonal space.
- catan_hex.py: contains CatanEdge, CatanCorner, CatanHex, and CatanHexBoard, which adds Catan data such as resource, number, settlement, etc. to base class.
- catan_player.py: contains necessary player information such as number of tokens left, resources, development cards, etc. It is implemented as a dataclass.
- catan_ai.py: is implemented as a subclass of CatanPlayer to add 'ai personality' and ai call for automatic play. The CatanAI itself is composed of many CatanSubAI to be called for different decisions.
- catan_canvas.py: contains tkinter Canvas drawing implementation of CatanHexBoard
- catan_session.py: is the main game engine that stores necessary data and implements game mechanisms. It is not used on its own but will be subclassed.
- catan_app.py: is the main interactive app and a subclass of both CatanSession and Tk.
- catan_simulation.py: is another subclass of catan session that'll play the whole AI game with one click and log the necessary information for AI training.

More comments and cleaning will be done after the main planned functionalities and engines are done.