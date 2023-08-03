## 24+ Solver

This is a simple 1-day project that find solutions for the number game, 
in which you're given a few integer numbers and told to find ways to arrive at a target integer.

All inner workings are described in functions and classes. All operations make use of generators, and are executed sequentially.

1. The engine uses a number of custom Arithmetic classes, 
which override right shift operator (>>) to chain operations.
2. Each Arithmetic subclass contains value and repr as attribute,
such that both the current value and the representation used to arrive at current value are stored.
3. None of the Arithmetic subclasses is meant to be initialized directly,
instead, use get method on Number and Operator subclasses.

Entry point is ```main.py``` and the main functions are ```search_range()``` and ```search_value()```