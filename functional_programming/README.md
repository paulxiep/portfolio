## Functional Programming

A prototype for a meta-tool for Functional Programming in Python.

### Content

The only real content of this project is ```make_functional()``` in [utils/functional_tools.py](utils/functional_tools.py).

The rest of the files are only to demonstrate how to use ```make_functional()```.

### Motivation

Some backstory into what I used to do. 

- I used to subclass some Python object type, like ```list``` for example.
- I subclass these object classes to override some methods I wanted to use, and make them return self.
  - To make them functional methods, and allow method chaining.

### What is does

Which brings us to this prototype project.

- With this new ```make_functional()``` prototype,
  - I don't have to manually subclass each object individually.
  - Nor do I have to manually override each single method individually.
  - Instead, I can throw any object into it, and it'll override **all** methods and make them functional automatically.
  - I can also add normal functions to ```additional_methods``` that'll be callable as class methods for the new object.

### How it does so

- This prototype is different from subclassing.
  - Instead, it creates a new wrapper class that sets the input (original) object as its ```_content``` attribute.
  - And for most methods of the original object, this wrapper class will 'inherit' them, 
    - and apply them to ```_content```, the stored original object.
    - and also return ```self``` each time it does so, allowing for methods to be chained.
      - But it can also be the case the original method originally returns something, 
        - in which case it creates a new wrapper class with the return value as ```_content```
          - The return value could be of the same object type or a different class entirely.
    - Exceptions that will not be inherited are the private methods (those starting with ```__```).

### Even more

- ```return_content()``` method is also created to return the current ```_content```, 
  - returning to the original class (or a different class if the object has been transformed)
- ```freeze()``` and ```restore()``` are also added to 'freeze' a state of ```_content```
  - Allowing method chaining to transform it, 
    - and then to return to the frozen content later within the same method chain.

### Some notes on origin of this project

- This project was originally written as part of an employment test.
- I'll be writing new contents to demonstrate the use of ```make_functional()``` later when I have time.