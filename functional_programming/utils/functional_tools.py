import copy
import logging
from dataclasses import dataclass


def add_log_and_comment(func):
    def function_wrapper(*args, logs=None, log_level=logging.DEBUG, comment=None, **kwargs):
        if logs is not None:
            logging.log(log_level, logs)
        return func(*args, **kwargs)

    return function_wrapper


def add_print(func):
    def function_wrapper(*args, print_value=False, **kwargs):
        out = func(*args, **kwargs)
        if print_value:
            if print_value in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]:
                logging.log(print_value, out._content)
            else:
                print(out._content)
        return out

    return function_wrapper

def delegate(object_class, additional_methods, meta_methods):
    '''
    Used to wrap object methods to return self as default,
    conforming to functional programming style

    also add functions in additional_methods as methods
    meta_methods are for functions that apply to metaclass, not to _content
    '''

    def decorate(cls):

        def delegate_method(method):
            '''
            new decorated class will inherit _content class methods, and apply methods to _content
            '''

            @add_log_and_comment
            @add_print
            def return_self_as_default(self, *args, **kwargs):
                args = list(map(lambda x: x if x.__class__.__name__ != 'FunctionalObject' else x._content, args))
                out = getattr(getattr(self, '_content'), method)(*args, **kwargs)
                if out is None:
                    return self
                else:
                    return make_functional(out, additional_methods, meta_methods,
                                           frozen_content=self._frozen_content)

            return return_self_as_default

        def function_to_method(func):
            '''
            add functions, for example the functions from preprocess.py, as methods
            '''

            @add_log_and_comment
            @add_print
            def f_to_m(self, *args, **kwargs):
                self._content = func(self._content, *args, **kwargs)
                return self

            return f_to_m

        def meta_function_to_method(func):
            '''
            add meta-level functions as methods
            '''

            @add_log_and_comment
            @add_print
            def mf_to_m(self, *args, **kwargs):
                self._content = func(self, *args, **kwargs)
                return self

            return mf_to_m

        def freeze():
            '''
            allow one instance of _content freeze
            '''

            @add_log_and_comment
            @add_print
            def freeze_content(self, freeze_key=None):
                self._frozen_content[freeze_key] = copy.deepcopy(self._content)
                return self

            return freeze_content

        def restore():
            '''
            restore frozen _content
            and swap current _content into frozen
            '''

            @add_log_and_comment
            @add_print
            def restore_content(self, restore_key=None):
                try:
                    return make_functional(copy.deepcopy(self._frozen_content[restore_key]),
                                           additional_methods, meta_methods,
                                           frozen_content=copy.deepcopy({**self._frozen_content,
                                                                         restore_key: self._content}))
                except Exception as e:
                    if isinstance(e, KeyError):
                        raise Exception(f"Can't restore key '{restore_key}', no value stored")
                    else:
                        raise e

            return restore_content

        def return_content():
            '''
            exit functional wrapper class and return current _content
            '''

            @add_log_and_comment
            def content_return(self):
                return self._content

            return content_return

        def pipe():
            @add_log_and_comment
            @add_print
            def pipe_func(self, func, *args, input_args_name=None, **kwargs):
                if input_args_name is None:
                    out = func(self._content, *args, **kwargs)
                else:
                    out = func(*args, **{input_args_name: self._content, **kwargs})

                if out is not None:
                    return make_functional(
                        out,
                        additional_methods=additional_methods,
                        meta_methods=meta_methods,
                        frozen_content=self._frozen_content
                    )
                else:
                    return self

            return pipe_func

        def meta_pipe():
            @add_log_and_comment
            @add_print
            def meta_pipe_func(self, func, *args, **kwargs):
                # print(func(self, *args, **kwargs)._content)
                return make_functional(
                    func(self, *args, **kwargs)._content,
                    additional_methods=additional_methods,
                    meta_methods=meta_methods,
                    frozen_content=self._frozen_content)

            return meta_pipe_func

        for method in dir(object_class):
            if method not in ['__class__', '__new__', '__init__',
                              '__getattribute__', '__setattr__', '__delattr__',
                              '__dict__', '__str__', '__repr__', '__hash__']:
            # if method[:2] != '__' or method in ['__add__']:
                '''
                override only unprotected methods
                '''
                setattr(cls, method, delegate_method(method))

        for func in additional_methods:
            '''
            add these functions as methods
            '''
            setattr(cls, func.__name__, function_to_method(func))

        for meta_func in meta_methods:
            '''
            add these meta functions as methods
            '''
            setattr(cls, meta_func.__name__, meta_function_to_method(meta_func))

        setattr(cls, 'pipe', pipe())
        setattr(cls, 'meta_pipe', meta_pipe())
        setattr(cls, 'return_content', return_content())
        setattr(cls, 'freeze', freeze())
        setattr(cls, 'restore', restore())

        return cls

    return decorate


def make_functional(obj, additional_methods=tuple(),
                    meta_methods=tuple(), frozen_content=None):
    '''
    Used to wrap object methods to return self as default
    when nothing else is returned,
    conforming to functional programming style
    '''

    @delegate(obj.__class__, additional_methods, meta_methods)
    @dataclass(init=False, repr=False)
    class FunctionalObject:
        def __init__(self, content):
            self._content = content
            self._class = content.__class__
            self._frozen_content = {} if frozen_content is None else {**frozen_content}

    return FunctionalObject(obj)
