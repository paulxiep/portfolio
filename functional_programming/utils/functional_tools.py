import copy


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

            def return_self_as_default(self, *args, **kwargs):
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

            def f_to_m(self, *args, **kwargs):
                self._content = func(self._content, *args, **kwargs)
                return self

            return f_to_m

        def meta_function_to_method(func):
            '''
            add meta-level functions as methods
            '''

            def mf_to_m(self, *args, **kwargs):
                self._content = func(self, *args, **kwargs)
                return self

            return mf_to_m

        def freeze():
            '''
            allow one instance of _content freeze
            '''

            def freeze_content(self):
                self._frozen_content = copy.deepcopy(self._content)
                return self

            return freeze_content

        def restore():
            '''
            restore frozen _content
            and swap current _content into frozen
            '''

            def restore_content(self):
                return make_functional(copy.deepcopy(self._frozen_content),
                                       additional_methods, meta_methods,
                                       frozen_content=copy.deepcopy(self._content))

            return restore_content

        def return_content():
            '''
            exit functional wrapper class and return current _content
            '''

            def content_return(self):
                return self._content

            return content_return

        for method in dir(object_class):
            if method[:2] != '__':
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

        setattr(cls, 'return_content', return_content())
        setattr(cls, 'freeze', freeze())
        setattr(cls, 'restore', restore())

        return cls

    return decorate


def make_functional(obj, additional_methods=tuple(), meta_methods=tuple(), frozen_content=None):
    '''
    Used to wrap object methods to return self as default when nothing else is returned,
    conforming to functional programming style
    '''

    @delegate(obj.__class__, additional_methods, meta_methods)
    class FunctionalObject:
        def __init__(self, content):
            self._content = content
            self._class = content.__class__
            self._frozen_content = frozen_content

    return FunctionalObject(obj)
