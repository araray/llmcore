import inspect

def list_class_members(cls):
    """
    Print sorted lists of methods vs. attributes for the given class.
    """
    # members = inspect.getmembers(cls)  # list of (name, value) pairs

    # # Separate methods from non-callables
    # methods    = [(n, v) for n, v in members if inspect.isroutine(v)]
    # attributes = [(n, v) for n, v in members if not inspect.isroutine(v)]

    # print(f"Methods ({len(methods)}):")
    # for name, func in methods:
    #     print(f"  • {name}{inspect.signature(func)}")
    # print()
    # print(f"Attributes ({len(attributes)}):")
    # for name, val in attributes:
    #     print(f"  • {name} = {val!r}")
    for i in dir(cls):
        print(i)
