import importlib


def split_var_spec(qualified_spec: str):
    """
    Splits a qualified variable specifier string into its module and variable components.

    Args:
        qualified_spec (str): A string in the format "<module>:<var>"
                              that specifies a variable within a module.

    Returns:
        tuple: A tuple containing (module, var), where module is the name of
               the module and var is the name of the variable.

    Raises:
        Exception: If the qualified_spec does not follow the required format.
    """
    if ":" not in qualified_spec:
        raise Exception(f"Invalid specifier '{qualified_spec}' does not contain ':'")

    parts = qualified_spec.split(":")
    if len(parts) != 2:
        raise Exception(f"Invalid specifier '{qualified_spec}' has {len(parts)} parts")

    (module, var) = parts

    if not var:
        raise Exception(
            f"You must specify <module>:<var>, '{qualified_spec}' is not valid!"
        )

    return module, var


def resolve_var(qualified_spec: str):
    """
    Resolves a variable from a given qualified specifier.

    Args:
        qualified_spec (str): A string in the format "<module>:<var>"
                              that specifies a variable within a module.

    Returns:
        The value of the resolved variable.

    Raises:
        Exception: If the module or variable cannot be found.
    """

    module, var = split_var_spec(qualified_spec)
    _mod = importlib.import_module(module)

    if not hasattr(_mod, var):
        raise Exception(f"Module {_mod} does not contain var {var}")
    return getattr(_mod, var)


class VarRef(str):
    """
    A string subclass representing a reference to a variable within a module.

    Extends the built-in `str` class and adds the ability to resolve the variable it refers to.

    Methods:
        resolve(self): Resolves the variable referred to by the VarRef object.

    Attributes:
        None: It inherits the attributes of the `str` class, so it has all the standard string methods and properties.
    """

    def __int__(self, value):
        super().__init__(value)
        # checks validity and throws if not
        split_var_spec(self)

    def resolve(self):
        return resolve_var(self)
