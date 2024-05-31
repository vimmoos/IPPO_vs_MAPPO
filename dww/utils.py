import inspect as i
from typing import Any, Callable, Dict


def sel_args(kw: Dict[str, Any], fun: Callable) -> Dict[str, Any]:
    """Select only the args needed by the function from the kw."""
    return {k: v for k, v in kw.items() if k in list(i.signature(fun).parameters)}


def sel_and_rm(kw: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Select only the entries which starts
    with the prefix and remove the prefix."""
    lpre = len(prefix)
    return {k[lpre:]: v for k, v in kw.items() if prefix == k[0:lpre]}
