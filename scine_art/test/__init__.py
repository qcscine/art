#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

import unittest
from functools import wraps
from pkgutil import iter_modules
from typing import Callable
import subprocess


def _skip(func: Callable, error: str):
    if 'pytest' in (name for loader, name, ispkg in iter_modules()):
        import pytest
        return pytest.mark.skip(reason=error)(func)
    else:
        return unittest.skip(error)(func)


def skip_without_dot() -> Callable:
    def wrap(f: Callable):
        completed_proc = subprocess.run("dot -V", shell=True)
        if completed_proc.returncode == 0:
            @wraps(f)
            def wrapped_f(*args, **kwargs):
                f(*args, **kwargs)
            return wrapped_f
        else:
            return _skip(f, "Test requires 'graphviz - dot' ")

    return wrap
