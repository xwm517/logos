"""Binoculars (Hans et al., 2024), reference [14].

Ported from the authors' official implementation at
https://github.com/ahans30/Binoculars (Apache 2.0). The scoring code in
detector.py and metrics.py is theirs; see MODIFICATIONS in detector.py for the
small number of changes made here.
"""

from .binoculars import Binoculars

__all__ = ["Binoculars"]