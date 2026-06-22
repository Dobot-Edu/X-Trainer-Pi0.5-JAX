__all__ = ["AtomUpperControl"]


def __getattr__(name):
    if name == "AtomUpperControl":
        from examples.atom_real.hardware.atom_upper_control import AtomUpperControl

        return AtomUpperControl
    raise AttributeError(name)
