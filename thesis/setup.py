import platform
from os.path import abspath

IS_HERMES = platform.node() == "hermes"
DEFAULT_CHECKPOINTS = "./checkpoints"

# These signals are ordered:
TOY_SIGNALS = ["sin", "square", "saw", "triangle"]
DEFAULT_TOY = "/home/morris/var/data/toy" if IS_HERMES else abspath("../data/toy")

MUSDB_SIGNALS = ["drums", "bass", "other", "vocals"]
DEFAULT_MUSDB = "/home/morris/var/data/musdb18" if IS_HERMES else abspath("../data/musdb")


class __DEFAULT:
    musdb = False
    all_signals = TOY_SIGNALS + MUSDB_SIGNALS

    @property
    def signals(self):
        return MUSDB_SIGNALS if self.musdb else TOY_SIGNALS

    @property
    def data(self):
        return DEFAULT_MUSDB if self.musdb else DEFAULT_TOY


DEFAULT = __DEFAULT()
