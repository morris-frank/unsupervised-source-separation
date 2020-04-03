import platform
from os.path import abspath

IS_HERMES = platform.node() == "hermes"
TOY_SIGNALS = ["sin", "square", "saw", "triangle"]

DEFAULT_DATA = "/home/morris/var/data/toy" if IS_HERMES else abspath("../data/toy")
DEFAULT_CHECKPOINTS = "./checkpoints"
