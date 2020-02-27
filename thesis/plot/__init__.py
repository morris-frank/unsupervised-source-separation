from pathlib import Path
import matplotlib as mpl

from . import toy

mpl.use("TkAgg")
mpl.style.use(f"{Path(__file__).parent.absolute()}/mpl.style")
mpl.colors._colors_full_map["r"] = (0.80, 0.14, 0.11)
mpl.colors._colors_full_map["g"] = (0.60, 0.59, 0.10)
mpl.colors._colors_full_map["b"] = (0.27, 0.52, 0.53)
mpl.colors._colors_full_map["c"] = (0.41, 0.62, 0.42)
mpl.colors._colors_full_map["m"] = (0.69, 0.38, 0.53)
mpl.colors._colors_full_map["y"] = (0.84, 0.60, 0.13)
mpl.colors._colors_full_map["k"] = (0.16, 0.16, 0.16)
mpl.colors._colors_full_map["w"] = (0.98, 0.95, 0.78)
