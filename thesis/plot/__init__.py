from pathlib import Path
import matplotlib as mpl

mpl.use('TkAgg')
mpl.style.use(f'{Path(__file__).parent.absolute()}/mpl.style')
mpl.colors._colors_full_map['r'] = (.80, .14, .11)
mpl.colors._colors_full_map['g'] = (.60, .59, .10)
mpl.colors._colors_full_map['b'] = (.27, .52, .53)
mpl.colors._colors_full_map['c'] = (.41, .62, .42)
mpl.colors._colors_full_map['m'] = (.69, .38, .53)
mpl.colors._colors_full_map['y'] = (.84, .60, .13)
mpl.colors._colors_full_map['k'] = (.16, .16, .16)
mpl.colors._colors_full_map['w'] = (.98, .95, .78)
