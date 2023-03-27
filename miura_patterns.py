import matplotlib.pyplot as plt
import numpy as np

from patterns_lib import MiuraOriFold

# paper size in mm
din_a4 = np.array([210, 297])

paper_size = din_a4
cells_per_row = 3
cells_per_column = 4

# does not yet correspond to proper angle, 90deg = rectangular pattern
fold_angle = 80

miura_pattern = MiuraOriFold(paper_size=paper_size, n_cells_x=cells_per_row, n_cells_y=cells_per_column,
                             fold_angle=fold_angle / 180.0 * np.pi, fit_end_vertex=True)

miura_pattern.generate_pattern_from_unit_cell()

fig_p, ax_p = miura_pattern.export_svg()

plt.show()
