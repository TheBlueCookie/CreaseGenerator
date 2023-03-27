import matplotlib.pyplot as plt
import numpy as np

from patterns_lib import MiuraOriFold

# paper size in mm
din_a4 = np.array([210, 297])
din_a3 = np.array([297, 420])

square = np.array([1000, 1000])

paper_size = din_a4
cells_per_row = 12
cells_per_column = 8

# does not yet correspond to proper angle, 90deg = rectangular pattern
fold_angle = 60

miura_pattern = MiuraOriFold(paper_size=paper_size, n_cells_x=cells_per_row, n_cells_y=cells_per_column,
                             fold_angle=fold_angle / 180.0 * np.pi, fit_end_vertex=True)

miura_pattern.generate_pattern_from_unit_cell()

fig, ax = miura_pattern.plot_pattern()

miura_pattern.export(format='pdf')

plt.show()
