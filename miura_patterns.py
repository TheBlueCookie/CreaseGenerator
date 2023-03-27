import matplotlib.pyplot as plt
import numpy as np

from patterns_lib import MiuraOriFold

# paper size in mm
din_a4_portrait = np.array([210, 297])
din_a3_portrait = np.array([297, 420])

paper_size = din_a4_portrait
cells_per_row = 5
cells_per_column = 4

fold_angles = np.linspace(45, 85, 10)

for ang in fold_angles:
    fold_angle = ang

    miura_pattern = MiuraOriFold(paper_size=paper_size, n_cells_x=cells_per_row, n_cells_y=cells_per_column,
                                 fold_angle=fold_angle / 180.0 * np.pi, fit_end_vertex=True)

    miura_pattern.generate_pattern_from_unit_cell()

    # fig, ax = miura_pattern.plot_pattern()
    # plt.show()

    # miura_pattern.export(format='pdf')


