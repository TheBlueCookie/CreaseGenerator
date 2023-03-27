import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import os

from numpy.typing import NDArray


class UnitCell:
    def __init__(self, n_edges: int, origin: NDArray = np.zeros(2), rotation: float = 0, mirror: bool = False,
                 mirror_axis: NDArray = np.zeros((2, 2))):
        self.raw_edges = np.zeros((n_edges, 2, 2))
        self.origin = origin
        self.rotation = rotation
        self.mirror = mirror
        self.mirror_axis = mirror_axis
        self.cell = np.zeros((n_edges, 2, 2))

    def add_edge(self, ind: int, end_point: NDArray):
        assert ind > 0, "Can only add edge after initial edge is defined!"
        self.raw_edges[ind] = [self.raw_edges[ind - 1][1], end_point]

    def plot_cell(self, **kwargs):
        fig, ax = plt.subplots(figsize=(4, 4))
        for start, end in self.cell:
            ax.plot([start[0], end[0]], [start[1], end[1]], color='k', **kwargs)

        return fig, ax

    def _rotate(self):
        raise NotImplementedError

    def _mirror(self):
        raise NotImplementedError

    def generate_cell(self):
        for i, edge in enumerate(self.raw_edges):
            self.cell[i] = [edge[0] + self.origin, edge[1] + self.origin]


class Pattern(ABC):
    def __init__(self, unit_cell: UnitCell, n_unit_cells: int, scaling: float = 1):
        self.unit_cell = unit_cell
        self.cells = []
        self.edges = np.zeros((n_unit_cells, unit_cell.cell.size))
        self.save_dir = 'exported_folds'

    @abstractmethod
    def generate_pattern_from_unit_cell(self):
        pass


class MiuraOriUnitCell(UnitCell):
    def __init__(self, fold_angle: float, x_len: float = 1, y_len: float = 1, bottom_cell: bool = False, **kwargs):
        if bottom_cell:
            n = 3
        else:
            n = 4
        super().__init__(n_edges=n, **kwargs)
        assert 0 <= fold_angle <= np.pi, "Choose fold angle between 0 and 90 deg!"
        self.fold_angle = fold_angle
        self.x_len = x_len
        self.y_len = y_len
        self.center_vertex_offset = np.cos(self.fold_angle) * self.y_len

        offset = -x_len + self.center_vertex_offset

        self.raw_edges[0] = [[x_len + offset, 2 * y_len], [x_len - self.center_vertex_offset + offset, y_len]]
        self.add_edge(ind=1, end_point=[- self.center_vertex_offset + offset, y_len])
        self.raw_edges[2] = [[x_len - self.center_vertex_offset + offset, y_len], [x_len + offset, 0]]

        if not bottom_cell:
            self.add_edge(ind=3, end_point=[offset, 0])

        self.generate_cell()


class MiuraOriFold(Pattern):
    def __init__(self, paper_size: NDArray, n_cells_x: int, n_cells_y: int, fold_angle: float,
                 fit_end_vertex: bool = True):
        assert 0 <= fold_angle <= np.pi, "Choose fold angle between 0 and 90 deg!"

        self.deg_angle = int(fold_angle * 180.0 / np.pi)
        self.n_cells_x = n_cells_x
        self.n_cells_y = n_cells_y
        self.paper_size = paper_size
        self.cell_height = self.paper_size[1] / self.n_cells_y
        if fit_end_vertex:
            self.cell_width = (self.paper_size[0] - 0.5 * np.cos(fold_angle) * self.cell_height) / self.n_cells_x
        else:
            self.cell_width = self.paper_size[0] / self.n_cells_x

        self.unit_cell = MiuraOriUnitCell(fold_angle=fold_angle, x_len=self.cell_width, y_len=0.5 * self.cell_height)
        self.paper_size = paper_size
        super().__init__(unit_cell=self.unit_cell, n_unit_cells=int(self.n_cells_x * self.n_cells_y))

    def generate_pattern_from_unit_cell(self):
        x_coords = np.array([i * self.unit_cell.x_len for i in range(self.n_cells_x + 1)])
        y_coords = np.array([i * self.cell_height for i in range(self.n_cells_y)])
        origins = np.array(np.meshgrid(x_coords, y_coords)).T.reshape(-1, 2)

        for i, origin in enumerate(origins):
            if np.mod(i, self.n_cells_y) == 0:
                bottom_cell = True
            else:
                bottom_cell = False
            self.cells.append(MiuraOriUnitCell(fold_angle=self.unit_cell.fold_angle, x_len=self.unit_cell.x_len,
                                               y_len=self.unit_cell.y_len, origin=origin, bottom_cell=bottom_cell))

    def plot_pattern(self, **kwargs):
        ratio = self.paper_size[0] / self.paper_size[1]
        fig, ax = plt.subplots(figsize=(5 * ratio, 5))

        for cell in self.cells:
            for start, end in cell.cell:
                ax.plot([start[0], end[0]], [start[1], end[1]], color='k', **kwargs)

        ax.set_xlim(0, self.paper_size[0])
        ax.set_ylim(0, self.paper_size[1])

        ax.set_xlabel('mm')
        ax.set_ylabel('mm')

        return fig, ax

    def export_svg(self):
        fig, ax = self.plot_pattern()

        ax.axis('off')
        ax.margins(0, 0)
        fig.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

        save_name = f"miuri_ori-fold_angle_{self.deg_angle}-{self.n_cells_x}_rows-{self.n_cells_y}_" \
                   f"columns_paper_size_{self.paper_size[0]:.0f}-{self.paper_size[1]:.0f}.pdf"

        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)

        fig.savefig(os.path.join(self.save_dir, save_name), format='pdf', bbox_inches="tight", pad_inches=0)
        print(f'Saved to {save_name}')

        return fig, ax
