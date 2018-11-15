#!/usr/bin/env python
# coding: utf-8

import logging

from numbers import Number
from typing import Tuple, List, Callable, Union, Iterable
from itertools import cycle, accumulate, chain, product

import numpy as np
from matplotlib.patches import Rectangle

LOG = logging.getLogger(__name__)


class BlockMatrix:
    """A (2D) matrix, stored as a set of submatrices (or blocks)."""

    ndim = 2  # Other dimensions have not been implemented.

    def __init__(self, blocks, _stored_block_shapes=None, check=True):
        self._stored_blocks = np.asarray(blocks)
        assert self._stored_blocks[0][0].ndim == self.ndim

        self._stored_nb_blocks = self._stored_blocks.shape[:self.ndim]

        if _stored_block_shapes is None:
            self._stored_block_shapes = ([block.shape[0] for block in self._stored_blocks[:, 0]],
                                         [block.shape[1] for block in self._stored_blocks[0, :]])
        else:
            # To avoid going down the tree if it is already known.
            self._stored_block_shapes = _stored_block_shapes

        # Total shape of the full matrix
        self.shape = self._compute_shape()

        self.dtype = self._stored_blocks[0][0].dtype

        LOG.debug(f"New block matrix: %s", self)

        if check:
            assert self._check_dimensions_of_blocks()
            assert self._check_dtype()

    def _compute_shape(self):
        # In a dedicated routine because it will be overloaded by subclasses.
        return sum(self._stored_block_shapes[0]), sum(self._stored_block_shapes[1])

    def _check_dimensions_of_blocks(self) -> bool:
        """Check that the dimensions of the blocks are consistent."""
        for line in self.all_blocks:
            block_height = line[0].shape[0]
            for block in line[1:]:
                if not block.shape[0] == block_height:  # Same height on a given line
                    return False

        for col in np.moveaxis(self.all_blocks, 1, 0):
            block_width = col[0].shape[1]
            for block in col[1:]:
                if not block.shape[1] == block_width:  # Same width on a given column
                    return False
        return True

    def _check_dtype(self) -> bool:
        """Check that the type of the blocks are consistent."""
        for line in self._stored_blocks:
            for block in line:
                if block.dtype != self.dtype:
                    return False
        return True

    # ACCESSING DATA

    @property
    def all_blocks(self) -> np.ndarray:
        """The matrix of matrices. For a full block matrix, all the blocks are stored in memory."""
        return self._stored_blocks

    @property
    def block_shapes(self) -> Tuple[List[int], List[int]]:
        """The shapes of the blocks composing the BlockMatrix.

        Example::

            AAAABB
            AAAABB  ->  block_shapes = ([3], [4, 2])
            AAAABB
        """
        return self._stored_block_shapes

    @property
    def nb_blocks(self) -> Tuple[int, int]:
        """The number of blocks in each directions.

        Example::

            AAAABB
            AAAABB  ->  nb_blocks = (1, 2)
            AAAABB
        """
        return self._stored_nb_blocks

    def _stored_block_positions(self, global_frame=(0, 0)) -> Iterable[List[Tuple[int, int]]]:
        """The position of each blocks in the matrix as a generator.
        The list is used by subclasses where the same block may appear several times in different positions.

        Parameters
        ----------
        global_frame: Tuple[int], optional
            the coordinate of the top right corner. Default: 0, 0.

        Example::

            AAAABB
            AAAABB  ->  list(matrix._stored_block_positions) = [[(0,0)], [(0, 4)], [(2, 0)], [(2, 4)]]
            CCCCDD
        """
        x_acc = accumulate([0] + self.block_shapes[0][:-1])
        y_acc = accumulate([0] + self.block_shapes[1][:-1])
        return ([(global_frame[0] + x, global_frame[1] + y)] for x, y in product(x_acc, y_acc))

    def _put_in_full_matrix(self, full_matrix: np.ndarray, where=(0, 0)) -> None:
        """Copy the content of the block matrix in a matrix, which is modified in place."""
        all_blocks_in_flat_iterator = (block for line in self._stored_blocks for block in line)
        positions_of_all_blocks = self._stored_block_positions(global_frame=where)
        for block, positions_of_the_block in zip(all_blocks_in_flat_iterator, positions_of_all_blocks):
            if isinstance(block, BlockMatrix):
                position_of_first_appearance = positions_of_the_block[0]
                frame_of_first_appearance = (slice(position_of_first_appearance[0], position_of_first_appearance[0]+block.shape[0]),
                                             slice(position_of_first_appearance[1], position_of_first_appearance[1]+block.shape[1]))
                block._put_in_full_matrix(full_matrix, where=position_of_first_appearance)

                for position in positions_of_the_block[1:]:  # For the other appearances, only copy the first appearance
                    block_frame = (slice(position[0], position[0]+block.shape[0]),
                                   slice(position[1], position[1]+block.shape[1]))
                    full_matrix[block_frame] = full_matrix[frame_of_first_appearance]

            else:
                full_block = block if isinstance(block, np.ndarray) else block.full_matrix()
                for position in positions_of_the_block:
                    block_frame = (slice(position[0], position[0]+block.shape[0]),
                                   slice(position[1], position[1]+block.shape[1]))
                    full_matrix[block_frame] = full_block

    def full_matrix(self) -> np.ndarray:
        """Flatten the block structure and return a full matrix."""
        full_matrix = np.empty(self.shape, dtype=self.dtype)
        self._put_in_full_matrix(full_matrix)
        return full_matrix

    def __hash__(self):
        # Temporary
        return id(self)

    # TRANSFORMING DATA

    def _apply_unary_op(self, op: Callable) -> 'BlockMatrix':
        """Helper function applying a function recursively on all submatrices."""
        LOG.debug(f"Apply op {op.__name__} to {self}")
        result = [[op(block) for block in line] for line in self._stored_blocks]
        return self.__class__(result, _stored_block_shapes=self._stored_block_shapes, check=False)

    def _apply_binary_op(self, op: Callable, other: 'BlockMatrix') -> 'BlockMatrix':
        """Helper function applying a binary operator recursively on all submatrices."""
        if isinstance(other, self.__class__) and self.nb_blocks == other.nb_blocks:
            LOG.debug(f"Apply op {op.__name__} to {self} and {other}")
            result = [
                [op(block, other_block) for block, other_block in zip(line, other_line)]
                for line, other_line in zip(self._stored_blocks, other._stored_blocks)
            ]
            return self.__class__(result, _stored_block_shapes=self._stored_block_shapes, check=False)
        else:
            return NotImplemented

    def __add__(self, other: 'BlockMatrix') -> 'BlockMatrix':
        from operator import add
        return self._apply_binary_op(add, other)

    def __radd__(self, other: 'BlockMatrix') -> 'BlockMatrix':
        return self + other

    def __neg__(self) -> 'BlockMatrix':
        from operator import neg
        return self._apply_unary_op(neg)

    def __sub__(self, other: 'BlockMatrix') -> 'BlockMatrix':
        from operator import sub
        return self._apply_binary_op(sub, other)

    def __rsub__(self, other: 'BlockMatrix') -> 'BlockMatrix':
        from operator import sub
        return other._apply_binary_op(sub, self)

    def __mul__(self, other: Union['BlockMatrix', Number]) -> 'BlockMatrix':
        if isinstance(other, Number):
            return self._apply_unary_op(lambda x: other*x)
        else:
            from operator import mul
            return self._apply_binary_op(mul, other)

    def __rmul__(self, other: Union['BlockMatrix', Number]) -> 'BlockMatrix':
        return self * other

    def __truediv__(self, other: Union['BlockMatrix', Number]) -> 'BlockMatrix':
        from numbers import Number
        if isinstance(other, Number):
            return self._apply_unary_op(lambda x: x/other)
        else:
            from operator import truediv
            return self._apply_binary_op(truediv, other)

    def __rtruediv__(self, other: Union['BlockMatrix', Number]) -> 'BlockMatrix':
        from numbers import Number
        if isinstance(other, Number):
            return self._apply_unary_op(lambda x: other/x)
        else:
            return self._apply_binary_op(lambda x, y: y/x, other)

    def matvec(self, other):
        """Matrix vector product.
        Named as such to be used as scipy LinearOperator."""
        LOG.debug(f"Multiplication of {self} with a full vector of size {other.shape}.")
        result = np.zeros(self.shape[0], dtype=other.dtype)
        line_heights = self.block_shapes[0]
        line_positions = list(accumulate(chain([0], line_heights)))
        col_widths = self.block_shapes[1]
        col_positions = list(accumulate(chain([0], col_widths)))
        for line, line_position, line_height in zip(self.all_blocks, line_positions, line_heights):
            line_slice = slice(line_position, line_position+line_height)
            for block, col_position, col_width in zip(line, col_positions, col_widths):
                col_slice = slice(col_position, col_position+col_width)
                result[line_slice] += block @ other[col_slice]
        return result

    def matmat(self, other):
        """Matrix-matrix product."""
        if isinstance(other, BlockMatrix) and self.block_shapes[1] == other.block_shapes[0]:
            LOG.debug(f"Multiplication of %s with %s", self, other)
            own_blocks = self.all_blocks
            other_blocks = np.moveaxis(other.all_blocks, 1, 0)
            new_matrix = []
            for own_line in own_blocks:
                new_line = []
                for other_col in other_blocks:
                    new_line.append(sum(own_block @ other_block for own_block, other_block in zip(own_line, other_col)))
                new_matrix.append(new_line)
            return BlockMatrix(new_matrix, check=False)

        elif isinstance(other, np.ndarray) and self.shape[1] == other.shape[0]:
            LOG.debug(f"Multiplication of {self} with a full matrix of shape {other.shape}.")
            # Cut the matrix and recursively call itself to use the code above.
            from capytaine.matrices.builders import cut_matrix
            cut_other = cut_matrix(other, self.block_shapes[1], [other.shape[1]], check=False)
            return (self @ cut_other).full_matrix()

    def __matmul__(self, other: Union['BlockMatrix', np.ndarray]) -> Union['BlockMatrix', np.ndarray]:
        if not (isinstance(other, BlockMatrix) or isinstance(other, np.ndarray)):
            return NotImplemented
        elif other.ndim == 2:
            return self.matmat(other)
        elif other.ndim == 1:
            return self.matvec(other)
        else:
            return NotImplemented

    def astype(self, dtype: np.dtype) -> 'BlockMatrix':
        return self._apply_unary_op(lambda x: x.astype(dtype))

    @property
    def T(self) -> 'BlockMatrix':
        """Transposed matrix."""
        transposed_blocks = self._apply_unary_op(lambda x: x.T)  # Transpose subblocks recursively
        transposed_blocks._stored_blocks = transposed_blocks._stored_blocks.T  # Change position of blocks
        return transposed_blocks

    def fft_of_list(*block_matrices, check=True) -> List['BlockMatrix']:
        """Compute the fft of a list of block matrices of the same type and shape.
        The output is a list of block matrices of the same shape as the input ones.
        The fft is computed element-wise, so the block structure does not cause any mathematical difficulty.
        """
        from capytaine.matrices.builders import zeros_like

        class_of_matrices = type(block_matrices[0])
        nb_blocks = block_matrices[0]._stored_nb_blocks

        if check:
            # Check the validity of the shapes of the matrices given as input
            shape = block_matrices[0].shape
            assert [nb_blocks == matrix._stored_nb_blocks for matrix in block_matrices[1:]]
            assert [shape == matrix.shape for matrix in block_matrices[1:]]
            assert [class_of_matrices == type(matrix) for matrix in block_matrices[1:]]

        result = [zeros_like(matrix, dtype=np.complex) for matrix in block_matrices]

        for i_block, j_block in product(range(nb_blocks[0]), range(nb_blocks[1])):
            list_of_i_j_blocks = [block_matrices[i_matrix]._stored_blocks[i_block, j_block]
                                  for i_matrix in range(len(block_matrices))]

            if isinstance(list_of_i_j_blocks[0], np.ndarray):  # All blocks should be of the same shape.
                fft_of_blocks = np.fft.fft(np.array(list_of_i_j_blocks), axis=0)
            else:
                fft_of_blocks = BlockMatrix.fft_of_list(*list_of_i_j_blocks, check=False)

            for matrix, computed_block in zip(result, fft_of_blocks):
                matrix._stored_blocks[i_block, j_block] = computed_block

        return result

    # COMPARISON AND REDUCTION

    def __eq__(self, other: 'BlockMatrix') -> 'BlockMatrix[bool]':
        from operator import eq
        return self._apply_binary_op(eq, other)

    def __invert__(self) -> 'BlockMatrix':
        """Boolean not (~)"""
        from operator import invert
        return self._apply_unary_op(invert)

    def __ne__(self, other: 'BlockMatrix') -> 'BlockMatrix[bool]':
        return ~(self == other)

    def all(self) -> bool:
        for line in self._stored_blocks:
            for block in line:
                if not block.all():
                    return False
        return True

    def any(self) -> bool:
        for line in self._stored_blocks:
            for block in line:
                if block.any():
                    return True
        return False

    def min(self) -> Number:
        return min(block.min() for line in self._stored_blocks for block in line)

    def max(self) -> Number:
        return max(block.max() for line in self._stored_blocks for block in line)

    # DISPLAYING DATA

    def __str__(self):
        args = [f"nb_blocks={self.nb_blocks}", f"shape={self.shape}"]
        if self.dtype not in [np.float64, np.float]:
            args.append(f"dtype={self.dtype}")
        return f"{self.__class__.__name__}(" + ", ".join(args) + ")"

    display_color = cycle([f'C{i}' for i in range(10)])

    def _patches(self,
                 global_frame: Union[Tuple[int, int], np.ndarray]
                 ) -> List[Rectangle]:
        """Helper function for displaying the shape of the matrix.
        Recursively returns a list of rectangles representing the sub-blocks of the matrix.

        Parameters
        ----------
        global_frame: tuple of ints
            coordinates of the origin in the top left corner.
        """
        all_blocks_in_flat_iterator = (block for line in self._stored_blocks for block in line)
        positions_of_all_blocks = self._stored_block_positions(global_frame=global_frame)
        patches = []
        for block, positions_of_the_block in zip(all_blocks_in_flat_iterator, positions_of_all_blocks):
            position_of_first_appearance = positions_of_the_block[0]
            # Exchange coordinates: row index i -> y, column index j -> x
            position_of_first_appearance = np.array((position_of_first_appearance[1], position_of_first_appearance[0]))

            if isinstance(block, BlockMatrix):
                patches_of_this_block = block._patches(position_of_first_appearance)
            elif isinstance(block, np.ndarray):
                patches_of_this_block = [Rectangle(position_of_first_appearance,
                                                   block.shape[1], block.shape[0],
                                                   edgecolor='k', facecolor=next(self.display_color))]
            else:
                raise NotImplementedError()

            patches.extend(patches_of_this_block)

            # For the other appearances, copy the patches of the first appearance
            for block_position in positions_of_the_block[1:]:
                block_position = np.array((block_position[1], block_position[0]))
                for patch in patches_of_this_block:  # A block can be made of several patches.
                    shift = block_position - position_of_first_appearance
                    patch_position = np.array(patch.get_xy()) + shift
                    patches.append(Rectangle(patch_position, patch.get_width(), patch.get_height(),
                                             facecolor=patch.get_facecolor(), alpha=0.5))

        return patches

    def plot_shape(self):
        """Plot the structure of the matrix using matplotlib."""
        import matplotlib.pyplot as plt
        plt.figure()
        for patch in self._patches((0, 0)):
            plt.gca().add_patch(patch)
        plt.axis('equal')
        plt.xlim(0, self.shape[1])
        plt.ylim(0, self.shape[0])
        plt.gca().invert_yaxis()
        plt.show()

