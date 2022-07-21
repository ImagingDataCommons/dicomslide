import logging
from typing import Sequence, Tuple, Union

import highdicom as hd
import numpy as np
from pydicom.dataset import Dataset

from dicomslide._channel import _get_channel_info
from dicomslide.utils import is_tiled_image


logger = logging.getLogger(__name__)


def disassemble_total_pixel_matrix(
    total_pixel_matrix: np.ndarray,
    tile_positions: Union[Sequence[Tuple[int, int]], np.ndarray],
    rows: int,
    columns: int,
) -> np.ndarray:
    """Disassemble a total pixel matrix into individual tiles.

    Parameters
    ----------
    total_pixel_matrix: numpy.ndarray
        Total pixel matrix
    tile_positions: Union[Sequence[Tuple[int, int]], numpy.ndarray]
        Zero-based (row, column) position of each tile in the total pixel matrix
    rows: int
        Number of rows per tile
    columns: int
        Number of columns per tile

    Returns
    -------
    numpy.ndarray
        Stacked image tiles

    """
    logger.debug('disassemble total pixel matrix')
    tiles = []
    tile_shape: Tuple[int, ...]
    if total_pixel_matrix.ndim == 3:
        tile_shape = (rows, columns, total_pixel_matrix.shape[-1])
    elif total_pixel_matrix.ndim == 2:
        tile_shape = (rows, columns)
    else:
        raise ValueError(
            "Total pixel matrix has unexpected number of dimensions."
        )
    for row_offset, col_offset in tile_positions:
        tile = np.zeros(tile_shape, dtype=total_pixel_matrix.dtype)
        pixel_array = total_pixel_matrix[
            row_offset:(row_offset + rows),
            col_offset:(col_offset + columns),
            ...,
        ]
        tile[
            0:pixel_array.shape[0],
            0:pixel_array.shape[1],
            ...
        ] = pixel_array
        tiles.append(tile)
    return np.stack(tiles)


def assemble_total_pixel_matrix(
    tiles: Sequence[np.ndarray],
    tile_positions: Union[Sequence[Tuple[int, int]], np.ndarray],
    total_pixel_matrix_rows: int,
    total_pixel_matrix_columns: int,
) -> np.ndarray:
    """Assemble a total pixel matrix from individual tiles.

    Parameters
    ----------
    tiles: Sequence[numpy.ndarray]
        Individual image tiles
    tile_positions: Union[Sequence[Tuple[int, int]], numpy.ndarray]
        Zero-based (row, column) position of each tile in the total pixel matrix
    total_pixel_matrix_rows: int
        Number of total rows
    total_pixel_matrix_columns: int
        Number of total columns

    Returns
    -------
    numpy.ndarray
        Total pixel matrix

    """
    logger.debug('assemble total pixel matrix')
    if tiles[0].ndim == 3:
        rows, columns = tiles[0].shape[-3:-1]
        total_pixel_matrix = np.zeros(
            (
                total_pixel_matrix_rows,
                total_pixel_matrix_columns,
                tiles[0].shape[2],
            ),
            dtype=tiles[0].dtype,
        ) + tiles[0].dtype.type(255)
    else:
        rows, columns = tiles[0].shape[-2:]
        total_pixel_matrix = np.zeros(
            (
                total_pixel_matrix_rows,
                total_pixel_matrix_columns,
            ),
            dtype=tiles[0].dtype,
        )
    for i, tile in enumerate(tiles):
        row_start = tile_positions[i][0]
        col_start = tile_positions[i][1]
        row_stop = row_start + rows
        row_diff = total_pixel_matrix_rows - row_stop
        if row_diff < 0:
            frame_row_stop = row_diff
        else:
            frame_row_stop = None
        col_stop = col_start + columns
        col_diff = total_pixel_matrix_columns - col_stop
        if col_diff < 0:
            frame_col_stop = col_diff
        else:
            frame_col_stop = None
        total_pixel_matrix[
            row_start:row_stop,
            col_start:col_stop,
            ...,
        ] = tile[:frame_row_stop, :frame_col_stop, ...]

    return total_pixel_matrix


def compute_frame_positions(
    image: Dataset
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the positions of frames.

    Parameters
    ----------
    image : pydicom.Dataset
        Metadata of a tiled image

    Returns
    -------
    total_pixel_matrix_positions: numpy.ndarray
        Zero-based (row, column) offset of the center of the top lefthand corner
        pixel of each frame from the origin of the total pixel matrix in
        pixel unit.  Values are unsigned integers in the range
        [0, Total Pixel Matrix Rows) and [0, Total Pixel Matrix Columns).
        The position of the top lefthand corner tile is (0, 0).
    slide_positions: numpy.ndarray
        Zero-based (x, y, z) offset of the center of the top lefthand corner
        pixel of each frame from the origin of the slide coordinate system
        (frame of reference) in millimeter unit. Values are floating-point
        numbers in the range [-inf, inf].
    channel_indices: numpy.ndarray
        Zero-based index for each frame into channels along the direction
        defined by successive items of the appropriate attribute. In case of
        a VL Whole Slide Microscopy Image, the attribute is the Optical Path
        Sequence, and in case of Segmentation, the attribute is the Segment
        Sequence.
    focal_plane_indices: numpy.ndarray
        Zero-based index for each frame into focal planes along depth direction
        from the glass slide towards the coverslip in the slide coordinate
        system specified by the Z Offset in Slide Coordinate System attribute.
        Values are integers in the range
        [0, Total Pixel Matrix Focal Planes).

    """
    if not is_tiled_image(image):
        raise ValueError('Argument "image" is not a a tiled image.')

    sop_instance_uid = image.SOPInstanceUID
    logger.debug(
        f'compute frame positions for image "{sop_instance_uid}"'
    )
    channels, get_channel_identifier = _get_channel_info(image)
    channel_identifier_lut = {
        str(ch.channel_identifier): i
        for i, ch in enumerate(channels)
    }
    num_channels = len(channels)

    num_frames = int(getattr(image, 'NumberOfFrames', '1'))
    focal_plane_indices = np.zeros((num_frames, ), dtype=int)
    matrix_positions = np.zeros((num_frames, 2), dtype=int)
    slide_positions = np.zeros((num_frames, 3), dtype=float)
    if hasattr(image, 'PerFrameFunctionalGroupsSequence'):
        channel_indices = np.zeros((num_frames, ), dtype=int)
        for i in range(num_frames):
            frame_item = image.PerFrameFunctionalGroupsSequence[i]
            try:
                plane_pos_item = frame_item.PlanePositionSlideSequence[0]
            except AttributeError as error:
                raise AttributeError(
                    f'Item #{i + 1} of Per-Frame Functional Groups Sequence '
                    'does not have attribute Plane Position Slide Sequence: '
                    f'{error}'
                )
            matrix_positions[i, :] = (
                int(plane_pos_item.RowPositionInTotalImagePixelMatrix) - 1,
                int(plane_pos_item.ColumnPositionInTotalImagePixelMatrix) - 1,
            )
            slide_positions[i, :] = (
                float(plane_pos_item.XOffsetInSlideCoordinateSystem),
                float(plane_pos_item.YOffsetInSlideCoordinateSystem),
                float(plane_pos_item.ZOffsetInSlideCoordinateSystem),
            )
            channel_indices[i] = channel_identifier_lut[
                get_channel_identifier(frame_item)
            ]
    else:
        channel_indices = np.repeat(
            np.arange(num_channels),
            repeats=int(num_frames / num_channels)
        )
        plane_positions = hd.utils.compute_plane_position_slide_per_frame(image)
        for i in range(num_frames):
            plane_pos_item = plane_positions[i][0]
            matrix_positions[i, :] = (
                int(plane_pos_item.RowPositionInTotalImagePixelMatrix) - 1,
                int(plane_pos_item.ColumnPositionInTotalImagePixelMatrix) - 1,
            )
            slide_positions[i, :] = (
                float(plane_pos_item.XOffsetInSlideCoordinateSystem),
                float(plane_pos_item.YOffsetInSlideCoordinateSystem),
                float(plane_pos_item.ZOffsetInSlideCoordinateSystem),
            )

    z_offset_values = slide_positions[:, 2]
    unique_z_offset_values = np.unique(z_offset_values)
    for index, value in enumerate(unique_z_offset_values):
        matches = z_offset_values == value
        focal_plane_indices[matches] = index

    if hasattr(image, 'TotalPixelMatrixFocalPlanes'):
        num_focal_planes = image.TotalPixelMatrixFocalPlanes
        if len(unique_z_offset_values) != num_focal_planes:
            raise ValueError(
                'Could not compute tile positions, because '
                'an unexpected number of focal planes was found.'
            )

    return (
        matrix_positions,
        slide_positions,
        channel_indices,
        focal_plane_indices,
    )
