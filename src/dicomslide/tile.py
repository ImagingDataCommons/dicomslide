import logging
import itertools
from typing import Optional, Sequence, Tuple, Union

import highdicom as hd
import numpy as np
from pydicom.dataset import Dataset
from pydicom.tag import Tag

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
    tile_shape: Tuple[int, ...]
    if total_pixel_matrix.ndim == 3:
        tile_shape = (rows, columns, total_pixel_matrix.shape[-1])
    elif total_pixel_matrix.ndim == 2:
        tile_shape = (rows, columns)
    else:
        raise ValueError(
            "Total pixel matrix has unexpected number of dimensions."
        )
    tiles = np.zeros(
        (len(tile_positions), *tile_shape),
        dtype=total_pixel_matrix.dtype
    )
    for i, (row_offset, col_offset) in enumerate(tile_positions):
        pixel_array = total_pixel_matrix[
            row_offset:(row_offset + rows),
            col_offset:(col_offset + columns),
            ...,
        ]
        tiles[
            i,
            0:pixel_array.shape[0],
            0:pixel_array.shape[1],
            ...
        ] = pixel_array.copy()
        del pixel_array
    return tiles


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

    def _get_channel_index(item: Dataset) -> float:
        return float(channel_identifier_lut[get_channel_identifier(item)])

    # Use tags via the lower-level pydicom API to avoid repeated tag lookup
    # operations and speed up parsing of Per-Frame Functional Groups Sequence.
    plane_pos_seq_tag = Tag('PlanePositionSlideSequence')
    row_pos_tag = Tag('RowPositionInTotalImagePixelMatrix')
    col_pos_tag = Tag('ColumnPositionInTotalImagePixelMatrix')
    x_offset_tag = Tag('XOffsetInSlideCoordinateSystem')
    y_offset_tag = Tag('YOffsetInSlideCoordinateSystem')
    z_offset_tag = Tag('ZOffsetInSlideCoordinateSystem')

    def _get_position_indices(item: Dataset) -> Tuple[
        float, float, float, float, float
    ]:
        pos_item = item[plane_pos_seq_tag].value[0]
        return (
            float(pos_item[row_pos_tag].value) - 1.0,
            float(pos_item[col_pos_tag].value) - 1.0,
            float(pos_item[x_offset_tag].value),
            float(pos_item[y_offset_tag].value),
            float(pos_item[z_offset_tag].value),
        )

    num_frames = int(getattr(image, 'NumberOfFrames', '1'))
    if hasattr(image, 'PerFrameFunctionalGroupsSequence'):
        # The information may be stored in the item of the Shared Functional
        # Groups Sequence in case it is shared across all frames.
        if hasattr(image, 'SharedFunctionalGroupsSequence'):
            shared_item = image.SharedFunctionalGroupsSequence[0]
        else:
            shared_item = Dataset()
        channel_index: Optional[float] = None
        if 'OpticalPathIdentificationSequence' in shared_item:
            channel_index = _get_channel_index(shared_item)
        position_indices: Optional[
            Tuple[float, float, float, float, float]
        ] = None
        if 'PlanePositionSlideSequence' in shared_item:
            position_indices = _get_position_indices(shared_item)
        # Not pretty, but more performant than a for loop.
        if channel_index is not None and position_indices is not None:
            positions = np.tile(
                np.array([
                    channel_index,
                    *position_indices,
                ]),
                (image.NumberOfFrames, 1)
            )
        elif channel_index is None and position_indices is not None:
            positions = np.stack([
                np.array([
                    _get_channel_index(frame_item),
                    *position_indices,
                ])
                for frame_item in image.PerFrameFunctionalGroupsSequence
            ])
        elif channel_index is not None and position_indices is None:
            positions = np.stack([
                np.array([
                    channel_index,
                    *_get_position_indices(frame_item),
                ])
                for frame_item in image.PerFrameFunctionalGroupsSequence
            ])
        else:
            positions = np.stack([
                np.array([
                    _get_channel_index(frame_item),
                    *_get_position_indices(frame_item),
                ])
                for frame_item in image.PerFrameFunctionalGroupsSequence
            ])
    else:
        msg = (
            'Image lacks a PerFrameFunctionalGroupsSequence '
            'but is not "TILED_FULL". Note that sometimes this '
            'occurs due to limitations on the length of sequences '
            'retrieved using WADO.'
        )
        if not hasattr(image, 'DimensionOrganizationType'):
            raise AttributeError(msg)
        elif image.DimensionOrganizationType != 'TILED_FULL':
            raise AttributeError(msg)

        image_origin = image.TotalPixelMatrixOriginSequence[0]
        image_orientation = (
            float(image.ImageOrientationSlide[0]),
            float(image.ImageOrientationSlide[1]),
            float(image.ImageOrientationSlide[2]),
            float(image.ImageOrientationSlide[3]),
            float(image.ImageOrientationSlide[4]),
            float(image.ImageOrientationSlide[5]),
        )
        tiles_per_column = int(
            np.ceil(image.TotalPixelMatrixRows / image.Rows)
        )
        tiles_per_row = int(
            np.ceil(image.TotalPixelMatrixColumns / image.Columns)
        )
        num_focal_planes = getattr(
            image,
            'TotalPixelMatrixFocalPlanes',
            1
        )
        num_optical_paths = getattr(
            image,
            'NumberOfOpticalPaths',
            len(image.OpticalPathSequence)
        )

        shared_fg = image.SharedFunctionalGroupsSequence[0]
        pixel_measures = shared_fg.PixelMeasuresSequence[0]
        pixel_spacing = (
            float(pixel_measures.PixelSpacing[0]),
            float(pixel_measures.PixelSpacing[1]),
        )
        spacing_between_slices = float(
            getattr(
                pixel_measures,
                'SpacingBetweenSlices',
                1.0
            )
        )
        x_offset = float(image_origin.XOffsetInSlideCoordinateSystem)
        y_offset = float(image_origin.YOffsetInSlideCoordinateSystem)

        transformer_lut = {}
        for s in range(1, num_focal_planes + 1):
            # These checks are needed for mypy to determine the correct type
            z_offset = float(s - 1) * spacing_between_slices
            transformer_lut[s] = hd.spatial.PixelToReferenceTransformer(
                image_position=(x_offset, y_offset, z_offset),
                image_orientation=image_orientation,
                pixel_spacing=pixel_spacing
            )

        rows = int(image.Rows)
        columns = int(image.Columns)
        channel_indices = np.repeat(
            np.arange(num_channels),
            repeats=int(num_frames / num_channels)
        )

        # This is ugly, but the list comprehensive speeds up the computation
        # when compared to a for loop.
        positions = np.stack([
            np.concatenate([
                np.array([channel_index - 1], dtype=float),
                np.array([(r - 1) * rows, (c - 1) * columns], dtype=float),
                transformer_lut[slice_index](
                    np.array([[(c - 1) * columns, (r - 1) * rows]], dtype=int)
                )[0]
            ])
            for channel_index, slice_index, r, c in itertools.product(
                range(1, num_optical_paths + 1),
                range(1, num_focal_planes + 1),
                range(1, tiles_per_column + 1),
                range(1, tiles_per_row + 1),
            )
        ])

    channel_indices = positions[:, 0].astype(int)
    matrix_positions = positions[:, 1:3].astype(int)
    slide_positions = positions[:, 3:].astype(float)

    focal_plane_indices = np.zeros((num_frames, ), dtype=int)
    z_offset_values = slide_positions[:, 2]
    unique_z_offset_values = np.unique(z_offset_values)
    for index, value in enumerate(unique_z_offset_values):
        focal_plane_indices[z_offset_values == value] = index

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
