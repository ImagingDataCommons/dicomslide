from typing import Sequence, Tuple

import highdicom as hd
import numpy as np
from pydicom.dataset import Dataset


def is_image(dataset: Dataset) -> bool:
    """Determine whether a dataset is an image.

    Parameters
    ----------
    dataset: pydicom.dataset.Dataset
        Dataset

    Returns
    -------
    bool
        Whether dataset is an image

    """
    return all([
        hasattr(dataset, 'BitsAllocated'),
        hasattr(dataset, 'Columns'),
        hasattr(dataset, 'Rows'),
        hasattr(dataset, 'PhotometricInterpretation'),
    ])


def is_tiled_image(dataset: Dataset) -> bool:
    """Determine whether a dataset is a tiled image.

    Parameters
    ----------
    dataset: pydicom.dataset.Dataset
        Dataset

    Returns
    -------
    bool
        Whether dataset is a tiled image

    """
    return all([
        is_image(dataset),
        hasattr(dataset, 'TotalPixelMatrixColumns'),
        hasattr(dataset, 'TotalPixelMatrixRows'),
    ])


def disassemble_total_pixel_matrix(
    total_pixel_matrix: np.ndarray,
    tile_positions: Sequence[Tuple[int, int]],
    rows: int,
    columns: int,
) -> np.ndarray:
    """Disassemble a total pixel matrix into individual tiles.

    Parameters
    ----------
    total_pixel_matrix: numpy.ndarray
        Total pixel matrix
    tile_positions: Sequence[Tuple[int, int]]
        Column, Row position of each tile relative to the slide
    rows: int
        Number of rows per tile
    columns: int
        Number of columns per tile

    Returns
    -------
    numpy.ndarray
        Stacked image tiles

    """
    tiles = []
    if total_pixel_matrix.ndim == 3:
        tile_shape = (rows, columns, total_pixel_matrix.shape[-1])
    elif total_pixel_matrix.ndim == 2:
        tile_shape = (rows, columns)
    else:
        raise ValueError(
            "Total pixel matrix has unexpected number of dimensions."
        )
    for row_offset, column_offset in tile_positions:
        tile = np.zeros(tile_shape, dtype=total_pixel_matrix.dtype)
        pixel_array = total_pixel_matrix[
            row_offset:(row_offset + rows),
            column_offset:(column_offset + columns),
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
    tile_positions: Sequence[Tuple[int, int]],
    total_pixel_matrix_rows: int,
    total_pixel_matrix_columns: int,
) -> np.ndarray:
    """Assemble a total pixel matrix from individual tiles.

    Parameters
    ----------
    tiles: Sequence[numpy.ndarray]
        Individual image tiles
    tile_positions: Sequence[Tuple[int, int]]
        Column, Row position of each tile relative to the slide
    total_pixel_matrix_rows: int
        Number of total rows
    total_pixel_matrix_columns: int
        Number of total columns

    Returns
    -------
    numpy.ndarray
        Total pixel matrix

    """
    if tiles[0].ndim == 3:
        rows, columns = tiles[0].shape[-3:-1]
        total_pixel_matrix = (
            np.ones(
                (
                    total_pixel_matrix_rows,
                    total_pixel_matrix_columns,
                    tiles[0].shape[2],
                ),
                dtype=tiles[0].dtype,
            )
            * 255
        )
    else:
        rows, columns = tiles[0].shape[-2:]
        total_pixel_matrix = np.zeros(
            (
                total_pixel_matrix_rows,
                total_pixel_matrix_columns,
            ),
            dtype=tiles[0].dtype,
        )
    for i, frame in enumerate(tiles):
        row_start = tile_positions[i][1] - 1
        col_start = tile_positions[i][0] - 1
        row_stop = row_start + rows
        col_stop = col_start + columns
        total_pixel_matrix[
            row_start:row_stop,
            col_start:col_stop,
            ...,
        ] = frame

    return total_pixel_matrix


def compute_tile_positions(
    image: Dataset
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the offset of tiles from the origin of the total pixel matrix.

    Parameters
    ----------
    image : pydicom.Dataset
        Metadata of a tiled image

    Returns
    -------
    total_pixel_matrix_positions: numpy.ndarray
        One-based (column, row) offset of the center of the top lefthand corner
        pixel of each tile (frame) from the origin of the total pixel matrix in
        pixel unit.  Values are unsigned integers in the range
        [1, Total Pixel Matrix Columns] and [1, Total Pixel Matrix Rows].
        The position of the top lefthand corner tile is (1, 1).
    slide_positions: numpy.ndarray
        Zero-based (x, y, z) offset of the center of the top lefthand corner
        pixel of each tile (frame) from the origin of the slide coordinate
        system (frame of reference) in millimeter unit. Values are
        floating-point numbers in the range [-inf, inf].
    optical_path_indices: numpy.ndarray
        One-based index for each frame into optical paths along the direction
        defined by successive items of the Optical Path Sequence attribute.
        Values are integers in the range [1, Number of Optical Paths].
    focal_plane_indices: numpy.ndarray
        One-based index for each frame into focal planes along depth direction
        from the glass slide towards the coverslip in the slide coordinate
        system specified by the Z Offset in Slide Coordinate System attribute.
        Values are integers in the range
        [1, Total Pixel Matrix Focal Planes].

    """
    if not is_tiled_image(image):
        raise ValueError('Argument "image" is not a a tiled image.')

    optical_path_identifier_lut = {
        str(item.OpticalPathIdentifier): i
        for i, item in enumerate(image.OpticalPathSequence)
    }
    num_optical_paths = len(optical_path_identifier_lut)

    num_frames = int(image.NumberOfFrames)
    focal_plane_indices = np.zeros((num_frames, ), dtype=int)
    matrix_positions = np.zeros((num_frames, 2), dtype=int)
    slide_positions = np.zeros((num_frames, 3), dtype=float)
    if hasattr(image, 'PerFrameFunctionalGroupsSequence'):
        optical_path_indices = np.zeros((num_frames, ), dtype=int)
        for i in range(num_frames):
            frame_item = image.PerFrameFunctionalGroupsSequence[i]
            plane_position_item = frame_item.PlanePositionSlideSequence[0]
            optical_path_item = frame_item.OpticalPathIdentificationSequence[0]
            matrix_positions[i, :] = (
                int(plane_position_item.ColumnPositionInTotalImagePixelMatrix),
                int(plane_position_item.RowPositionInTotalImagePixelMatrix),
            )
            slide_positions[i, :] = (
                float(plane_position_item.XOffsetInSlideCoordinateSystem),
                float(plane_position_item.YOffsetInSlideCoordinateSystem),
                float(plane_position_item.ZOffsetInSlideCoordinateSystem),
            )
            optical_path_indices[i] = optical_path_identifier_lut[
                optical_path_item.OpticalPathIdentifier
            ]
    else:
        optical_path_indices = np.repeat(
            np.arange(num_optical_paths),
            repeats=int(num_frames / num_optical_paths)
        )
        plane_positions = hd.utils.compute_plane_position_slide_per_frame(image)
        for i in range(num_frames):
            plane_position_item = plane_positions[i][0]
            matrix_positions[i, :] = (
                int(plane_position_item.ColumnPositionInTotalImagePixelMatrix),
                int(plane_position_item.RowPositionInTotalImagePixelMatrix),
            )
            slide_positions[i, :] = (
                float(plane_position_item.XOffsetInSlideCoordinateSystem),
                float(plane_position_item.YOffsetInSlideCoordinateSystem),
                float(plane_position_item.ZOffsetInSlideCoordinateSystem),
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

    # These are one-based indices for consistency with other DICOM indexes
    optical_path_indices += 1
    focal_plane_indices += 1

    return (
        matrix_positions,
        slide_positions,
        optical_path_indices,
        focal_plane_indices,
    )


def compute_tile_grid_index(image: Dataset) -> np.ndarray:
    """Compute the index of each tile within the tile grid.

    Parameters
    ----------
    image : pydicom.Dataset
        Metadata of a tiled image

    Returns
    -------
    numpy.ndarray
        Zero-based (row, column) index of each tile in the tile grid in
        tile unit

    """
    if not is_tiled_image(image):
        raise ValueError('Argument "image" is not a a tiled image.')

    tile_positions = compute_tile_positions(image)[0]
    columns = image.Columns
    rows = image.Rows
    return np.column_stack([
        np.floor((tile_positions[:, 1] - 1) / rows).astype(int),
        np.floor((tile_positions[:, 0] - 1) / columns).astype(int),
    ])


def compute_tile_contours(
    image: Dataset
) -> Tuple[hd.sr.GraphicTypeValues3D, Sequence[np.ndarray], hd.UID]:
    """Compute contours of image tiles in the slide coordinate system.

    Parameters
    ----------
    image: pydicom.dataset.Dataset
        Metadata of a DICOM VL Whole Slide Microscopy Image
    frame_numbers: Union[Sequence[int], None], optional
        One-based index number of frames for which contours should be obtained

    Returns
    -------
    graphic_type: highdicom.sr.GraphicTypeValues3D
        Graphic type (``highdicom.sr.GraphicTypeValues3D.POLYGON``)
    graphic_data: Sequence[numpy.ndarray]
        Graphic data (3D spatial coordinates in slide coordinate system)
    frame_of_reference_uid: highdicom.UID
        Unique identifier of frame of reference (slide coordinate system)

    """
    if not is_tiled_image(image):
        raise ValueError('Argument "image" is not a a tiled image.')

    image_orientation = image.ImageOrientationSlide
    image_origin = image.TotalPixelMatrixOriginSequence[0]
    image_position = (
        image_origin.XOffsetInSlideCoordinateSystem,
        image_origin.YOffsetInSlideCoordinateSystem,
        0.0,  # TODO
    )
    pixel_spacing = (
        image.SharedFunctionalGroupsSequence[0]
        .PixelMeasuresSequence[0]
        .PixelSpacing
    )
    transformer = hd.spatial.ImageToReferenceTransformer(
        image_orientation=image_orientation,
        image_position=image_position,
        pixel_spacing=pixel_spacing,
    )

    num_focal_planes = getattr(image, "NumberOfFocalPlanes", 1)
    if num_focal_planes > 1:
        raise ValueError(
            "Images with multiple focal planes are not supported."
        )

    if "PerFrameFunctionalGroupsSequence" in image:
        plane_positions = [
            item.PlanePositionSequence
            for item in image.PerFrameFunctionalGroupsSequence
        ]
    else:
        plane_positions = hd.utils.compute_plane_position_slide_per_frame(
            image
        )

    rows = image.Rows
    cols = image.Columns
    contours = []
    for frame_index in range(int(image.NumberOfFrames)):
        frame_position = plane_positions[frame_index][0]
        r = frame_position.RowPositionInTotalImagePixelMatrix
        c = frame_position.ColumnPositionInTotalImagePixelMatrix

        frame_pixel_coordinates = np.array(
            [
                [c, r],
                [c + cols, r],
                [c + cols, r + rows],
                [c, r + rows],
                [c, r],
            ]
        )
        contours.append(transformer(frame_pixel_coordinates))

    return (
        hd.sr.GraphicTypeValues3D.POLYGON,
        contours,
        image.FrameOfReferenceUID,
    )
