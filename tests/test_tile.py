import itertools

import highdicom as hd
import numpy as np
import pytest
from pydicom.uid import ExplicitVRLittleEndian

from dicomslide.tile import (
    assemble_total_pixel_matrix,
    compute_frame_positions,
    disassemble_total_pixel_matrix,
    get_frame_contours,
)

from .dummy import VLWholeSlideMicroscopyImage


def test_assemble_disassemble_total_pixel_matrix():
    rows = 4
    columns = 6
    samples_per_pixel = 3
    dtype = np.uint8
    total_pixel_matrix_rows = 14
    total_pixel_matrix_columns = 20
    n_tile_rows = 4
    n_tile_columns = 4
    tiles = []
    tile_positions = []
    for i, (r, c) in enumerate(
        itertools.product(range(n_tile_rows), range(n_tile_columns))
    ):
        if r % 2 == 0 and c % 2 == 0:
            color = (255, 0, 0)
        elif r % 2 == 0 and c % 2 == 1:
            color = (0, 255, 0)
        elif r % 2 == 1 and c % 2 == 0:
            color = (0, 0, 255)
        else:
            color = (255, 255, 255)
        tile = np.ones((rows, columns, samples_per_pixel), dtype=dtype) * color
        tiles.append(tile.astype(dtype))
        position = (r * rows, c * columns)
        tile_positions.append(position)

    total_pixel_matrix = assemble_total_pixel_matrix(
        tiles=tiles,
        tile_positions=tile_positions,
        total_pixel_matrix_rows=total_pixel_matrix_rows,
        total_pixel_matrix_columns=total_pixel_matrix_columns
    )

    assert total_pixel_matrix.dtype == dtype
    assert total_pixel_matrix.shape == (
        total_pixel_matrix_rows,
        total_pixel_matrix_columns,
        samples_per_pixel
    )
    np.testing.assert_equal(
        total_pixel_matrix[0, 0, :],
        np.array([255, 0, 0]).astype(dtype)
    )
    np.testing.assert_equal(
        total_pixel_matrix[rows, columns, :],
        np.array([255, 255, 255]).astype(dtype)
    )

    retrieved_tiles = disassemble_total_pixel_matrix(
        total_pixel_matrix=total_pixel_matrix,
        tile_positions=tile_positions,
        rows=rows,
        columns=columns
    )

    assert len(retrieved_tiles) == len(tiles)
    for i, (r, c) in enumerate(
        itertools.product(range(n_tile_rows), range(n_tile_columns))
    ):
        if r < (n_tile_rows - 1) and c < (n_tile_columns - 1):
            np.testing.assert_array_equal(tiles[i], retrieved_tiles[i])


@pytest.mark.parametrize(
    'dimension_organization_type',
    [
        hd.DimensionOrganizationTypeValues.TILED_FULL,
        hd.DimensionOrganizationTypeValues.TILED_SPARSE,
    ]
)
@pytest.mark.parametrize(
    'mode',
    ['color', 'grayscale']
)
def test_get_frame_contours(dimension_organization_type, mode):
    if mode == 'color':
        kwargs = dict(
            number_of_focal_planes=1,
            number_of_optical_paths=1,
            samples_per_pixel=3,
        )
    else:
        kwargs = dict(
            number_of_focal_planes=1,
            number_of_optical_paths=1,
            samples_per_pixel=1,
        )
    image = VLWholeSlideMicroscopyImage(
        study_instance_uid=hd.UID(),
        series_instance_uid=hd.UID(),
        sop_instance_uid=hd.UID(),
        series_number=1,
        instance_number=1,
        extended_depth_of_field=False,
        image_position=(0.0, 0.0, 0.0),
        image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
        dimension_organization_type=dimension_organization_type,
        frame_of_reference_uid=hd.UID(),
        container_id='1',
        specimen_id='1',
        specimen_uid=hd.UID(),
        image_type=('DERIVED', 'PRIMARY', 'VOLUME', 'RESAMPLED'),
        total_pixel_matrix_rows=128,
        total_pixel_matrix_columns=64,
        rows=32,
        columns=32,
        pixel_spacing=(0.004, 0.004),
        spacing_between_slices=0.001,
        transfer_syntax_uid=ExplicitVRLittleEndian,
        **kwargs
    )
    contours = get_frame_contours(image)

    n_tile_rows = int(np.ceil(image.TotalPixelMatrixRows / image.Rows))
    n_tile_cols = int(np.ceil(image.TotalPixelMatrixColumns / image.Columns))

    assert len(contours) == int(image.NumberOfFrames)

    for i, (r, c) in enumerate(
        itertools.product(range(n_tile_rows), range(n_tile_cols))
    ):
        assert contours[i].dtype == np.float64
        assert contours[i].shape == (5, 3)
        np.testing.assert_array_equal(
            contours[i][0],
            contours[i][-1],
        )


@pytest.mark.parametrize(
    'dimension_organization_type',
    [
        hd.DimensionOrganizationTypeValues.TILED_FULL,
        hd.DimensionOrganizationTypeValues.TILED_SPARSE,
    ]
)
@pytest.mark.parametrize(
    'mode',
    ['color', 'grayscale']
)
def test_compute_frame_positions(dimension_organization_type, mode):
    if mode == 'color':
        kwargs = dict(
            number_of_focal_planes=1,
            number_of_optical_paths=1,
            samples_per_pixel=3,
        )
    else:
        kwargs = dict(
            number_of_focal_planes=4,
            number_of_optical_paths=3,
            samples_per_pixel=1,
        )
    image = VLWholeSlideMicroscopyImage(
        study_instance_uid=hd.UID(),
        series_instance_uid=hd.UID(),
        sop_instance_uid=hd.UID(),
        series_number=1,
        instance_number=1,
        extended_depth_of_field=False,
        image_position=(0.0, 0.0, 0.0),
        image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
        dimension_organization_type=dimension_organization_type,
        frame_of_reference_uid=hd.UID(),
        container_id='1',
        specimen_id='1',
        specimen_uid=hd.UID(),
        image_type=('DERIVED', 'PRIMARY', 'VOLUME', 'RESAMPLED'),
        total_pixel_matrix_rows=128,
        total_pixel_matrix_columns=64,
        rows=32,
        columns=32,
        pixel_spacing=(0.004, 0.004),
        spacing_between_slices=0.001,
        transfer_syntax_uid=ExplicitVRLittleEndian,
        **kwargs
    )

    (
        total_pixel_matrix_positions,
        slide_positions,
        optical_path_indices,
        focal_plane_indices,
    ) = compute_frame_positions(image)

    n = int(image.NumberOfFrames)
    assert total_pixel_matrix_positions.dtype == np.int64
    assert total_pixel_matrix_positions.shape == (n, 2)
    assert slide_positions.dtype == np.float64
    assert slide_positions.shape == (n, 3)
    assert optical_path_indices.dtype == np.int64
    assert optical_path_indices.shape == (n, )
    assert focal_plane_indices.dtype == np.int64
    assert focal_plane_indices.shape == (n, )
