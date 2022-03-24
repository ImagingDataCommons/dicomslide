import highdicom as hd
import numpy as np
from pydicom.uid import JPEGBaseline8Bit, JPEG2000Lossless

from dicomslide.matrix import TotalPixelMatrix

from .dummy import VLWholeSlideMicroscopyImage

TILED_FULL = hd.DimensionOrganizationTypeValues.TILED_FULL


def test_color(client):
    image = VLWholeSlideMicroscopyImage(
        study_instance_uid=hd.UID(),
        series_instance_uid=hd.UID(),
        sop_instance_uid=hd.UID(),
        series_number=1,
        instance_number=1,
        rows=4,
        columns=6,
        total_pixel_matrix_rows=30,
        total_pixel_matrix_columns=28,
        number_of_focal_planes=1,
        number_of_optical_paths=1,
        samples_per_pixel=3,
        image_type=('ORIGINAL', 'PRIMARY', 'VOLUME', 'NONE'),
        extended_depth_of_field=False,
        pixel_spacing=(0.001, 0.001),
        image_position=(0.0, 0.0, 0.0),
        image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
        dimension_organization_type=TILED_FULL,
        transfer_syntax_uid=JPEGBaseline8Bit,
        frame_of_reference_uid=hd.UID(),
        container_id='1',
        specimen_id='1',
        specimen_uid=hd.UID()
    )
    client.store_instances(datasets=[image])

    matrix = TotalPixelMatrix(
        client=client,
        image_metadata=image,
        channel_index=0,
        focal_plane_index=0,
    )
    assert len(matrix) == int(image.NumberOfFrames)
    assert matrix.ndim == 3
    assert matrix.dtype == np.dtype('uint8')
    assert matrix.shape == (
        image.TotalPixelMatrixRows,
        image.TotalPixelMatrixColumns,
        image.SamplesPerPixel
    )
    array = matrix[:, :, :]
    assert array.ndim == 3
    assert array.dtype == np.dtype('uint8')
    assert array.shape == (
        image.TotalPixelMatrixRows,
        image.TotalPixelMatrixColumns,
        image.SamplesPerPixel
    )
    array = matrix[3:23, 5:15, :]
    assert array.shape == (20, 10, image.SamplesPerPixel)
    array = matrix[:256, 256:512, 0]
    assert array.shape == (28, 0, 1)

    offset, size = matrix.get_tile_bounding_box(0)
    assert offset == (0, 0)
    assert size == (image.Rows, image.Columns)

    offset, size = matrix.get_tile_bounding_box(1)
    assert offset == (0, image.Columns)
    assert size == (image.Rows, image.Columns)

    for tile in iter(matrix):
        assert tile.shape == matrix.tile_shape
        assert tile.dtype == matrix.dtype


def test_monochrome_multiple_optical_paths_and_multiple_focal_planes(client):
    image = VLWholeSlideMicroscopyImage(
        study_instance_uid=hd.UID(),
        series_instance_uid=hd.UID(),
        sop_instance_uid=hd.UID(),
        series_number=2,
        instance_number=2,
        rows=4,
        columns=6,
        total_pixel_matrix_rows=30,
        total_pixel_matrix_columns=28,
        number_of_focal_planes=10,
        number_of_optical_paths=3,
        samples_per_pixel=1,
        image_type=('ORIGINAL', 'PRIMARY', 'VOLUME', 'NONE'),
        extended_depth_of_field=False,
        pixel_spacing=(0.001, 0.001),
        image_position=(0.0, 0.0, 0.0),
        image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
        dimension_organization_type=TILED_FULL,
        transfer_syntax_uid=JPEG2000Lossless,
        frame_of_reference_uid=hd.UID(),
        container_id='1',
        specimen_id='1',
        specimen_uid=hd.UID()
    )
    client.store_instances(datasets=[image])

    matrix = TotalPixelMatrix(
        client=client,
        image_metadata=image,
        channel_index=0,
        focal_plane_index=0,
    )
    assert len(matrix) == int(
        np.ceil(image.TotalPixelMatrixRows / image.Rows) *
        np.ceil(image.TotalPixelMatrixColumns / image.Columns)
    )
    assert matrix.ndim == 3
    assert matrix.dtype == np.dtype('uint16')
    assert matrix.shape == (
        image.TotalPixelMatrixRows,
        image.TotalPixelMatrixColumns,
        image.SamplesPerPixel
    )
    assert matrix.tile_shape == (
        image.Rows,
        image.Columns,
        image.SamplesPerPixel
    )
    array = matrix[:, :, :]
    assert array.ndim == 3
    assert array.dtype == np.dtype('uint16')
    assert array.shape == (
        image.TotalPixelMatrixRows,
        image.TotalPixelMatrixColumns,
        image.SamplesPerPixel
    )
    array = matrix[3:23, 5:15, :]
    assert array.shape == (20, 10, image.SamplesPerPixel)
    array = matrix[0:10, 2:10, 0]
    assert array.shape == (10, 8, 1)

    matrix = TotalPixelMatrix(
        client=client,
        image_metadata=image,
        channel_index=0,
        focal_plane_index=3,
    )
    assert matrix.ndim == 3
    assert matrix.dtype == np.dtype('uint16')
    assert matrix.shape == (
        image.TotalPixelMatrixRows,
        image.TotalPixelMatrixColumns,
        image.SamplesPerPixel
    )
    assert matrix.tile_shape == (
        image.Rows,
        image.Columns,
        image.SamplesPerPixel
    )
    array = matrix[:, :, :]
    assert array.shape == (
        image.TotalPixelMatrixRows,
        image.TotalPixelMatrixColumns,
        image.SamplesPerPixel
    )
    array = matrix[3:23, 5:15, :]
    assert array.shape == (20, 10, image.SamplesPerPixel)
    array = matrix[0:10, 2:10, 0]
    assert array.shape == (10, 8, 1)

    offset, size = matrix.get_tile_bounding_box(0)
    assert offset == (0, 0)
    assert size == (image.Rows, image.Columns)

    offset, size = matrix.get_tile_bounding_box(1)
    assert offset == (0, image.Columns)
    assert size == (image.Rows, image.Columns)

    for tile in iter(matrix):
        assert tile.shape == matrix.tile_shape
        assert tile.dtype == matrix.dtype
