from typing import Tuple

import highdicom as hd
import pytest
from pydicom.dataset import Dataset
from pydicom.uid import ExplicitVRLittleEndian

from dicomslide.pyramid import (
    Pyramid,
    sort_images_by_size,
    sort_images_by_pixel_spacing,
    sort_images_by_resolution,
)

from .dummy import VLWholeSlideMicroscopyImage


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
def test_pyramid(dimension_organization_type, mode):
    if mode == 'color':
        shared_kwargs = dict(
            number_of_focal_planes=1,
            number_of_optical_paths=1,
            optical_path_identifiers=['1'],
            samples_per_pixel=3,
            spacing_between_slices=0.001,
        )
    else:
        shared_kwargs = dict(
            number_of_focal_planes=3,
            number_of_optical_paths=2,
            optical_path_identifiers=['a', 'b'],
            samples_per_pixel=1,
            spacing_between_slices=0.001,
        )
    image_kwargs = (
        dict(
            image_type=('ORIGINAL', 'PRIMARY', 'VOLUME', 'NONE'),
            total_pixel_matrix_rows=512,
            total_pixel_matrix_columns=256,
            rows=32,
            columns=32,
            pixel_spacing=(0.001, 0.001),
        ),
        dict(
            image_type=('DERIVED', 'PRIMARY', 'VOLUME', 'RESAMPLED'),
            total_pixel_matrix_rows=256,
            total_pixel_matrix_columns=128,
            rows=32,
            columns=32,
            pixel_spacing=(0.002, 0.002),
        ),
        dict(
            image_type=('DERIVED', 'PRIMARY', 'VOLUME', 'RESAMPLED'),
            total_pixel_matrix_rows=128,
            total_pixel_matrix_columns=64,
            rows=32,
            columns=32,
            pixel_spacing=(0.004, 0.004),
        ),
        dict(
            image_type=('DERIVED', 'PRIMARY', 'THUMBNAIL', 'RESAMPLED'),
            total_pixel_matrix_rows=64,
            total_pixel_matrix_columns=32,
            rows=64,
            columns=32,
            pixel_spacing=(0.008, 0.008),
        ),
    )

    metadata = []
    for i, kwargs in enumerate(image_kwargs):
        image = VLWholeSlideMicroscopyImage(
            study_instance_uid=hd.UID(),
            series_instance_uid=hd.UID(),
            sop_instance_uid=hd.UID(),
            series_number=1,
            instance_number=i + 1,
            extended_depth_of_field=False,
            image_position=(0.0, 0.0, 0.0),
            image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            dimension_organization_type=dimension_organization_type,
            transfer_syntax_uid=ExplicitVRLittleEndian,
            frame_of_reference_uid=hd.UID(),
            container_id='1',
            specimen_id='1',
            specimen_uid=hd.UID(),
            **kwargs,
            **shared_kwargs
        )
        metadata.append(image)

    num_expected_levels = len(image_kwargs)
    expected_downsampling_factors = (
        (1.0, 1.0),
        (2.0, 2.0),
        (4.0, 4.0),
        (8.0, 8.0),
    )
    pyramid = Pyramid(metadata, tolerance=0.01)

    assert len(pyramid) == num_expected_levels
    for i in range(num_expected_levels):
        assert pyramid[i].total_pixel_matrix_dimensions == (
            image_kwargs[i]['total_pixel_matrix_rows'],
            image_kwargs[i]['total_pixel_matrix_columns'],
        )
        assert pyramid[i].pixel_spacing == (
            image_kwargs[i]['pixel_spacing'][1],
            image_kwargs[i]['pixel_spacing'][0],
        )
        assert pyramid[i].downsampling_factors == (
            expected_downsampling_factors[i][0],
            expected_downsampling_factors[i][1],
        )
        assert pyramid[i].imaged_volume_dimensions[:2] == (
            (
                image_kwargs[i]['total_pixel_matrix_columns'] *
                image_kwargs[i]['pixel_spacing'][0]
            ),
            (
                image_kwargs[i]['total_pixel_matrix_rows'] *
                image_kwargs[i]['pixel_spacing'][1]
            ),
        )


def test_sort_images():
    def generate_image_metadata(
        total_pixel_matrix_rows: int,
        total_pixel_matrix_columns: int,
        pixel_spacing: Tuple[float, float]
    ):
        dataset = Dataset()
        dataset.TotalPixelMatrixRows = total_pixel_matrix_rows
        dataset.TotalPixelMatrixColumns = total_pixel_matrix_columns
        pixel_measures_item = Dataset()
        pixel_measures_item.PixelSpacing = [pixel_spacing[0], pixel_spacing[1]]
        sfg_item = Dataset()
        sfg_item.PixelMeasuresSequence = [pixel_measures_item]
        dataset.SharedFunctionalGroupsSequence = [sfg_item]
        return dataset

    metadata = [
        generate_image_metadata(1000, 1000, (0.5, 0.5)),
        generate_image_metadata(500, 500, (0.8, 0.8)),
        generate_image_metadata(750, 350, (0.2, 0.2)),
    ]

    expected_size_sort_index = (0, 2, 1)
    metadata_sorted_by_size = sort_images_by_size(metadata)
    for i, j in enumerate(expected_size_sort_index):
        assert metadata[j] == metadata_sorted_by_size[i]

    expected_pixel_spacing_sort_index = (2, 0, 1)
    metadata_sorted_by_pixel_spacing = sort_images_by_pixel_spacing(metadata)
    for i, j in enumerate(expected_pixel_spacing_sort_index):
        assert metadata[j] == metadata_sorted_by_pixel_spacing[i]

    expected_resolution_sort_index = (1, 0, 2)
    metadata_sorted_by_resolution = sort_images_by_resolution(metadata)
    for i, j in enumerate(expected_resolution_sort_index):
        assert metadata[j] == metadata_sorted_by_resolution[i]
