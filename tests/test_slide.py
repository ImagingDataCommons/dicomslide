import collections
from typing import List, Mapping, Tuple

import highdicom as hd
import numpy as np
import pytest
from PIL import Image, ImageChops
from pydicom.uid import JPEGBaseline8Bit, JPEG2000Lossless

from dicomslide.slide import find_slides, Slide
from dicomslide.openslide import (
    OpenSlide,
    OPENSLIDE_MPP_X,
    OPENSLIDE_MPP_Y,
)

from .dummy import VLWholeSlideMicroscopyImage

TILED_FULL = hd.DimensionOrganizationTypeValues.TILED_FULL
TILED_SPARSE = hd.DimensionOrganizationTypeValues.TILED_SPARSE


def generate_test_images(
    number_of_optical_paths: int,
    number_of_focal_planes: int,
    samples_per_pixel: int,
    transfer_syntax_uid: str,
    dimension_organization_type: hd.DimensionOrganizationTypeValues
) -> Mapping[Tuple[str, str], List[VLWholeSlideMicroscopyImage]]:
    lut = collections.defaultdict(list)
    for _ in range(4):
        study_instance_uid = hd.UID()
        for i in range(2):
            series_instance_uid = hd.UID()
            frame_of_reference_uid = hd.UID()
            container_id = str(np.random.choice(range(100)))
            specimen_id = str(np.random.choice(range(100)))
            specimen_uid = hd.UID()
            image_kwargs = (
                dict(
                    image_type=('ORIGINAL', 'PRIMARY', 'VOLUME', 'NONE'),
                    total_pixel_matrix_rows=512,
                    total_pixel_matrix_columns=256,
                    rows=32,
                    columns=32,
                    pixel_spacing=(0.001, 0.001),
                    number_of_focal_planes=number_of_focal_planes,
                    number_of_optical_paths=number_of_optical_paths,
                    samples_per_pixel=samples_per_pixel,
                    transfer_syntax_uid=transfer_syntax_uid
                ),
                dict(
                    image_type=('DERIVED', 'PRIMARY', 'VOLUME', 'RESAMPLED'),
                    total_pixel_matrix_rows=256,
                    total_pixel_matrix_columns=128,
                    rows=32,
                    columns=32,
                    pixel_spacing=(0.002, 0.002),
                    number_of_focal_planes=number_of_focal_planes,
                    number_of_optical_paths=number_of_optical_paths,
                    samples_per_pixel=samples_per_pixel,
                    transfer_syntax_uid=transfer_syntax_uid
                ),
                dict(
                    image_type=('DERIVED', 'PRIMARY', 'VOLUME', 'RESAMPLED'),
                    total_pixel_matrix_rows=128,
                    total_pixel_matrix_columns=64,
                    rows=32,
                    columns=32,
                    pixel_spacing=(0.004, 0.004),
                    number_of_focal_planes=number_of_focal_planes,
                    number_of_optical_paths=number_of_optical_paths,
                    samples_per_pixel=samples_per_pixel,
                    transfer_syntax_uid=transfer_syntax_uid
                ),
                dict(
                    image_type=('DERIVED', 'PRIMARY', 'THUMBNAIL', 'RESAMPLED'),
                    total_pixel_matrix_rows=64,
                    total_pixel_matrix_columns=32,
                    rows=64,
                    columns=32,
                    pixel_spacing=(0.008, 0.008),
                    number_of_focal_planes=number_of_focal_planes,
                    number_of_optical_paths=number_of_optical_paths,
                    samples_per_pixel=samples_per_pixel,
                    transfer_syntax_uid=transfer_syntax_uid
                ),
                dict(
                    image_type=('ORIGINAL', 'PRIMARY', 'LABEL', 'NONE'),
                    total_pixel_matrix_rows=50,
                    total_pixel_matrix_columns=50,
                    rows=50,
                    columns=50,
                    pixel_spacing=(0.003, 0.003),
                    number_of_focal_planes=1,
                    number_of_optical_paths=1,
                    samples_per_pixel=3,
                    transfer_syntax_uid=JPEGBaseline8Bit
                ),
                dict(
                    image_type=('ORIGINAL', 'PRIMARY', 'OVERVIEW', 'NONE'),
                    total_pixel_matrix_rows=50,
                    total_pixel_matrix_columns=150,
                    rows=50,
                    columns=150,
                    pixel_spacing=(0.003, 0.003),
                    number_of_focal_planes=1,
                    number_of_optical_paths=1,
                    samples_per_pixel=3,
                    transfer_syntax_uid=JPEGBaseline8Bit
                ),
            )
            for j, kwargs in enumerate(image_kwargs):
                image = VLWholeSlideMicroscopyImage(
                    study_instance_uid=study_instance_uid,
                    series_instance_uid=series_instance_uid,
                    sop_instance_uid=hd.UID(),
                    series_number=i + 1,
                    instance_number=j + 1,
                    extended_depth_of_field=False,
                    # TODO: image position needs to be slightly adjusted
                    image_position=(0.0, 0.0, 0.0),
                    image_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
                    dimension_organization_type=dimension_organization_type,
                    frame_of_reference_uid=frame_of_reference_uid,
                    container_id=container_id,
                    specimen_id=specimen_id,
                    specimen_uid=specimen_uid,
                    **kwargs
                )
                lut[(container_id, frame_of_reference_uid)].append(image)
    return lut


@pytest.mark.parametrize(
    'dimension_organization_type',
    [TILED_FULL, TILED_SPARSE]
)
def test_color_images(client, dimension_organization_type):
    expected_num_optical_paths = 1
    expected_num_focal_planes = 1
    expected_samples_per_pixel = 3
    groups = generate_test_images(
        number_of_optical_paths=expected_num_optical_paths,
        number_of_focal_planes=expected_num_focal_planes,
        samples_per_pixel=expected_samples_per_pixel,
        transfer_syntax_uid=JPEGBaseline8Bit,
        dimension_organization_type=dimension_organization_type
    )

    for datasets in groups.values():
        client.store_instances(datasets)

    found_slides = find_slides(client)

    expected_num_levels = 4
    expected_downsampling_factors = (1.0, 2.0, 4.0, 8.0)

    assert len(found_slides) == len(groups)
    for slide in found_slides:
        assert isinstance(slide, Slide)
        assert slide.num_optical_paths == expected_num_optical_paths
        assert slide.num_focal_planes == expected_num_focal_planes
        assert slide.num_levels == expected_num_levels
        assert slide.downsampling_factors == expected_downsampling_factors
        volume_images = slide.get_volume_images()
        assert len(volume_images) == expected_num_levels
        expected_dimensions = tuple([
            (
                image.metadata.TotalPixelMatrixColumns,
                image.metadata.TotalPixelMatrixRows,
            )
            for image in volume_images
        ])
        assert slide.total_pixel_matrix_dimensions == expected_dimensions
        assert slide.imaged_volume_dimensions == tuple([
            (
                image.metadata.ImagedVolumeWidth,
                image.metadata.ImagedVolumeHeight,
                image.metadata.ImagedVolumeDepth,
            )
            for image in volume_images
        ])
        assert len(slide.label_images) == 1
        assert len(slide.overview_images) == 1
        np.testing.assert_array_equal(
            slide.get_image_region(
                pixel_indices=(0, 0),
                level=0,
                size=(100, 52),
                optical_path_index=1,
                focal_plane_index=1
            ),
            np.ones((52, 100, expected_samples_per_pixel), dtype=np.uint8) * 255
        )
        np.testing.assert_array_equal(
            slide.get_image_region(
                pixel_indices=(10, 5),
                level=2,
                size=(37, 28),
                optical_path_index=1,
                focal_plane_index=1
            ),
            np.ones((28, 37, expected_samples_per_pixel), dtype=np.uint8) * 255
        )

        openslide = OpenSlide(slide)
        assert openslide.level_count == expected_num_levels
        assert openslide.dimensions == expected_dimensions[0]
        assert openslide.level_dimensions == expected_dimensions
        assert openslide.level_downsamples == expected_downsampling_factors
        assert len(openslide.associated_images) == 2
        expected_mpp_x = slide.pixel_spacings[0][0] * 10**3
        expected_mpp_y = slide.pixel_spacings[0][1] * 10**3
        assert openslide.properties[OPENSLIDE_MPP_X] == str(expected_mpp_x)
        assert openslide.properties[OPENSLIDE_MPP_Y] == str(expected_mpp_y)

        image_region = openslide.read_region(
            location=(0, 0),
            level=0,
            size=(100, 52)
        )
        expected_image_region = Image.new(
            mode='RGBA',
            size=(100, 52),
            color=(255, 255, 255)
        )
        diff = ImageChops.difference(image_region, expected_image_region)
        assert not diff.getbbox()

        image_region = openslide.read_region(
            location=(10, 5),
            level=2,
            size=(37, 28),
        )
        expected_image_region = Image.new(
            mode='RGBA',
            size=(37, 28),
            color=(255, 255, 255)
        )
        diff = ImageChops.difference(image_region, expected_image_region)
        assert not diff.getbbox()


def test_grayscale_images(client):
    expected_num_optical_paths = 2
    expected_num_focal_planes = 5
    expected_samples_per_pixel = 1
    groups = generate_test_images(
        number_of_optical_paths=expected_num_optical_paths,
        number_of_focal_planes=expected_num_focal_planes,
        samples_per_pixel=expected_samples_per_pixel,
        transfer_syntax_uid=JPEG2000Lossless,
        dimension_organization_type=TILED_SPARSE
    )

    for datasets in groups.values():
        client.store_instances(datasets)

    found_slides = find_slides(client)

    expected_num_levels = 4
    expected_downsampling_factors = (1.0, 2.0, 4.0, 8.0)

    assert len(found_slides) == len(groups)
    for slide in found_slides:
        assert isinstance(slide, Slide)
        assert slide.num_optical_paths == expected_num_optical_paths
        assert slide.num_focal_planes == expected_num_focal_planes
        assert slide.num_levels == expected_num_levels
        assert slide.downsampling_factors == expected_downsampling_factors
        for optical_path_index in range(1, expected_num_optical_paths + 1):
            for focal_plane_index in range(1, expected_num_focal_planes + 1):
                volume_images = slide.get_volume_images(
                    optical_path_index=optical_path_index,
                    focal_plane_index=focal_plane_index
                )
                assert len(volume_images) == expected_num_levels
                assert slide.total_pixel_matrix_dimensions == tuple([
                    (
                        image.metadata.TotalPixelMatrixColumns,
                        image.metadata.TotalPixelMatrixRows,
                    )
                    for image in volume_images
                ])
                assert slide.imaged_volume_dimensions == tuple([
                    (
                        image.metadata.ImagedVolumeWidth,
                        image.metadata.ImagedVolumeHeight,
                        image.metadata.ImagedVolumeDepth,
                    )
                    for image in volume_images
                ])
                assert len(slide.label_images) == 1
                assert len(slide.overview_images) == 1
                np.testing.assert_array_equal(
                    slide.get_image_region(
                        pixel_indices=(0, 0),
                        level=0,
                        size=(100, 52),
                        optical_path_index=optical_path_index,
                        focal_plane_index=focal_plane_index
                    ),
                    np.zeros(
                        (52, 100, expected_samples_per_pixel),
                        dtype=np.uint16
                    )
                )
                np.testing.assert_array_equal(
                    slide.get_image_region(
                        pixel_indices=(10, 5),
                        level=2,
                        size=(37, 28),
                        optical_path_index=optical_path_index,
                        focal_plane_index=focal_plane_index
                    ),
                    np.zeros(
                        (28, 37, expected_samples_per_pixel),
                        dtype=np.uint16
                    )
                )
