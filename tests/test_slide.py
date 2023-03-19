import collections
import uuid
from typing import List, Mapping, Tuple

import highdicom as hd
import numpy as np
import pytest
from PIL import Image, ImageChops
from pydicom.sr.codedict import codes
from pydicom.uid import JPEGBaseline8Bit, JPEG2000Lossless

from dicomslide.slide import find_slides, Slide
from dicomslide.openslide import (
    OpenSlide,
    OPENSLIDE_MPP_X,
    OPENSLIDE_MPP_Y,
)

from .dummy import VLWholeSlideMicroscopyImage


def generate_test_images(
    number_of_optical_paths: int,
    number_of_focal_planes: int,
    samples_per_pixel: int,
    image_orientation: Tuple[float, float, float, float, float, float],
    transfer_syntax_uid: str,
    dimension_organization_type: hd.DimensionOrganizationTypeValues
) -> Mapping[Tuple[str, str], List[VLWholeSlideMicroscopyImage]]:
    num_studies = 4
    num_series_per_study = 2
    lut = collections.defaultdict(list)
    for _ in range(num_studies):
        study_instance_uid = hd.UID()
        for i in range(num_series_per_study):
            series_instance_uid = hd.UID()
            frame_of_reference_uid = hd.UID()
            container_id = str(uuid.uuid4())
            specimen_id = str(uuid.uuid4())
            specimen_uid = hd.UID()
            optical_path_identifiers = [
                str(i + 1) for i in range(number_of_optical_paths)
            ]
            image_kwargs = (
                dict(
                    image_type=('ORIGINAL', 'PRIMARY', 'VOLUME', 'NONE'),
                    total_pixel_matrix_rows=512,
                    total_pixel_matrix_columns=256,
                    rows=32,
                    columns=32,
                    pixel_spacing=(0.001, 0.001),
                    spacing_between_slices=0.001,
                    number_of_focal_planes=number_of_focal_planes,
                    number_of_optical_paths=number_of_optical_paths,
                    optical_path_identifiers=optical_path_identifiers,
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
                    spacing_between_slices=0.001,
                    number_of_focal_planes=number_of_focal_planes,
                    number_of_optical_paths=number_of_optical_paths,
                    optical_path_identifiers=optical_path_identifiers,
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
                    spacing_between_slices=0.001,
                    number_of_focal_planes=number_of_focal_planes,
                    number_of_optical_paths=number_of_optical_paths,
                    optical_path_identifiers=optical_path_identifiers,
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
                    spacing_between_slices=0.001,
                    number_of_focal_planes=number_of_focal_planes,
                    number_of_optical_paths=number_of_optical_paths,
                    optical_path_identifiers=optical_path_identifiers,
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
                    optical_path_identifiers=['1'],
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
                    optical_path_identifiers=['1'],
                    samples_per_pixel=3,
                    transfer_syntax_uid=JPEGBaseline8Bit
                ),
            )
            image_position = (0.0, 0.0, 0.0)
            for j, kwargs in enumerate(image_kwargs):
                image = VLWholeSlideMicroscopyImage(
                    study_instance_uid=study_instance_uid,
                    series_instance_uid=series_instance_uid,
                    sop_instance_uid=hd.UID(),
                    series_number=i + 1,
                    instance_number=j + 1,
                    extended_depth_of_field=False,
                    # TODO: image position needs to be slightly adjusted
                    image_position=image_position,
                    image_orientation=image_orientation,
                    dimension_organization_type=dimension_organization_type,
                    frame_of_reference_uid=frame_of_reference_uid,
                    container_id=container_id,
                    specimen_id=specimen_id,
                    specimen_uid=specimen_uid,
                    **kwargs
                )
                key = (study_instance_uid, container_id, frame_of_reference_uid)
                lut[key].append(image)
    return lut


@pytest.mark.parametrize(
    'dimension_organization_type',
    [
        hd.DimensionOrganizationTypeValues.TILED_FULL,
        hd.DimensionOrganizationTypeValues.TILED_SPARSE,
    ]
)
@pytest.mark.parametrize(
    'image_orientation',
    [
        (0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0, -1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    ]
)
def test_color_images(client, dimension_organization_type, image_orientation):
    expected_num_optical_paths = 1
    expected_num_focal_planes = 1
    expected_samples_per_pixel = 3
    groups = generate_test_images(
        number_of_optical_paths=expected_num_optical_paths,
        number_of_focal_planes=expected_num_focal_planes,
        samples_per_pixel=expected_samples_per_pixel,
        image_orientation=image_orientation,
        transfer_syntax_uid=JPEGBaseline8Bit,
        dimension_organization_type=dimension_organization_type
    )

    for datasets in groups.values():
        client.store_instances(datasets)

    found_slides = find_slides(
        client,
        study_instance_uid=datasets[0].StudyInstanceUID
    )
    assert len(found_slides) == 2
    found_slides = find_slides(
        client,
        study_instance_uid='1.2.3.4'
    )
    assert len(found_slides) == 0

    found_slides = find_slides(
        client,
        container_id=datasets[0].ContainerIdentifier
    )
    assert len(found_slides) == 1
    found_slides = find_slides(
        client,
        container_id='foo'
    )
    assert len(found_slides) == 0

    found_slides = find_slides(client)
    assert len(found_slides) == len(groups)

    expected_num_levels = 4
    expected_downsampling_factors = (1.0, 2.0, 4.0, 8.0)

    for slide in found_slides:
        assert isinstance(slide, Slide)
        assert slide.num_channels == expected_num_optical_paths
        assert slide.num_focal_planes == expected_num_focal_planes
        assert slide.num_levels == expected_num_levels
        assert slide.downsampling_factors == expected_downsampling_factors
        volume_images = slide.get_volume_images()
        assert len(volume_images) == expected_num_levels
        expected_sizes = [
            (
                image.metadata.TotalPixelMatrixRows,
                image.metadata.TotalPixelMatrixColumns,
            )
            for image in volume_images
        ]
        assert slide.size == expected_sizes[0]
        assert set(
            round(slide.physical_size[0], 3),
            round(slide.physical_size[1], 3),
        ) == set(
            round(float(volume_images[0].metadata.ImagedVolumeHeight), 3),
            round(float(volume_images[0].metadata.ImagedVolumeWidth), 3),
        )
        assert len(slide.label_images) == 1
        assert len(slide.overview_images) == 1

        assert len(slide.find_optical_paths()) == expected_num_optical_paths
        assert len(slide.find_optical_paths(identifier='1')) == 1
        assert len(slide.find_optical_paths(identifier='2')) == 0
        assert len(slide.find_segments()) == 0

        np.testing.assert_array_equal(
            slide.get_image_region(
                offset=(0, 0),
                level=0,
                size=(52, 100),
                channel_index=0,
                focal_plane_index=0
            ),
            np.ones((52, 100, expected_samples_per_pixel), dtype=np.uint8) * 255
        )
        np.testing.assert_array_equal(
            slide.get_image_region(
                offset=(5, 10),
                level=2,
                size=(28, 37),
                channel_index=0,
                focal_plane_index=0
            ),
            np.ones((28, 37, expected_samples_per_pixel), dtype=np.uint8) * 255
        )

        np.testing.assert_array_equal(
            slide.get_slide_region(
                offset=(0.0, 0.0),
                level=3,
                size=(0.5, 0.25),
                channel_index=0,
                focal_plane_index=0
            ),
            np.ones((31, 62, expected_samples_per_pixel), dtype=np.uint8) * 255
        )
        np.testing.assert_array_equal(
            slide.get_slide_region(
                offset=(10.0, 20.0),
                level=2,
                size=(0.05, 0.03),
                channel_index=0,
                focal_plane_index=0
            ),
            np.ones((8, 12, expected_samples_per_pixel), dtype=np.uint8) * 255
        )

        np.testing.assert_array_equal(
            slide.get_slide_region_for_annotation(
                annotation=hd.sr.Scoord3DContentItem(
                    name=codes.DCM.ImageRegion,
                    graphic_type=hd.sr.GraphicTypeValues3D.POLYGON,
                    graphic_data=np.array([
                        (10.0, 20.0, 0.0),
                        (10.05, 20.0, 0.0),
                        (10.05, 20.03, 0.0),
                        (10.0, 20.03, 0.0),
                        (10.0, 20.0, 0.0),
                    ]),
                    frame_of_reference_uid=(
                        volume_images[0]
                        .metadata
                        .FrameOfReferenceUID
                    )
                ),
                level=2,
                channel_index=0
            ),
            np.ones((8, 12, expected_samples_per_pixel), dtype=np.uint8) * 255
        )
        np.testing.assert_array_equal(
            slide.get_slide_region_for_annotation(
                annotation=hd.sr.Scoord3DContentItem(
                    name=codes.DCM.ImageRegion,
                    graphic_type=hd.sr.GraphicTypeValues3D.POLYGON,
                    graphic_data=np.array([
                        (10.0, 20.0, 0.0),
                        (10.05, 20.0, 0.0),
                        (10.05, 20.03, 0.0),
                        (10.0, 20.03, 0.0),
                        (10.0, 20.0, 0.0),
                    ]),
                    frame_of_reference_uid=(
                        volume_images[0]
                        .metadata
                        .FrameOfReferenceUID
                    )
                ),
                level=2,
                channel_index=0,
                padding=(0.05, 0.01)
            ),
            np.ones((13, 37, expected_samples_per_pixel), dtype=np.uint8) * 255
        )
        np.testing.assert_array_equal(
            slide.get_slide_region_for_annotation(
                annotation=hd.sr.Scoord3DContentItem(
                    name=codes.DCM.ImageRegion,
                    graphic_type=hd.sr.GraphicTypeValues3D.ELLIPSE,
                    graphic_data=np.array([
                        (10.0, 20.015, 0.0),
                        (10.05, 20.015, 0.0),
                        (10.025, 20.0, 0.0),
                        (10.025, 20.03, 0.0),
                    ]),
                    frame_of_reference_uid=(
                        volume_images[0]
                        .metadata
                        .FrameOfReferenceUID
                    )
                ),
                level=3,
                channel_index=0
            ),
            np.ones((4, 6, expected_samples_per_pixel), dtype=np.uint8) * 255
        )
        with pytest.raises(ValueError):
            slide.get_slide_region_for_annotation(
                annotation=hd.sr.Scoord3DContentItem(
                    name=codes.DCM.ImageRegion,
                    graphic_type=hd.sr.GraphicTypeValues3D.POLYGON,
                    graphic_data=np.array([
                        (10.0, 20.0, 0.0),
                        (10.05, 20.0, 0.0),
                        (10.05, 20.03, 0.0),
                        (10.0, 20.03, 0.0),
                        (10.0, 20.0, 0.0),
                    ]),
                    frame_of_reference_uid='1.2.3.4'
                ),
                level=2,
                channel_index=0
            )

        openslide = OpenSlide(slide)
        assert openslide.level_count == expected_num_levels
        assert openslide.dimensions == (
            expected_sizes[0][1],
            expected_sizes[0][0],
        )
        assert openslide.level_dimensions == tuple([
            (dimensions[1], dimensions[0])
            for dimensions in expected_sizes
        ])
        assert openslide.level_downsamples == expected_downsampling_factors
        assert len(openslide.associated_images) == 2
        label_image = openslide.associated_images['LABEL']
        assert isinstance(label_image, Image.Image)
        assert label_image.mode == 'RGBA'
        overview_image = openslide.associated_images['OVERVIEW']
        assert isinstance(overview_image, Image.Image)
        assert overview_image.mode == 'RGBA'
        expected_mpp_x = slide.pixel_spacings[0][0] * 10**3
        expected_mpp_y = slide.pixel_spacings[0][1] * 10**3
        assert openslide.properties[OPENSLIDE_MPP_X] == str(expected_mpp_x)
        assert openslide.properties[OPENSLIDE_MPP_Y] == str(expected_mpp_y)

        thumbnail_image = openslide.get_thumbnail(size=(10, 10))
        assert isinstance(thumbnail_image, Image.Image)
        assert thumbnail_image.mode == 'RGB'

        image_region = openslide.read_region(
            location=(0, 0),
            level=0,
            size=(100, 52)
        )
        assert isinstance(image_region, Image.Image)
        assert image_region.mode == 'RGBA'
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


@pytest.mark.parametrize(
    'image_orientation',
    [
        (0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0, -1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    ]
)
def test_grayscale_images(client, image_orientation):
    expected_num_optical_paths = 2
    expected_num_focal_planes = 3
    expected_samples_per_pixel = 1
    groups = generate_test_images(
        number_of_optical_paths=expected_num_optical_paths,
        number_of_focal_planes=expected_num_focal_planes,
        samples_per_pixel=expected_samples_per_pixel,
        image_orientation=image_orientation,
        transfer_syntax_uid=JPEG2000Lossless,
        dimension_organization_type=(
            hd.DimensionOrganizationTypeValues.TILED_SPARSE
        )
    )

    for datasets in groups.values():
        client.store_instances(datasets)

    group_keys = list(groups.keys())

    found_slides = find_slides(
        client,
        study_instance_uid=group_keys[0][0]
    )
    assert len(found_slides) == 2
    assert found_slides[0].num_channels == expected_num_optical_paths

    found_slides = find_slides(
        client,
        study_instance_uid=group_keys[0][0],
        container_id=group_keys[0][1]
    )
    assert len(found_slides) == 1
    assert found_slides[0].num_channels == expected_num_optical_paths

    found_slides = find_slides(
        client,
        study_instance_uid=group_keys[0][0],
        container_id=group_keys[2][1]
    )
    assert len(found_slides) == 0

    found_slides = find_slides(client)
    assert len(found_slides) == len(groups)
    for i in range(len(groups)):
        assert found_slides[i].num_channels == expected_num_optical_paths

    expected_num_levels = 4
    expected_downsampling_factors = (1.0, 2.0, 4.0, 8.0)

    for slide in found_slides:
        assert isinstance(slide, Slide)
        assert slide.num_channels == expected_num_optical_paths
        assert slide.num_focal_planes == expected_num_focal_planes
        assert slide.num_levels == expected_num_levels
        assert slide.downsampling_factors == expected_downsampling_factors
        assert len(slide.find_optical_paths()) == expected_num_optical_paths
        assert len(slide.find_optical_paths(identifier='1')) == 1
        assert len(slide.find_segments()) == 0

        for channel_index in range(expected_num_optical_paths):
            for focal_plane_index in range(expected_num_focal_planes):
                volume_images = slide.get_volume_images(
                    channel_index=channel_index,
                    focal_plane_index=focal_plane_index
                )
                assert len(volume_images) == expected_num_levels
                assert slide.total_pixel_matrix_dimensions == tuple([
                    (
                        image.metadata.TotalPixelMatrixRows,
                        image.metadata.TotalPixelMatrixColumns,
                    )
                    for image in volume_images
                ])
                assert len(slide.label_images) == 1
                assert len(slide.overview_images) == 1
                np.testing.assert_array_equal(
                    slide.get_image_region(
                        offset=(0, 0),
                        level=0,
                        size=(52, 100),
                        channel_index=channel_index,
                        focal_plane_index=focal_plane_index
                    ),
                    np.zeros(
                        (52, 100, expected_samples_per_pixel),
                        dtype=np.uint16
                    )
                )
                np.testing.assert_array_equal(
                    slide.get_image_region(
                        offset=(5, 10),
                        level=2,
                        size=(28, 37),
                        channel_index=channel_index,
                        focal_plane_index=focal_plane_index
                    ),
                    np.zeros(
                        (28, 37, expected_samples_per_pixel),
                        dtype=np.uint16
                    )
                )

                np.testing.assert_array_equal(
                    slide.get_slide_region(
                        offset=(0.0, 0.0),
                        level=3,
                        size=(0.5, 0.25),
                        channel_index=0,
                        focal_plane_index=0
                    ),
                    np.zeros(
                        (31, 62, expected_samples_per_pixel),
                        dtype=np.uint8
                    )
                )
                np.testing.assert_array_equal(
                    slide.get_slide_region(
                        offset=(10.0, 20.0),
                        level=2,
                        size=(0.05, 0.03),
                        channel_index=0,
                        focal_plane_index=1
                    ),
                    np.zeros(
                        (8, 12, expected_samples_per_pixel),
                        dtype=np.uint8
                    )
                )
