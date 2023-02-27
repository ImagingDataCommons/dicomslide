import uuid
from typing import List, Mapping, Tuple

import highdicom as hd
import pytest
from pydicom.uid import JPEGBaseline8Bit

from dicomslide.image import TiledImage

from .dummy import VLWholeSlideMicroscopyImage


def generate_test_image(
    image_orientation: Tuple[float, float, float, float, float, float],
) -> Mapping[Tuple[str, str], List[VLWholeSlideMicroscopyImage]]:
    study_instance_uid = hd.UID()
    series_instance_uid = hd.UID()
    frame_of_reference_uid = hd.UID()
    container_id = str(uuid.uuid4())
    specimen_id = str(uuid.uuid4())
    specimen_uid = hd.UID()
    image_kwargs = dict(
        image_type=('ORIGINAL', 'PRIMARY', 'VOLUME', 'NONE'),
        total_pixel_matrix_rows=512,
        total_pixel_matrix_columns=256,
        rows=32,
        columns=32,
        pixel_spacing=(0.001, 0.001),
        spacing_between_slices=0.001,
        number_of_focal_planes=1,
        number_of_optical_paths=1,
        optical_path_identifiers=["1"],
        samples_per_pixel=3,
        transfer_syntax_uid=JPEGBaseline8Bit
    )

    image_position = (0.0, 0.0, 0.0)
    return VLWholeSlideMicroscopyImage(
        study_instance_uid=study_instance_uid,
        series_instance_uid=series_instance_uid,
        sop_instance_uid=hd.UID(),
        series_number=1,
        instance_number=1,
        extended_depth_of_field=False,
        # TODO: image position needs to be slightly adjusted
        image_position=image_position,
        image_orientation=image_orientation,
        dimension_organization_type=hd.DimensionOrganizationTypeValues.TILED_FULL,
        frame_of_reference_uid=frame_of_reference_uid,
        container_id=container_id,
        specimen_id=specimen_id,
        specimen_uid=specimen_uid,
        **image_kwargs
    )


@pytest.mark.parametrize(
    'image_orientation,rotation_degrees',
    [
        [(1.0, 0.0, 0.0, 0.0, 1.0, 0.0), -180],
        [(0.0, 1.0, 0.0, 1.0, 0.0, 0.0), -270],
    ]
)
def test_tile_rotation(client, image_orientation, rotation_degrees):
    dataset = generate_test_image(image_orientation=image_orientation)
    client.store_instances([dataset])
    image = TiledImage(client, dataset)
    assert image.get_rotation() == rotation_degrees
