from typing import Optional, Sequence, Tuple

import highdicom as hd
import numpy as np
from pydicom.dataset import Dataset


def get_frame_contours(
    image: Dataset, frame_numbers: Optional[Sequence[int]] = None
) -> Tuple[hd.sr.GraphicTypeValues3D, Sequence[np.ndarray], hd.UID]:
    """Get contours of image frames in the slide coordinate system.

    Parameters
    ----------
    image: pydicom.dataset.Dataset
        Metadata of a DICOM VL Whole Slide Microscopy Image
    frame_numbers: Union[Sequence[int], None], optional
        One-based index number of frames for which contours should be obtained

    Returns
    -------
    graphic_type: highdicom.sr.GraphicTypeValues3D
        Graphic type
    graphic_data: Sequence[numpy.ndarray]
        Graphic data (3D spatial coordinates in slide coordinate system)
    frame_of_reference_uid: highdicom.UID
        Unique identifier of frame of reference (slide coordinate system)

    """
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

    if frame_numbers is None:
        frame_numbers = list(range(1, int(image.NumberOfFrames) + 1))

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
    for n in frame_numbers:
        frame_index = n - 1
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
