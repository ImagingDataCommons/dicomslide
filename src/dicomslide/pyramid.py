import logging
from typing import List, Optional, Sequence, Tuple

import highdicom as hd
import numpy as np
from pydicom.dataset import Dataset

from dicomslide.utils import is_tiled_image, is_volume_image

logger = logging.getLogger(__name__)


def get_image_pixel_spacing(image: Dataset) -> Tuple[float, float]:
    """Get pixel spacing (spacing between pixels) of an image.

    Parameters
    ----------
    image: pydicom.dataset.Dataset
        Metadata of a DICOM VL Whole Slide Microscopy Image instance
        derived image instance (e.g., DICOM Segmentation)

    Returns
    -------
    Tuple[float, float]
        Pixel spacing

    Note
    ----
    It is assumed that pixels are square.

    """
    pixel_spacing = (
        image
        .SharedFunctionalGroupsSequence[0]
        .PixelMeasuresSequence[0]
        .PixelSpacing
    )
    if len(set(pixel_spacing)) > 1:
        logger.warn(
            'pixel spacing is different along row and column directions'
        )
    return (float(pixel_spacing[0]), float(pixel_spacing[1]))


def get_image_size(image: Dataset) -> int:
    """Get size of an image.

    Parameters
    ----------
    image: pydicom.dataset.Dataset
        Metadata of a DICOM VL Whole Slide Microscopy Image instance or a
        derived image instance (e.g., DICOM Segmentation)

    Returns
    -------
    int
        Number of pixels in each total pixel matrix

    """
    return int(image.TotalPixelMatrixRows) * int(image.TotalPixelMatrixColumns)


def sort_images_by_resolution(
    collection: Sequence[Dataset]
) -> List[Dataset]:
    """Sort images by resolution in descending order (highest to lowest).

    Parameters
    ----------
    collection: Sequence[pydicom.dataset.Dataset]
        Metadata of DICOM VL Whole Slide Microscopy Image instances

    Returns
    -------
    List[pydicom.dataset.Dataset]
        Sorted metadata of DICOM VL Whole Slide Microscopy Image instances

    """
    def get_image_pixel_spacing_rows(image: Dataset) -> float:
        return get_image_pixel_spacing[0]

    return sorted(collection, key=get_image_pixel_spacing_rows, reverse=True)


def sort_images_by_pixel_spacing(
    collection: Sequence[Dataset]
) -> List[Dataset]:
    """Sort images by pixel spacing in ascending order (lowest to highest).

    Parameters
    ----------
    collection: Sequence[pydicom.dataset.Dataset]
        Metadata of DICOM VL Whole Slide Microscopy Image instances

    Returns
    -------
    List[pydicom.dataset.Dataset]
        Sorted metadata of DICOM VL Whole Slide Microscopy Image instances

    """
    return sorted(collection, key=get_image_pixel_spacing, reverse=False)


def sort_images_by_size(collection: Sequence[Dataset]) -> List[Dataset]:
    """Sort images by size in descending order (largest to smallest).

    Parameters
    ----------
    collection: Sequence[pydicom.dataset.Dataset]
        Metadata of DICOM VL Whole Slide Microscopy Image instances

    Returns
    -------
    List[pydicom.dataset.Dataset]
        Sorted metadata of DICOM VL Whole Slide Microscopy Image instances

    """
    return sorted(collection, key=get_image_size, reverse=True)


def select_image_at_magnification(
    collection: Sequence[Dataset],
    magnification: int,
    tolerance: Optional[float] = None,
) -> Dataset:
    """Select an image from a collection at a desired magnification.

    Parameters
    ----------
    collection: Sequence[pydicom.dataset.Dataset]
        Metadata of DICOM VL Whole Slide Microscopy Image instances
    magnification:
        Magnification level (corresponds roughly to object lens power of a
        microscope) of the image that should be selected.
        Note that an image with an exactly matching magnification may not
        exist. In this case, the nearest level will be chosen.
        Choices: ``{2, 4, 10, 20, 40}``
    tolerance: Union[float, None], optional
        Difference between target magnification and closest available
        magnification in millimeter that can be tolerated.

    Returns
    -------
    pydicom.dataset.Dataset
        Image closest to the desired magnification

    Raises
    ------
    ValueError
        When argument `collection` is an emtpy sequence, when argument
        `magnification` does not match one of the available options, or when
        `tolerance` is exceeded.

    """
    if len(collection) == 0:
        raise ValueError('Argument "collection" must not be empty.')

    magnification_to_pixel_spacing = {
        1: 0.01,
        2: 0.005,
        4: 0.0025,
        10: 0.001,
        20: 0.0005,
        40: 0.00025,
    }
    try:
        pixel_spacing = magnification_to_pixel_spacing[magnification]
    except KeyError:
        raise ValueError(
            'Argument "magnification" should be one of the following: '
            '"{}"'.format('", "'.join(magnification_to_pixel_spacing.keys()))
        )

    return select_image_at_pixel_spacing(
        collection=collection, pixel_spacing=pixel_spacing, tolerance=tolerance
    )


def select_image_at_pixel_spacing(
    collection: Sequence[Dataset],
    pixel_spacing: Tuple[float, float],
    tolerance: Optional[float] = None,
) -> Dataset:
    """Select an image from a collection at a desired spatial pixel spacing.

    Parameters
    ----------
    collection: Sequence[pydicom.dataset.Dataset]
        Metadata of DICOM VL Whole Slide Microscopy Image instances
    pixel_spacing: Tuple[float, float]
        Desired spacing between two pixels along the row and column direction
        of the image from top to bottom and left to right, respectively.
    tolerance: Union[float, None], optional
        Difference between target magnification and closest available
        magnification in millimeter that can be tolerated.

    Returns
    -------
    pydicom.dataset.Dataset
        Image closest to the desired pixel spacing

    Raises
    ------
    ValueError
        When argument `collection` is an emtpy sequence or when
        `tolerance` is exceeded.

    Note
    ----
    If multiple images with the same pixel spacing are contained in
    `collection`, the first matching image will be returned. It is the
    responsibility of the caller to filter the images beforehand if necessary.

    """
    if len(collection) == 0:
        raise ValueError('Argument "collection" must not be empty.')

    all_pixel_spacings = np.array([
        get_image_pixel_spacing(image)
        for image in collection
    ])
    distances = np.abs(
        np.mean(
            all_pixel_spacings[:, 0] - pixel_spacing[0],
            all_pixel_spacings[:, 1] - pixel_spacing[1],
        )
    )
    index_nearest = int(np.argmin(distances))
    distance_nearest = distances[index_nearest]

    if tolerance is not None:
        if distance_nearest > tolerance:
            raise ValueError(
                'Could not find image with suitable pixel spacing '
                f'{pixel_spacing}. Distance between requested pixel spacing '
                'and nearest available pixel spacing exceeded '
                f'tolerance of {tolerance} mm by {distance_nearest} mm.'
            )

    return collection[index_nearest]


def compute_image_center_position(image: Dataset) -> Tuple[float, float, float]:
    """Compute position of image center in slide coordinate system.

    Parameters
    ----------
    image: pydicom.dataset.Dataset
        Metadata of DICOM VL Whole Slide Microscopy Image instance

    Returns
    -------
    Tuple[float, float, float]
        (x, y, z) coordinates

    """
    image_origin = image.TotalPixelMatrixOriginSequence[0]
    transformer = hd.spatial.ImageToReferenceTransformer(
        image_position=(
            image_origin.XOffsetInSlideCoordinateSystem,
            image_origin.YOffsetInSlideCoordinateSystem,
            0.0,
        ),
        image_orientation=image.ImageOrientationSlide,
        pixel_spacing=(
            image
            .SharedFunctionalGroupsSequence[0]
            .PixelMeasuresSequence[0]
            .PixelSpacing
        )
    )
    coordinates = transformer(
        np.array([
            (
                image.TotalPixelMatrixColumns / 2,
                image.TotalPixelMatrixRows / 2,
            )
        ])
    )
    return (coordinates[0, 0], coordinates[0, 1], coordinates[0, 2])


def assert_valid_pyramid(
    collection: Sequence[Dataset],
    tolerance: float = 0.001
) -> None:
    """Assert that images form a valid pyramid.

    Parameters
    ----------
    collection: Sequence[pydicom.dataset.Dataset]
        Metadata of DICOM image instances
    tolerance: float, optional
        Maximally tolerated distances between image centers in the slide
        coordinate system in millimeter unit

    Raises
    ------
    ValueError
        When images do not form a valid pyramid

    """
    if not all([is_volume_image(image) for image in collection]):
        raise ValueError('Pyramid must consist of VOLUME or THUMBNAIL images.')

    if not all([is_tiled_image(image) for image in collection]):
        raise ValueError('Images in pyramid must tiled.')

    sizes = [get_image_size(image) for image in collection]
    if len(set(sizes)) != len(sizes):
        raise ValueError('Images in pyramid must have unique sizes.')

    if not np.array_equal(np.argsort(sizes), np.flip(np.arange(len(sizes)))):
        raise ValueError(
            'Images in pyramid must be sorted by size in decending order.'
        )

    slide_coordinates = np.array([
        compute_image_center_position(image)
        for image in collection
    ])

    x_diff = np.max(slide_coordinates[:, 0]) - np.min(slide_coordinates[:, 0])
    if x_diff > tolerance:
        raise ValueError(
            'Images in pyramid must be spatially aligned. '
            'X coordinates of image centers differ by {x_diff} mm, '
            'which exceeds tolerance of {tolerance} mm.'
        )
    y_diff = np.max(slide_coordinates[:, 1]) - np.min(slide_coordinates[:, 1])
    if y_diff > tolerance:
        raise ValueError(
            'Images in pyramid must be spatially aligned. '
            'Y coordinates of image centers differ by {x_diff} mm, '
            'which exceeds tolerance of {tolerance} mm.'
        )
