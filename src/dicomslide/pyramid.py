import logging
from typing import List, Optional, Sequence

import numpy as np
from pydicom.dataset import Dataset

logger = logging.getLogger(__name__)


def get_image_resolution(image: Dataset) -> float:
    """Get resolution (spacing between pixels) of an image.

    Parameters
    ----------
    image: pydicom.dataset.Dataset
        Metadata of a DICOM VL Whole Slide Microscopy Image instance

    Returns
    -------
    float
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
            'pixel spacing is different along row and column direction'
        )
    return float(pixel_spacing[0])


def sort_images_by_resolution(collection: Sequence[Dataset]) -> List[Dataset]:
    """Sort images by resolution in decending order.

    Parameters
    ----------
    collection: Sequence[pydicom.dataset.Dataset]
        Metadata of DICOM VL Whole Slide Microscopy Image instances

    Returns
    -------
    List[pydicom.dataset.Dataset]
        Sorted metadata of DICOM VL Whole Slide Microscopy Image instances

    """
    return sorted(collection, key=get_image_resolution, reverse=True)


def sort_images_by_size(collection: Sequence[Dataset]) -> List[Dataset]:
    """Sort images by size in decending order.

    Parameters
    ----------
    collection: Sequence[pydicom.dataset.Dataset]
        Metadata of DICOM VL Whole Slide Microscopy Image instances

    Returns
    -------
    List[pydicom.dataset.Dataset]
        Sorted metadata of DICOM VL Whole Slide Microscopy Image instances

    """
    def get_image_size(image: Dataset) -> int:
        return image.TotalPixelMatrixRows * image.TotalPixelMatrixColumns

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

    magnification_to_resolution = {
        1: 0.01,
        2: 0.005,
        4: 0.0025,
        10: 0.001,
        20: 0.0005,
        40: 0.00025,
    }
    try:
        resolution = magnification_to_resolution[magnification]
    except KeyError:
        raise ValueError(
            'Argument "magnification" should be one of the following: '
            '"{}"'.format('", "'.join(magnification_to_resolution.keys()))
        )

    return select_image_at_resolution(
        collection=collection, resolution=resolution, tolerance=tolerance
    )


def select_image_at_resolution(
    collection: Sequence[Dataset],
    resolution: float,
    tolerance: Optional[float] = None,
) -> Dataset:
    """Select an image from a collection at a desired spatial resolution.

    Parameters
    ----------
    collection: Sequence[pydicom.dataset.Dataset]
        Metadata of DICOM VL Whole Slide Microscopy Image instances
    resolution: float
        Spacing between two pixels at the desired resolution
    tolerance: Union[float, None], optional
        Difference between target magnification and closest available
        magnification in millimeter that can be tolerated.

    Returns
    -------
    pydicom.dataset.Dataset
        Image closest to the desired resolution

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

    resolutions = np.array([
        get_image_resolution(image)
        for image in collection
    ])
    distances = resolutions - tolerance

    distance_nearest = np.abs(distances)
    if tolerance is not None:
        if distance_nearest > tolerance:
            raise ValueError(
                f"Could not find suitable resolution {resolution}. "
                "Distance to closest available resolution exceeded "
                f"tolerance of {tolerance} mm by {distance_nearest} mm."
            )

    index_nearest = int(np.argmin(distance_nearest))
    return collection[index_nearest]
