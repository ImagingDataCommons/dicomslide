import logging
from operator import eq
from typing import Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple

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
        return get_image_pixel_spacing(image)[0]

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
            '"{}"'.format('", "'.join([
                str(key) for key in magnification_to_pixel_spacing.keys()
            ]))
        )

    return select_image_at_pixel_spacing(
        collection=collection,
        pixel_spacing=(pixel_spacing, pixel_spacing),
        tolerance=tolerance
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
            [
                all_pixel_spacings[:, 0] - pixel_spacing[0],
                all_pixel_spacings[:, 1] - pixel_spacing[1],
            ],
            axis=0
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


class PyramidLevel(NamedTuple):

    """Image pyramid level."""

    total_pixel_matrix_dimensions: Tuple[int, int]
    pixel_spacing: Tuple[float, float]
    downsampling_factors: Tuple[float, float]
    has_pixels: bool

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return all([
            eq(
                self.total_pixel_matrix_dimensions,
                other.total_pixel_matrix_dimensions
            ),
            eq(
                self.pixel_spacing,
                other.pixel_spacing
            ),
            eq(
                self.downsampling_factors,
                other.downsampling_factors
            ),
            eq(
                self.has_pixels,
                other.has_pixels
            ),
        ])

    def __contains__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return all([
            eq(
                self.total_pixel_matrix_dimensions,
                other.total_pixel_matrix_dimensions
            ),
            eq(
                self.pixel_spacing,
                other.pixel_spacing
            ),
            eq(
                self.downsampling_factors,
                other.downsampling_factors
            ),
        ])


class Pyramid:

    """Image pyramid."""

    def __init__(
        self,
        metadata: Sequence[Dataset],
        tolerance: float,
        ref_metadata: Optional[Sequence[Dataset]] = None
    ) -> None:
        """

        Parameters
        ----------
        metadata: Sequence[pydicom.Dataset]
            Metadata of DICOM image instances
        tolerance: float
            Maximally tolerated distances between the centers of images at
            different pyramid levels in the slide coordinate system in
            millimeter unit
        ref_metadata: Union[Sequence[pydicom.Dataset], None], optional
            Metadata of referenced DICOM source image instances that may serve
            as a template

        """
        if not all([is_volume_image(image) for image in metadata]):
            raise ValueError(
                'Images in pyramid must have flavor VOLUME or THUMBNAIL.'
            )

        if not all([is_tiled_image(image) for image in metadata]):
            raise ValueError('Images in pyramid must be tiled.')

        sizes = [get_image_size(image) for image in metadata]
        if len(set(sizes)) != len(sizes):
            raise ValueError('Images in pyramid must have unique sizes.')

        if not np.array_equal(
            np.argsort(sizes),
            np.flip(np.arange(len(sizes)))
        ):
            raise ValueError(
                'Images in pyramid must be sorted by size in decending order.'
            )

        coordinates = np.array([
            compute_image_center_position(image)
            for image in metadata
        ])
        max_point = np.max(coordinates, axis=0)
        min_point = np.min(coordinates, axis=0)
        distance = np.linalg.norm(max_point - min_point)
        if distance > tolerance:
            raise ValueError(
                'Images in pyramid must be spatially aligned. '
                'Distances between image centers in slide coordinate system '
                f'differ by more than {tolerance} mm: \n{coordinates}'
            )

        if ref_metadata is None:
            ref_metadata = metadata

        level_lut: Dict[Tuple[int, int], PyramidLevel] = {}
        level_keys: List[Tuple[int, int]] = []
        ref_base_image = ref_metadata[0]
        for i, ref_image in enumerate(ref_metadata):
            total_pixel_matrix_dimensions = (
                ref_image.TotalPixelMatrixRows,
                ref_image.TotalPixelMatrixColumns,
            )
            pixel_spacing = (
                float(
                    ref_image
                    .SharedFunctionalGroupsSequence[0]
                    .PixelMeasuresSequence[0]
                    .PixelSpacing[1]
                ),
                float(
                    ref_image
                    .SharedFunctionalGroupsSequence[0]
                    .PixelMeasuresSequence[0]
                    .PixelSpacing[0]
                ),
            )
            downsampling_factors = (
                (
                    ref_base_image.TotalPixelMatrixRows /
                    ref_image.TotalPixelMatrixRows
                ),
                (
                    ref_base_image.TotalPixelMatrixColumns /
                    ref_image.TotalPixelMatrixColumns
                ),
            )
            pyramid_level = PyramidLevel(
                total_pixel_matrix_dimensions,
                pixel_spacing,
                downsampling_factors,
                False
            )
            key = (
                total_pixel_matrix_dimensions[0],
                total_pixel_matrix_dimensions[1],
            )
            level_lut[key] = pyramid_level
            level_keys.append(key)

        base_image = metadata[0]
        for i, image in enumerate(metadata):
            total_pixel_matrix_dimensions = (
                image.TotalPixelMatrixRows,
                image.TotalPixelMatrixColumns,
            )
            pixel_spacing = (
                float(
                    image
                    .SharedFunctionalGroupsSequence[0]
                    .PixelMeasuresSequence[0]
                    .PixelSpacing[1]
                ),
                float(
                    image
                    .SharedFunctionalGroupsSequence[0]
                    .PixelMeasuresSequence[0]
                    .PixelSpacing[0]
                ),
            )
            downsampling_factors = (
                (
                    base_image.TotalPixelMatrixRows /
                    image.TotalPixelMatrixRows
                ),
                (
                    base_image.TotalPixelMatrixColumns /
                    image.TotalPixelMatrixColumns
                ),
            )
            key = (
                total_pixel_matrix_dimensions[0],
                total_pixel_matrix_dimensions[1],
            )
            try:
                ref_pyramid_level = level_lut[key]
            except KeyError:
                raise ValueError(
                    f'Pyramid level #{i} does not match any level in the '
                    'reference pyramid. The reference pyramid has no level '
                    f'with total pixel matrix dimensions {key}.'
                )
            pyramid_level = PyramidLevel(
                total_pixel_matrix_dimensions,
                pixel_spacing,
                downsampling_factors,
                True
            )
            if not all([
                eq(
                    ref_pyramid_level.pixel_spacing,
                    pyramid_level.pixel_spacing
                ),
                eq(
                    ref_pyramid_level.downsampling_factors,
                    pyramid_level.downsampling_factors
                ),
            ]):
                raise ValueError(
                    f'Pyramid level #{i} does not match corresponding level '
                    'of reference pyramid. '
                    'Downsampling factors: '
                    f'{pyramid_level.downsampling_factors} versus '
                    f'{ref_pyramid_level.downsampling_factors}. '
                    'Pixel spacing: '
                    f'{pyramid_level.pixel_spacing} versus '
                    f'{ref_pyramid_level.pixel_spacing}.'
                )
            level_lut[key] = pyramid_level

        self._levels = tuple([level_lut[key] for key in level_keys])
        self._current_index = 0

    def __repr__(self) -> str:
        parts = []
        for i, level in enumerate(self):
            parts.append(f'=== Pyramid Level {i} ===')
            if level is not None:
                parts.append(
                    'Total Pixel Matrix Rows/Columns: '
                    f'{level.total_pixel_matrix_dimensions}'
                )
                parts.append(
                    f'Pixel Spacing: {level.pixel_spacing}'
                )
                parts.append(
                    f'Downsampling Factors: {level.downsampling_factors}'
                )
        return '\n'.join(parts)

    def __getitem__(self, key: int) -> PyramidLevel:
        try:
            return self._levels[key]
        except KeyError:
            raise IndexError(f'Pyramid does not have level {key}.')

    def __delitem__(self, key: int) -> None:
        raise AttributeError('Cannot delete pyramid level.')

    def __setitem__(self, key: int, value: PyramidLevel) -> None:
        raise AttributeError('Cannot set pyramid level.')

    def __iter__(self) -> Iterator[PyramidLevel]:
        return self

    def __next__(self) -> PyramidLevel:
        key = self._current_index
        if key < len(self._levels):
            self._current_index += 1
        else:
            self._current_index = 0
            raise StopIteration
        return self[key]

    def __len__(self) -> int:
        return len(self._levels)

    def __contains__(self, item: object) -> bool:
        if not isinstance(item, self.__class__):
            return False
        if len(self) != len(item):
            return False
        for i in range(len(self)):
            level = self[i]
            try:
                other_level = item[i]
            except IndexError:
                return False
            if level not in other_level:
                return False
        return True

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        if len(self) != len(other):
            return False
        for i in range(len(self)):
            level = self[i]
            try:
                other_level = other[i]
            except IndexError:
                return False
            if level != other_level:
                return False
        return True
