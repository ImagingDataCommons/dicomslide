import itertools
import logging
from collections import defaultdict, OrderedDict
from hashlib import sha256
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import highdicom as hd
import numpy as np
from dicomweb_client import DICOMClient
from pydicom import Dataset
from pydicom._storage_sopclass_uids import VLWholeSlideMicroscopyImageStorage

from dicomslide.image import TiledImage
from dicomslide.pyramid import get_image_size
from dicomslide.utils import (
    encode_dataset,
    is_volume_image,
    is_label_image,
    is_overview_image,
)

logger = logging.getLogger(__name__)


class Slide:

    """A digital slide.

    A collection of DICOM VL Whole Slide Microscopy Image instances that share
    the same Frame of Reference UID and Container Identifier, i.e., that have
    been acquired as part of one image acquisition for the same physical glass
    slide (container) and can be visualized and analyzed in the same frame of
    reference (coordinate system).

    A slide consists of one or more image pyramids - one for each unique pair
    of optical path and focal plane. The total pixel matrices of the different
    pyramid levels are stored in separate DICOM image instances. Individual
    optical paths or focal planes may be each stored in separate DICOM image
    instances or combined in a single DICOM image instance per pyramid level.
    Pyramids are expected to have the same number of levels and the same
    downsampling factors across optical paths and focal planes and the total
    pixel matrices at each level are expected to have the same dimensions
    (i.e., the same number of total pixel matrix columns and rows). However,
    the tiling of the total pixel matrices (i.e., the number of tile columns
    and rows) may differ across pyramid levels as well as across optical paths
    and focal planes at the same pyramid level.

    A slide may further be associated with additional DICOM Segmentation,
    Parametric Map, Comprehensive 3D SR, or other types of instances that were
    derived from the DICOM VL Whole Slide Microscopy Image instances belonging
    to the slide.

    """

    def __init__(
        self,
        client: DICOMClient,
        image_metadata: Sequence[Dataset],
        max_frame_cache_size: int = 6
    ):
        """

        Parameters
        ----------
        client: dicomweb_client.api.DICOMClient
            DICOMweb client
        image_metadata: Sequence[pydicom.Dataset]
            Metadata of DICOM VL Whole Slide Microscopy Image instances that
            belong to the slide
        max_frame_cache_size: int, optional
            Maximum number of frames that should be cached per image instance
            to avoid repeated retrieval requests

        Raises
        ------
        ValueError
            When for any item of `image_metadata` the value of attribute
            SOP Class UID attribute is not ``"1.2.840.10008.5.1.4.1.1.77.1.6"``
            (VL Whole Slide Microscopy Image) or when the values of attributes
            Container Identifier or Frame of Reference UID differ between
            items of `image_metadata`.

        """
        if not isinstance(image_metadata, Sequence):
            raise TypeError('Argument "image_metadata" must be a sequence.')
        if len(image_metadata) == 0:
            raise ValueError('Argument "image_metadata" cannot be empty.')
        ref_image = image_metadata[0]
        for i, metadata in enumerate(image_metadata):
            if not isinstance(metadata, Dataset):
                raise TypeError(
                    f'Item #{i} of argument "image_metadata" must have type '
                    'pydicom.Dataset.'
                )
            if metadata.SOPClassUID != VLWholeSlideMicroscopyImageStorage:
                raise ValueError(
                    f'Item #{i} of argument "image_metadata" must be a DICOM '
                    'VL Whole Slide Microscpy Image instance.'
                )
            if metadata.FrameOfReferenceUID != ref_image.FrameOfReferenceUID:
                raise ValueError(
                    'All items of argument "image_metadata" must be DICOM '
                    'VL Whole Slide Microscpy Image instance with the same '
                    'Frame of Reference UID.'
                )
            if metadata.ContainerIdentifier != ref_image.ContainerIdentifier:
                raise ValueError(
                    'All items of argument "image_metadata" must be DICOM '
                    'VL Whole Slide Microscpy Image instance with the same '
                    'Container Identifier.'
                )

        volume_images_lut = defaultdict(list)
        label_images = []
        overview_images = []
        for metadata in image_metadata:
            image = TiledImage(
                client=client,
                image_metadata=metadata,
                max_frame_cache_size=max_frame_cache_size
            )
            if is_volume_image(metadata):
                iterator = itertools.product(
                    range(1, image.num_optical_paths + 1),
                    range(1, image.num_focal_planes + 1),
                )
                for optical_path_index, focal_plane_index in iterator:
                    optical_path_identifier = image.get_optical_path_identifier(
                        optical_path_index
                    )
                    focal_plane_offset = image.get_focal_plane_offset(
                        focal_plane_index
                    )
                    key = (optical_path_identifier, focal_plane_offset)
                    volume_images_lut[key].append(image)
            if is_overview_image(metadata):
                overview_images.append(image)
            if is_label_image(metadata):
                label_images.append(image)

        if len(volume_images_lut) == 0:
            raise ValueError(
                'Slide must contain at least one VOLUME or THUMBNAIL image.'
            )
        self._label_images = tuple(label_images)
        self._overview_images = tuple(overview_images)

        unique_optical_path_identifiers = set()
        unique_focal_plane_offsets = set()
        for optical_path_id, focal_plane_offset in volume_images_lut.keys():
            unique_optical_path_identifiers.add(optical_path_id)
            unique_focal_plane_offsets.add(focal_plane_offset)

        self._number_of_optical_paths = len(unique_optical_path_identifiers)
        self._number_of_focal_planes = len(unique_focal_plane_offsets)
        self._optical_path_identifier_lut: Mapping[int, str] = OrderedDict({
            i + 1: optical_path_id
            for i, optical_path_id in enumerate(unique_optical_path_identifiers)
        })
        self._focal_plane_offset_lut: Mapping[int, float] = OrderedDict({
            i + 1: focal_plane_offset
            for i, focal_plane_offset in enumerate(unique_focal_plane_offsets)
        })
        self._volume_images: Dict[Tuple[int, int], Tuple[TiledImage, ...]] = {}
        encoded_image_metadata = []
        for optical_path_index in self._optical_path_identifier_lut:
            optical_path_id = self._optical_path_identifier_lut[
                optical_path_index
            ]
            for focal_plane_index in self._focal_plane_offset_lut:
                focal_plane_offset = self._focal_plane_offset_lut[
                    focal_plane_index
                ]
                volume_images = sorted(
                    volume_images_lut[(optical_path_id, focal_plane_offset)],
                    key=lambda image: get_image_size(image.metadata),
                    reverse=True
                )
                volume_image_key = (optical_path_index, focal_plane_index)
                self._volume_images[volume_image_key] = tuple(volume_images)
                encoded_image_metadata.extend([
                    encode_dataset(image.metadata)
                    for images in volume_images
                ])
        encoded_image_metadata.extend([
            encode_dataset(image.metadata)
            for image in self.overview_images + self.label_images
        ])

        ref_volume_images = self._volume_images[(1, 1)]
        ref_volume_base_image = ref_volume_images[0]
        self._pyramid = [
            {
                'image_dimensions': (
                    image.metadata.TotalPixelMatrixColumns,
                    image.metadata.TotalPixelMatrixRows,
                ),
                'volume_dimensions': (
                    float(image.metadata.ImagedVolumeWidth),
                    float(image.metadata.ImagedVolumeHeight),
                    float(image.metadata.ImagedVolumeDepth),
                ),
                'pixel_spacing': tuple([
                    float(value)
                    for value in (
                        image
                        .metadata
                        .SharedFunctionalGroupsSequence[0]
                        .PixelMeasuresSequence[0]
                        .PixelSpacing
                    )
                ]),
                'downsampling_factor': (
                    (
                        ref_volume_base_image.metadata.TotalPixelMatrixColumns /
                        image.metadata.TotalPixelMatrixColumns
                    ),
                    (
                        ref_volume_base_image.metadata.TotalPixelMatrixRows /
                        image.metadata.TotalPixelMatrixRows
                    ),
                ),
            }
            for image in ref_volume_images
        ]

        # The hash is computed using the image metadata rather than the pixel
        # data to avoid having to retrieve the potentially large pixel data.
        self._quickhash = sha256(b''.join(encoded_image_metadata)).hexdigest()

    def __hash__(self) -> int:
        return hash(self._quickhash)

    def get_volume_images(
        self,
        optical_path_index: int = 1,
        focal_plane_index: int = 1
    ) -> Tuple[TiledImage, ...]:
        """Get VOLUME or THUMBNAIL images for an optical path and focal plane.

        Parameters
        ----------
        optical_path_index: int, optional
            One-based index into optical paths along the direction defined by
            Optical Path Identifier attribute of VOLUME or THUMBNAIL images.
        focal_plane_index: int, optional
            One-based index into focal planes along depth direction from the
            glass slide towards the coverslip in the slide coordinate system
            specified by the Z Offset in Slide Coordinate System attribute of
            VOLUME or THUMBNAIL images.

        Returns
        -------
        Tuple[dicomslide.TiledImage, ...]
            Images sorted by size in descending order

        """
        key = (optical_path_index, focal_plane_index)
        try:
            return tuple(self._volume_images[key])
        except KeyError:
            raise IndexError(
                f'No VOLUME images found for optical path {optical_path_index} '
                f'and focal plane {focal_plane_index}.'
            )

    @property
    def label_images(self) -> Tuple[TiledImage, ...]:
        """Tuple[dicomslide.TiledImage, ...]: LABEL images of the slide"""
        return self._label_images

    @property
    def overview_images(self) -> Tuple[TiledImage, ...]:
        """Tuple[dicomslide.TiledImage, ...]: OVERVIEW images of the slide"""
        return self._overview_images

    @property
    def num_optical_paths(self) -> int:
        """int: Number of optical paths"""
        return self._number_of_optical_paths

    def get_optical_path_identifier(self, optical_path_index: int) -> str:
        """Get identifier of an optical path.

        Parameters
        ----------
        optical_path_index: int
            One-based index into optical paths along the direction defined by
            Optical Path Identifier attribute of VOLUME or THUMBNAIL images.

        Returns
        -------
        str
            Optical path identifier

        Raises
        ------
        ValueError
            When no optical path is found for `optical_path_index`

        """
        try:
            return self._optical_path_identifier_lut[optical_path_index]
        except IndexError:
            raise ValueError(
                'No VOLUME or THUMNAIL image found for optical path '
                f'{optical_path_index}.'
            )

    @property
    def num_focal_planes(self) -> int:
        """int: Number of focal planes"""
        return self._number_of_focal_planes

    def get_focal_plane_offset(self, focal_plane_index: int) -> float:
        """Get z offset in slide coordinate system of focal plane.

        Parameters
        ----------
        focal_plane_index: int
            One-based index into focal planes along depth direction from the
            glass slide towards the coverslip in the slide coordinate system
            specified by the Z Offset in Slide Coordinate System attribute of
            VOLUME or THUMBNAIL images.

        Returns
        -------
        float
            Offset of the focal plane from the from the slide surface along the
            Z axis of the slide coordinate system in micrometers

        Raises
        ------
        ValueError
            When no focal plane is found for `focal_plane_index`

        """
        try:
            return self._focal_plane_offset_lut[focal_plane_index]
        except IndexError:
            raise ValueError(
                'No VOLUME or THUMNAIL image found for focal plane '
                f'{focal_plane_index}.'
            )

    @property
    def num_levels(self) -> int:
        """int: Number of pyramid levels

        Note
        ----
        Levels are sorted by size in descending order from the base level
        (highest image resolution, smallest pixel spacing) to the top level
        (lowest image resolution, largest pixel spacing).

        """
        return len(self._pyramid)

    @property
    def total_pixel_matrix_dimensions(self) -> Tuple[Tuple[int, int], ...]:
        """Tuple[Tuple[int, int], ...]: Number of columns and rows in the total
        pixel matrix for images at each pyramid level

        """
        return tuple([
            (
                int(level['image_dimensions'][0]),
                int(level['image_dimensions'][1]),
            )
            for level in self._pyramid]
        )

    @property
    def pixel_spacings(self) -> Tuple[Tuple[float, float], ...]:
        """Tuple[Tuple[float, float], ...]: Distance between neighboring pixels
        along the row (left to right) and column (top to bottom) directions

        """
        return tuple([
            (
                float(level['pixel_spacing'][0]),
                float(level['pixel_spacing'][1]),
            )
            for level in self._pyramid
        ])

    @property
    def imaged_volume_dimensions(
        self
    ) -> Tuple[Tuple[float, float, float], ...]:
        """Tuple[Tuple[float, float, float], ...]: Width, height, and depth of
        the imaged volume at each pyramid level in millimeter unit.

        Note
        ----
        The imaged volume is supposed to be constant across pyramid levels.
        However, there may be small discrepencies due to errors in the metadata
        (e.g., rounding errors incurred during image resampling).

        """
        return tuple([
            (
                float(level['volume_dimensions'][0]),
                float(level['volume_dimensions'][1]),
                float(level['volume_dimensions'][2]),
            )
            for level in self._pyramid
        ])

    @property
    def downsampling_factors(self) -> Tuple[float, ...]:
        """Tuple[float]: Downsampling factors of images at each pyramid level
        relative to the base level

        """
        return tuple([
            float(np.mean(level['downsampling_factor']))
            for level in self._pyramid
        ])

    def get_image_region(
        self,
        pixel_indices: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
        optical_path_index: int = 1,
        focal_plane_index: int = 1
    ) -> np.ndarray:
        """Get image region.

        Parameters
        ----------
        pixel_indices: Tuple[int, int]
            Zero-based (column, row) indices in pixel unit in the range
            [0, Columns - 1] and [0, Rows - 1], respectively, that
            specify the offset of the image region in the total pixel matrix.
            The ``(0, 0)`` coordinate is located at the **center** of the top
            left pixel in the total pixel matrix.
        level: int
            Pyramid level
        size: Tuple[int, int]
            Width and height of the requested image region in pixel unit
        optical_path_index: int, optional
            One-based index into optical paths along the direction defined by
            Optical Path Identifier attribute of VOLUME or THUMBNAIL images.
        focal_plane_index: int, optional
            One-based index into focal planes along depth direction from the
            glass slide towards the coverslip in the slide coordinate system
            specified by the Z Offset in Slide Coordinate System attribute of
            VOLUME or THUMBNAIL images.

        Returns
        -------
        numpy.ndarray
            Three-dimensional pixel array of shape
            (Rows, Columns, Samples per Pixel) for the requested image region

        """
        logger.debug(
            f'get region of size {size} at position {pixel_indices} '
            f'at level {level} for image with optical path index '
            f'{optical_path_index} and focal plane index {focal_plane_index}'
        )
        volume_images = self.get_volume_images(
            optical_path_index=optical_path_index,
            focal_plane_index=focal_plane_index
        )
        try:
            image = volume_images[level]
        except IndexError:
            raise IndexError(f'Slide does not have level {level}.')

        # Each image may have one or more optical paths or focal planes and the
        # image-level indices differ from the slide-level indices.
        if image.num_optical_paths == 1 and image.num_focal_planes == 1:
            matrix = image.get_total_pixel_matrix(
                optical_path_index=1,
                focal_plane_index=1
            )
        else:
            image_optical_path_index = image.get_optical_path_index(
                self.get_optical_path_identifier(optical_path_index)
            )
            image_focal_plane_index = image.get_focal_plane_index(
                self.get_focal_plane_offset(focal_plane_index)
            )
            matrix = image.get_total_pixel_matrix(
                optical_path_index=image_optical_path_index,
                focal_plane_index=image_focal_plane_index
            )

        col_index, row_index = pixel_indices
        col_factor, row_factor = self._pyramid[level]['downsampling_factor']
        col_start = int(np.floor(col_index / col_factor))
        row_start = int(np.floor(row_index / row_factor))
        cols, rows = size
        row_end = row_start + rows
        col_end = col_start + cols
        logger.debug(
            f'region [{row_start}:{row_end}, {col_start}:{col_end}, :] '
            f'for optical path {optical_path_index} and '
            f'focal plane {focal_plane_index} '
            f'from image "{image.metadata.SOPInstanceUID}"'
        )
        return matrix[row_start:row_end, col_start:col_end, :]

    def get_slide_region(
        self,
        slide_coordinates: Tuple[float, float],
        level: int,
        size: Tuple[float, float],
        optical_path_index: int = 1,
        focal_plane_index: int = 1
    ) -> np.ndarray:
        """Get slide region.

        Parameters
        ----------
        slide_coordinates: Tuple[float, float]
            Zero-based (x, y) coordinates at millimeter resolution in
            the slide coordinate system defined by the frame of reference.
            The ``(0, 0)`` coordinate is located at the origin of the slide
            (which is usually the slide corner).
        level: int
            Pyramid level
        size: Tuple[float, float]
            Width and height of the requested image region in millimeter unit
        optical_path_index: int, optional
            One-based index into optical paths along the direction defined by
            Optical Path Identifier attribute of VOLUME or THUMBNAIL images.
        focal_plane_index: int, optional
            One-based index into focal planes along depth direction from the
            glass slide towards the coverslip in the slide coordinate system
            specified by the Z Offset in Slide Coordinate System attribute of
            VOLUME or THUMBNAIL images.

        Returns
        -------
        numpy.ndarray
            Three-dimensional pixel array of shape
            (Rows, Columns, Samples per Pixel) for the requested image region

        """
        logger.debug(
            f'get region on slide of size {size} [mm] at position '
            f'{slide_coordinates} [mm] at level {level} for image '
            f'with optical path index {optical_path_index} and focal plane '
            f'index {focal_plane_index}'
        )
        volume_images = self.get_volume_images(
            optical_path_index=optical_path_index,
            focal_plane_index=focal_plane_index
        )
        try:
            image = volume_images[level]
        except IndexError:
            raise IndexError(f'Slide does not have level {level}.')

        focal_plane_offset = self.get_focal_plane_offset(focal_plane_index)
        image_orientation = image.metadata.ImageOrientationSlide
        image_origin = image.metadata.TotalPixelMatrixOriginSequence[0]
        image_position = (
            float(image_origin.XOffsetInSlideCoordinateSystem),
            float(image_origin.YOffsetInSlideCoordinateSystem),
            focal_plane_offset,
        )
        pixel_spacing = self.pixel_spacings[level]
        transformer = hd.spatial.ReferenceToPixelTransformer(
            image_orientation=image_orientation,
            image_position=image_position,
            pixel_spacing=pixel_spacing,
        )

        # Each image may have one or more optical paths or focal planes and the
        # image-level indices differ from the slide-level indices.
        if image.num_optical_paths == 1 and image.num_focal_planes == 1:
            matrix = image.get_total_pixel_matrix(
                optical_path_index=1,
                focal_plane_index=1
            )
        else:
            image_optical_path_index = image.get_optical_path_index(
                self.get_optical_path_identifier(optical_path_index)
            )
            image_focal_plane_index = image.get_focal_plane_index(
                focal_plane_offset
            )
            matrix = image.get_total_pixel_matrix(
                optical_path_index=image_optical_path_index,
                focal_plane_index=image_focal_plane_index
            )

        pixel_indices = transformer(np.array([slide_coordinates]))
        col_index, row_index = pixel_indices[0, :]

        region_col_offset = 0
        if col_index < 0:
            region_col_offset = abs(col_index)
        region_row_offset = 0
        if row_index < 0:
            region_row_offset = abs(row_index)
        region_cols = int(np.ceil(size[0] * pixel_spacing[0]))
        region_rows = int(np.ceil(size[1] * pixel_spacing[1]))

        # Region may extend beyond the image's total pixel matrix
        col_offset = min([max([col_index, 0]), matrix.shape[1]])
        row_offset = min([max([row_index, 0]), matrix.shape[0]])
        cols = min([region_cols, matrix.shape[1] - region_col_offset])
        rows = min([region_rows, matrix.shape[0] - region_row_offset])

        region = np.zeros(
            (region_rows, region_cols, matrix.shape[2]),
            dtype=matrix.dtype
        )
        if image.metadata.SamplesPerPixel == 3:
            region += 255

        region[
            region_row_offset:(region_row_offset + rows),
            region_col_offset:(region_col_offset + cols)
        ] = matrix[
            row_offset:(row_offset + rows),
            col_offset:(col_offset + cols),
            :
        ]

        return region


def find_slides(
    client: DICOMClient,
    study_instance_uid: Optional[str] = None,
    max_frame_cache_size: int = 6,
    fail_on_error: bool = True
) -> List[Slide]:
    """Find slides.

    Parameters
    ----------
    client: dicomweb_client.api.DICOMClient
        DICOMweb client
    study_instance_uid: Union[str, None], optional
        DICOM Study Instance UID
    max_frame_cache_size: int, optional
        Maximum number of frames that should be cached per image instance to
        avoid repeated retrieval requests
    fail_on_error: bool, optional
        Wether the function should raise an exception in case an error occurs.
        If ``False``, slides will be skipped.

    Returns
    -------
    List[dicomslide.Slide]
        Digital slides

    """
    # Find VL Whole Slide Microscopy Image instances
    search_filters: Dict[str, str] = {
        'SOPClassUID': VLWholeSlideMicroscopyImageStorage
    }
    if study_instance_uid is not None:
        search_filters['StudyInstanceUID'] = study_instance_uid
    instance_search_results = [
        Dataset.from_json(ds)
        for ds in client.search_for_instances(search_filters=search_filters)
    ]

    # Retrieve metadata of each VL Whole Slide Microscopy Image instance

    def bulk_data_uri_handler(
        tag: str,
        vr: str,
        uri: str
    ) -> Union[bytes, None]:
        if tag != '00282000':
            return None
        uri = uri.replace(
            'http://arc:8080/dcm4chee-arc/aets/DCM4CHEE/rs',
            'http://localhost:8008/dicomweb'
        )
        bulkdata = client.retrieve_bulkdata(uri)[0]
        return bulkdata

    instance_metadata = [
        Dataset.from_json(
            client.retrieve_instance_metadata(
                study_instance_uid=result.StudyInstanceUID,
                series_instance_uid=result.SeriesInstanceUID,
                sop_instance_uid=result.SOPInstanceUID,
            ),
            bulk_data_uri_handler=bulk_data_uri_handler
        )
        for result in instance_search_results
    ]

    # Group images by Study Instance UID, Container Identifier, and
    # Frame of Reference UID.
    lut = defaultdict(list)
    for metadata in instance_metadata:
        try:
            study_instance_uid = metadata.StudyInstanceUID
            container_id = metadata.ContainerIdentifier
            frame_of_reference_uid = metadata.FrameOfReferenceUID
        except AttributeError:
            logger.warning(
                f'skip image "{metadata.SOPInstanceUID}" because of missing '
                'metadata elements'
            )
            continue
        key = (study_instance_uid, container_id, frame_of_reference_uid)
        lut[key].append(metadata)

    found_slides = []
    for image_metadata in lut.values():
        try:
            slide = Slide(
                client=client,
                image_metadata=image_metadata,
                max_frame_cache_size=max_frame_cache_size
            )
        except Exception as error:
            if fail_on_error:
                raise
            else:
                ref_image = image_metadata[0]
                logger.warning(
                    'failed to create slides for images of study '
                    f'"{ref_image.StudyInstanceUID}" for container '
                    f'"{ref_image.ContainerIdentifier}" in frame of reference '
                    f'"{ref_image.FrameOfReferenceUID}" due to the following '
                    f'error: {error}'
                )
                continue
        found_slides.append(slide)

    return found_slides
