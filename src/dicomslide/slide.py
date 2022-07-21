import itertools
import logging
from collections import defaultdict, OrderedDict
from hashlib import sha256
from typing import (
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import highdicom as hd
import numpy as np
from dicomweb_client import DICOMClient
from pydicom import Dataset
from pydicom.sr.coding import Code
from pydicom.uid import (
    ParametricMapStorage,
    SegmentationStorage,
    VLWholeSlideMicroscopyImageStorage,
)

from dicomslide.enum import ChannelTypes
from dicomslide.image import TiledImage
from dicomslide.pyramid import get_image_size, Pyramid
from dicomslide.utils import (
    does_optical_path_item_match,
    does_segment_item_match,
    does_specimen_description_item_match,
    _encode_dataset,
    is_volume_image,
    is_label_image,
    is_overview_image,
)

logger = logging.getLogger(__name__)


class Slide:

    """A digital slide.

    A collection of DICOM image instances that share the same Frame of
    Reference UID and Container Identifier, i.e., that have been acquired as
    part of one image acquisition for the same physical glass slide (container)
    and can be visualized and analyzed in the same frame of reference
    (coordinate system).

    A slide consists of one or more image pyramids - one for each unique pair
    of channel and focal plane. The total pixel matrices of the different
    pyramid levels are stored in separate DICOM image instances. Individual
    channels or focal planes may be each stored in separate DICOM image
    instances or combined in a single DICOM image instance per pyramid level.
    Pyramids are expected to have the same number of levels and the same
    downsampling factors across channels and focal planes and the total pixel
    matrices at each level are expected to have the same dimensions (i.e., the
    same number of total pixel matrix columns and rows). However, the tiling of
    the total pixel matrices (i.e., the number of tile columns and rows) may
    differ across pyramid levels as well as across channels and focal planes at
    the same pyramid level.

    """

    def __init__(
        self,
        client: DICOMClient,
        image_metadata: Sequence[Dataset],
        max_frame_cache_size: int = 6,
        pyramid_tolerance: float = 0.1
    ):
        """

        Parameters
        ----------
        client: dicomweb_client.api.DICOMClient
            DICOMweb client
        image_metadata: Sequence[pydicom.Dataset]
            Metadata of DICOM VL Whole Slide Microscopy Image instances or of
            derived DICOM Segmentation or Parametric Map instances that belong
            to the slide
        max_frame_cache_size: int, optional
            Maximum number of frames that should be cached per image instance
            to avoid repeated retrieval requests
        pyramid_tolerance: float, optional
            Maximally tolerated distances between the centers of images at
            different pyramid levels in the slide coordinate system in
            millimeter unit

        """
        if not isinstance(image_metadata, Sequence):
            raise TypeError('Argument "image_metadata" must be a sequence.')
        if len(image_metadata) == 0:
            raise ValueError('Argument "image_metadata" cannot be empty.')

        logger.debug(f'construct Slide for n={len(image_metadata)} images')
        ref_image = image_metadata[0]
        volume_images_lut: Dict[
            Tuple[ChannelTypes, str, float], List[TiledImage]
        ] = defaultdict(list)
        label_images = []
        overview_images = []
        for i, metadata in enumerate(image_metadata):
            if not isinstance(metadata, Dataset):
                raise TypeError(
                    f'Item #{i} of argument "image_metadata" must have type '
                    'pydicom.Dataset.'
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
                    'VL Whole Slide Microscopy Image instance with the same '
                    'Container Identifier.'
                )
            image = TiledImage(
                client=client,
                image_metadata=metadata,
                max_frame_cache_size=max_frame_cache_size
            )
            if is_volume_image(metadata):
                iterator: Iterator[Tuple[int, int]] = itertools.product(
                    range(image.num_channels),
                    range(image.num_focal_planes),
                )
                for channel_index, focal_plane_index in iterator:
                    channel_type = image.channel_type
                    channel_identifier = image.get_channel_identifier(
                        channel_index
                    )
                    focal_plane_offset = image.get_focal_plane_offset(
                        focal_plane_index
                    )
                    key: Tuple[ChannelTypes, str, float] = (
                        channel_type,
                        channel_identifier,
                        focal_plane_offset,
                    )
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
        self._volume_images: Dict[Tuple[int, int], Tuple[TiledImage, ...]] = {}

        logger.debug('assign images to channels')
        unique_channel_identifiers = defaultdict(set)
        unique_focal_plane_offsets = set()
        for key in volume_images_lut.keys():
            channel_type, channel_id, focal_plane_offset = key
            unique_channel_identifiers[channel_type].add(channel_id)
            unique_focal_plane_offsets.add(focal_plane_offset)

        self._number_of_channels = sum([
            len(ids) for ids in unique_channel_identifiers.values()
        ])
        self._channel_identifier_lut: Mapping[int, str] = OrderedDict()
        self._channel_type_lut: Mapping[int, ChannelTypes] = OrderedDict()
        self._channel_index_lut: Mapping[Tuple[ChannelTypes, str], int] = {}
        i = 0
        for channel_type, channel_ids in unique_channel_identifiers.items():
            for channel_id in sorted(channel_ids):
                self._channel_identifier_lut[i] = channel_id
                self._channel_type_lut[i] = channel_type
                self._channel_index_lut[(channel_type, channel_id)] = i
                i += 1

        self._number_of_focal_planes = len(unique_focal_plane_offsets)
        self._focal_plane_offset_lut: Mapping[int, float] = OrderedDict()
        self._focal_plane_index_lut: Mapping[float, int] = {}
        for i, focal_plane_offset in enumerate(
            sorted(unique_focal_plane_offsets)
        ):
            self._focal_plane_offset_lut[i] = focal_plane_offset
            self._focal_plane_index_lut[focal_plane_offset] = i

        ref_image_metadata: List[TiledImage] = []
        encoded_image_metadata = []
        for channel_index in self._channel_identifier_lut.keys():
            channel_id = self._channel_identifier_lut[channel_index]
            channel_type = self._channel_type_lut[channel_index]
            for focal_plane_index in self._focal_plane_offset_lut.keys():
                focal_plane_offset = self._focal_plane_offset_lut[
                    focal_plane_index
                ]
                volume_images: List[TiledImage] = sorted(
                    volume_images_lut[
                        (channel_type, channel_id, focal_plane_offset)
                    ],
                    key=lambda image: get_image_size(image.metadata),
                    reverse=True
                )
                volume_image_key: Tuple[int, int] = (
                    channel_index,
                    focal_plane_index,
                )
                self._volume_images[volume_image_key] = tuple(volume_images)
                if (
                    channel_type == ChannelTypes.OPTICAL_PATH and
                    len(ref_image_metadata) == 0
                ):
                    ref_image_metadata.extend(volume_images)
                encoded_image_metadata.extend([
                    _encode_dataset(image.metadata)
                    for images in volume_images
                ])
        encoded_image_metadata.extend([
            _encode_dataset(image.metadata)
            for image in self.overview_images + self.label_images
        ])

        logger.debug('build pyramids for each channel')
        pyramids: Dict[Tuple[int, int], Pyramid] = {}
        for (
            channel_index,
            focal_plane_index,
        ), image_collection in self._volume_images.items():
            try:
                pyramids[(channel_index, focal_plane_index)] = Pyramid(
                    metadata=[
                        image.metadata
                        for image in image_collection
                    ],
                    tolerance=pyramid_tolerance,
                    ref_metadata=[
                        image.metadata
                        for image in ref_image_metadata
                    ]
                )
            except ValueError as error:
                raise ValueError(
                    f'VOLUME and THUMBNAIL images for channel {channel_index} '
                    f'and focal plane {focal_plane_index} do not represent '
                    f'a valid image pyramid: {error}'
                )

        # For now, pyramids must be identical across channels and focal planes.
        # This requirement could potentially be relaxed in the future.
        ref_pyramid = pyramids[(0, 0)]
        for pyramid in pyramids.values():
            if pyramid not in ref_pyramid:
                raise ValueError(
                    'Pyramids for different channels and focal planes must '
                    'have the same structure, i.e., the same number of levels '
                    'as well as the same dimensions and downsampling factors '
                    'per level.'
                )
        self._pyramid = ref_pyramid

        # The hash is computed using the image metadata rather than the pixel
        # data to avoid having to retrieve the potentially large pixel data.
        self._quickhash = sha256(b''.join(encoded_image_metadata)).hexdigest()

    def __repr__(self) -> str:
        ref_images = self._volume_images[(0, 0)]
        metadata = ref_images[0].metadata
        return f'<Slide {metadata.ContainerIdentifier} {self._quickhash}>'

    def __hash__(self) -> int:
        return hash(self._quickhash)

    def find_segments(
        self,
        number: Optional[int] = None,
        label: Optional[str] = None,
        property_category: Optional[Union[hd.sr.CodedConcept, Code]] = None,
        property_type: Optional[Union[hd.sr.CodedConcept, Code]] = None
    ) -> Tuple[int, ...]:
        """Find segments.

        Parameters
        ----------
        number: Union[int, None], optional
            Segment number
        label: Union[str, None], optional,
            Segment label
        property_category: Union[hd.sr.CodedConcept, Code, None], optional
            Category of segmented property
        property_type: Union[hd.sr.CodedConcept, Code, None], optional
            Type of segmented property

        Returns
        -------
        Tuple[int, ...]
            Zero-based index into channels along the direction defined by
            successive items of the appropriate DICOM attribute of VOLUME
            or THUMBNAIL images.

        """
        matching_channel_indices = set()
        for (channel_index, _), images in self._volume_images.items():
            ref_image = images[0]
            if ref_image.channel_type != ChannelTypes.SEGMENT:
                continue
            channel_identifier = self.get_channel_identifier(channel_index)
            if number is not None:
                if channel_identifier == str(number):
                    return (channel_index, )

            matching_segment_items = [
                item
                for item in ref_image.metadata.SegmentSequence
                if channel_identifier == str(item.SegmentNumber)
            ]
            segment_item = matching_segment_items[0]
            if does_segment_item_match(
                segment_item,
                number,
                label,
                property_category,
                property_type,
            ):
                matching_channel_indices.add(channel_index)

        return tuple(matching_channel_indices)

    def find_optical_paths(
        self,
        identifier: Optional[str] = None,
        description: Optional[str] = None,
        illumination_wavelength: Optional[float] = None,
        specimen_stain: Optional[Union[hd.sr.CodedConcept, Code]] = None
    ) -> Tuple[int, ...]:
        """Find optical paths.

        Parameters
        ----------
        identifier: Union[str, None], optional
            Optical path identifier
        description: Union[str, None], optional,
            Optical path description
        illumination_wavelength: Union[float, None], optional,
            Optical path illumination wavelength
        specimen_stain: Union[hd.sr.CodedConcept, Code, None], optional
            Substance used for specimen staining

        Returns
        -------
        Tuple[int, ...]
            Zero-based index into channels along the direction defined by
            successive items of the appropriate DICOM attribute of VOLUME
            or THUMBNAIL images.

        """
        matching_channel_indices = set()
        for (channel_index, _), images in self._volume_images.items():
            ref_image = images[0]
            if ref_image.channel_type != ChannelTypes.OPTICAL_PATH:
                continue
            channel_identifier = self.get_channel_identifier(channel_index)
            if identifier is not None:
                if channel_identifier == identifier:
                    return (channel_index, )

            matching_optical_path_items = [
                item
                for item in ref_image.metadata.OpticalPathSequence
                if item.OpticalPathIdentifier == channel_identifier
            ]
            optical_path_item = matching_optical_path_items[0]
            specimen_description_item = (
                ref_image
                .metadata
                .SpecimenDescriptionSequence[0]
            )
            if all([
                does_optical_path_item_match(
                    optical_path_item,
                    identifier,
                    description,
                    illumination_wavelength
                ),
                does_specimen_description_item_match(
                    specimen_description_item,
                    specimen_stain
                )
            ]):
                matching_channel_indices.add(channel_index)

        return tuple(matching_channel_indices)

    def get_volume_images(
        self,
        channel_index: int = 0,
        focal_plane_index: int = 0
    ) -> Tuple[TiledImage, ...]:
        """Get VOLUME or THUMBNAIL images for an channel and focal plane.

        Parameters
        ----------
        channel_index: int, optional
            Zero-based index into channels along the direction defined by
            successive items of the appropriate DICOM attribute of VOLUME
            or THUMBNAIL images.
        focal_plane_index: int, optional
            Zero-based index into focal planes along depth direction from the
            glass slide towards the coverslip in the slide coordinate system
            specified by the Z Offset in Slide Coordinate System attribute of
            VOLUME or THUMBNAIL images.

        Returns
        -------
        Tuple[dicomslide.TiledImage, ...]
            Images sorted by size in descending order

        """
        key = (channel_index, focal_plane_index)
        try:
            return tuple(self._volume_images[key])
        except KeyError:
            raise IndexError(
                f'No VOLUME images found for channel {channel_index} '
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
    def num_channels(self) -> int:
        """int: Number of channels"""
        return self._number_of_channels

    def get_channel_type(self, channel_index: int) -> ChannelTypes:
        """Get type of a channel.

        Parameters
        ----------
        channel_index: int
            Zero-based index into channels along the direction defined by
            successive items of the appropriate DICOM attribute of VOLUME
            or THUMBNAIL images.

        Returns
        -------
        dicomslide.ChannelTypes
            Channel type

        Raises
        ------
        ValueError
            When no channel is found for `channel_index`

        """
        try:
            return self._channel_type_lut[channel_index]
        except IndexError:
            raise ValueError(
                'No VOLUME or THUMNAIL image found for channel index '
                f'{channel_index}.'
            )

    def get_channel_identifier(self, channel_index: int) -> str:
        """Get identifier of a channel.

        Parameters
        ----------
        channel_index: int
            Zero-based index into channels along the direction defined by
            successive items of the appropriate DICOM attribute of VOLUME
            or THUMBNAIL images.

        Returns
        -------
        str
            Channel identifier

        Raises
        ------
        ValueError
            When no channel is found for `channel_index`

        """
        try:
            return self._channel_identifier_lut[channel_index]
        except IndexError:
            raise ValueError(
                'No VOLUME or THUMNAIL image found for channel index '
                f'{channel_index}.'
            )

    def get_channel_index(
        self,
        channel_identifier: str,
        channel_type: Union[ChannelTypes, str]
    ) -> int:
        """Get index of a channel.

        Parameters
        ----------
        channel_identifier: str
            Channel identifier
        channel_type: Union[str, dicomslide.ChannelTypes]
            Channel type

        Returns
        -------
        int
            Zero-based index into channels along the direction defined by
            successive items of the appropriate DICOM attribute, which is
            dependend on the type of channel.

        Raises
        ------
        ValueError
            When no channel is found for `channel_identifier` and `channel_type`

        """
        channel_type = ChannelTypes(channel_type)
        try:
            return self._channel_index_lut[(channel_type, channel_identifier)]
        except IndexError:
            raise ValueError(
                'No VOLUME or THUMNAIL image found for channel identifier '
                f'{channel_identifier} and type {channel_type.value}.'
            )

    @property
    def frame_of_reference_uid(self) -> str:
        """str: Unique identifier of the frame of reference"""
        volume_images = self.get_volume_images(
            channel_index=0,
            focal_plane_index=0
        )
        return volume_images[0].frame_of_reference_uid

    @property
    def physical_offset(self) -> Tuple[float, float]:
        """Tuple[float, float]: Minimum offset of the total pixel matrices
        from the origin of the frame of reference along the X and Y axes of the
        slide coordinate system in millimeter

        """
        offsets = np.array([
            volume_images[0].physical_offset
            for volume_images in self._volume_images.values()
        ])
        min_offsets = np.min(offsets, axis=0)
        return (min_offsets[0], min_offsets[1])

    @property
    def size(self) -> Tuple[int, int]:
        """Tuple[int, int]: Maximum size of the total pixel matrices along
        the rows and columns axes of the total pixel matrix

        """
        sizes = np.array([
            volume_images[0].size
            for volume_images in self._volume_images.values()
        ])
        max_sizes = np.max(sizes, axis=0)
        return (max_sizes[0], max_sizes[1])

    @property
    def physical_size(self) -> Tuple[float, float]:
        """Tuple[float, float]: Maximum size of the total pixel matrices along
        the X and Y axes of the slide coordinate system in millimeter

        """
        x_offset, y_offset = self.physical_offset
        sizes = []
        for volume_images in self._volume_images.values():
            image = volume_images[0]
            x_endpoint, y_endpoint = image.get_slide_offset(image.size)
            sizes.append(
                (
                    abs(x_endpoint - x_offset),
                    abs(y_endpoint - y_offset),
                )
            )
        max_sizes = np.max(np.array(sizes), axis=0)
        return (max_sizes[0], max_sizes[1])

    @property
    def num_focal_planes(self) -> int:
        """int: Number of focal planes"""
        return self._number_of_focal_planes

    def get_focal_plane_offset(self, focal_plane_index: int) -> float:
        """Get z offset of focal plane in slide coordinate system.

        Parameters
        ----------
        focal_plane_index: int
            Zero-based index into focal planes along depth direction from the
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
                'No VOLUME or THUMNAIL image found for focal plane index '
                f'{focal_plane_index}.'
            )

    def get_focal_plane_index(self, focal_plane_offset: float) -> int:
        """Get index of a focal plane.

        Parameters
        ----------
        focal_plane_offset: float
            Offset of the focal plane from the from the slide surface along the
            Z axis of the slide coordinate system in micrometers

        Returns
        -------
        int
            Zero-based index into focal planes along depth direction from the
            glass slide towards the coverslip in the slide coordinate system
            specified by the Z Offset in Slide Coordinate System attribute of
            VOLUME or THUMBNAIL images.

        Raises
        ------
        ValueError
            When no focal plane is found for `focal_plane_offset`

        """
        try:
            return self._focal_plane_index_lut[focal_plane_offset]
        except IndexError:
            raise ValueError(
                'No VOLUME or THUMNAIL image found for focal plane offset '
                f'{focal_plane_offset}.'
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
            level.total_pixel_matrix_dimensions
            for level in self._pyramid
        ])

    @property
    def pixel_spacings(self) -> Tuple[Tuple[float, float], ...]:
        """Tuple[Tuple[float, float], ...]: Distance between neighboring pixels
        along the row (left to right) and column (top to bottom) directions

        """
        return tuple([level.pixel_spacing for level in self._pyramid])

    @property
    def downsampling_factors(self) -> Tuple[float, ...]:
        """Tuple[float]: Downsampling factors of images at each pyramid level
        relative to the base level

        """
        return tuple([
            float(np.mean(level.downsampling_factors))
            for level in self._pyramid
        ])

    def map_pixel_indices_to_slide_coordinates(
        self,
        pixel_indices: np.ndarray,
        level: int,
        channel_index: int = 0,
        focal_plane_index: int = 0
    ) -> np.ndarray:
        """Map pixel indices to slide coordinates.

        Parameters
        ----------
        pixel_indices: numpy.ndarray
            Zero-based (row, column) indices into the total pixel matrix of the
            image
        level: int
            Zero-based index into pyramid levels
        channel_index: int, optional
            Zero-based index into channels along the direction defined by
            successive items of the appropriate DICOM attribute of VOLUME
            or THUMBNAIL images.
        focal_plane_index: int, optional
            Zero-based index into focal planes along depth direction from the
            glass slide towards the coverslip in the slide coordinate system
            specified by the Z Offset in Slide Coordinate System attribute of
            VOLUME or THUMBNAIL images.

        Returns
        -------
        numpy.ndarray
            Zero-based (x, y, z) coordinates in the slide coordinate system in
            millimeter

        """
        volume_images = self.get_volume_images(
            channel_index=channel_index,
            focal_plane_index=focal_plane_index
        )
        try:
            image = volume_images[level]
        except IndexError:
            raise IndexError(f'Slide does not have level {level}.')
        return image.map_pixel_indices_to_slide_coordinates(pixel_indices)

    def map_slide_coordinates_to_pixel_indices(
        self,
        slide_coordinates: np.ndarray,
        level: int,
        channel_index: int = 0,
        focal_plane_index: int = 0
    ) -> np.ndarray:
        """Map slide coordinates to pixel indices.

        Parameters
        ----------
        slide_coordinates: numpy.ndarray
            Zero-based (x, y, z) coordinates in the slide coordinate system in
            millimeter
        level: int
            Zero-based index into pyramid levels
        channel_index: int, optional
            Zero-based index into channels along the direction defined by
            successive items of the appropriate DICOM attribute of VOLUME
            or THUMBNAIL images.
        focal_plane_index: int, optional
            Zero-based index into focal planes along depth direction from the
            glass slide towards the coverslip in the slide coordinate system
            specified by the Z Offset in Slide Coordinate System attribute of
            VOLUME or THUMBNAIL images.

        Returns
        -------
        numpy.ndarray
            Zero-based (row, column) indices into the total pixel matrix of the
            image

        """
        volume_images = self.get_volume_images(
            channel_index=channel_index,
            focal_plane_index=focal_plane_index
        )
        try:
            image = volume_images[level]
        except IndexError:
            raise IndexError(f'Slide does not have level {level}.')
        return image.map_slide_coordinates_to_pixel_indices(slide_coordinates)

    def get_slide_offset(
        self,
        pixel_indices: Tuple[int, int],
        level: int,
        channel_index: int = 0,
        focal_plane_index: int = 0
    ) -> Tuple[float, float]:
        """Get slide coordinates for a given total pixel matrix position.

        Parameters
        ----------
        pixel_indices: Tuple[int, int]
            Zero-based (row, column) offset in the total pixel matrix
        level: int
            Zero-based index into pyramid levels
        channel_index: int, optional
            Zero-based index into channels along the direction defined by
            successive items of the appropriate DICOM attribute of VOLUME
            or THUMBNAIL images.
        focal_plane_index: int, optional
            Zero-based index into focal planes along depth direction from the
            glass slide towards the coverslip in the slide coordinate system
            specified by the Z Offset in Slide Coordinate System attribute of
            VOLUME or THUMBNAIL images.

        Returns
        -------
        Tuple[float, float]
            Zero-based (x, y) position on the slide in the slide coordinate
            system in millimeter

        """
        volume_images = self.get_volume_images(
            channel_index=channel_index,
            focal_plane_index=focal_plane_index
        )
        try:
            image = volume_images[level]
        except IndexError:
            raise IndexError(f'Slide does not have level {level}.')
        return image.get_slide_offset(pixel_indices)

    def get_pixel_indices(
        self,
        offset: Tuple[float, float],
        level: int,
        channel_index: int = 0,
        focal_plane_index: int = 0
    ) -> Tuple[int, int]:
        """Get indices into total pixel matrix for a given slide position.

        Parameters
        ----------
        offset: Tuple[float, float]
            Zero-based (x, y) offset in the slide coordinate system in
            millimeter
        level: int
            Zero-based index into pyramid levels
        channel_index: int, optional
            Zero-based index into channels along the direction defined by
            successive items of the appropriate DICOM attribute of VOLUME
            or THUMBNAIL images.
        focal_plane_index: int, optional
            Zero-based index into focal planes along depth direction from the
            glass slide towards the coverslip in the slide coordinate system
            specified by the Z Offset in Slide Coordinate System attribute of
            VOLUME or THUMBNAIL images.

        Returns
        -------
        Tuple[int, int]
            Zero-based (row, column) position in the total pixel matrix of the
            image

        Note
        ----
        Pixel position may be negativ or extend beyond the size of the total
        pixel matrix if slide position at `offset` does fall into a region on
        the slide that was not imaged.

        """
        volume_images = self.get_volume_images(
            channel_index=channel_index,
            focal_plane_index=focal_plane_index
        )
        try:
            image = volume_images[level]
        except IndexError:
            raise IndexError(f'Slide does not have level {level}.')
        return image.get_pixel_indices(offset)

    def get_image_region(
        self,
        offset: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
        channel_index: int = 0,
        focal_plane_index: int = 0
    ) -> np.ndarray:
        """Get image region.

        Parameters
        ----------
        offset: Tuple[int, int]
            Zero-based (row, column) indices in the range [0, Rows) and
            [0, Columns), respectively, that specify the offset of the image
            region in the total pixel matrix of the image at the highest
            resolution level. The ``(0, 0)`` coordinate is located at the
            center of the topleft hand pixel in the total pixel matrix.
        level: int
            Zero-based index into pyramid levels
        size: Tuple[int, int]
            Rows and columns of the requested image region
        channel_index: int, optional
            Zero-based index into channels along the direction defined by
            successive items of the appropriate DICOM attribute of VOLUME
            or THUMBNAIL images.
        focal_plane_index: int, optional
            Zero-based index into focal planes along depth direction from the
            glass slide towards the coverslip in the slide coordinate system
            specified by the Z Offset in Slide Coordinate System attribute of
            VOLUME or THUMBNAIL images.

        Returns
        -------
        numpy.ndarray
            Three-dimensional pixel array of shape
            (Rows, Columns, Samples per Pixel) for the requested image region

        """
        volume_images = self.get_volume_images(
            channel_index=channel_index,
            focal_plane_index=focal_plane_index
        )
        try:
            image = volume_images[level]
        except IndexError:
            raise IndexError(f'Slide does not have level {level}.')

        row_index, col_index = offset
        col_factor, row_factor = self._pyramid[level].downsampling_factors
        col_start = int(np.floor(col_index / col_factor))
        row_start = int(np.floor(row_index / row_factor))
        image_offset = (row_start, col_start)

        # Each image may have one or more channels or focal planes and the
        # image-level indices may differ from the slide-level indices.
        image_channel_index = image.get_channel_index(
            self.get_channel_identifier(channel_index)
        )
        image_focal_plane_index = image.get_focal_plane_index(
            self.get_focal_plane_offset(focal_plane_index)
        )
        return image.get_image_region(
            offset=image_offset,
            size=size,
            channel_index=image_channel_index,
            focal_plane_index=image_focal_plane_index
        )

    def get_slide_region(
        self,
        offset: Tuple[float, float],
        level: int,
        size: Tuple[float, float],
        channel_index: int = 0,
        focal_plane_index: int = 0
    ) -> np.ndarray:
        """Get slide region.

        Parameters
        ----------
        offset: Tuple[float, float]
            Zero-based (x, y) offset in the slide coordinate system in
            millimeter resolution. The ``(0.0, 0.0)`` coordinate is located at
            the origin of the slide (usually the slide corner).
        level: int
            Zero-based index into pyramid levels
        size: Tuple[float, float]
            Width and height of the requested slide region in millimeter unit
            along the X and Y axis of the slide coordinate system, respectively.
        channel_index: int, optional
            Zero-based index into channels along the direction defined by
            successive items of the appropriate DICOM attribute of VOLUME
            or THUMBNAIL images.
        focal_plane_index: int, optional
            Zero-based index into focal planes along depth direction from the
            glass slide towards the coverslip in the slide coordinate system
            specified by the Z Offset in Slide Coordinate System attribute of
            VOLUME or THUMBNAIL images.

        Returns
        -------
        numpy.ndarray
            Three-dimensional pixel array of shape
            (Rows, Columns, Samples per Pixel) for the requested slide region

        Note
        ----
        The slide coordinate system is defined for the upright standing slide
        such that the X axis corresponds to the short side of the slide and the
        Y axis corresponds to the long side of the slide.
        The rows of the returned pixel array are thus parallel to the X axis of
        the slide coordinate system and the columns parallel to the Y axis of
        the slide coordinate system.

        """
        volume_images = self.get_volume_images(
            channel_index=channel_index,
            focal_plane_index=focal_plane_index
        )
        try:
            image = volume_images[level]
        except IndexError:
            raise IndexError(f'Slide does not have level {level}.')

        # Each image may have one or more channels or focal planes and the
        # image-level indices may differ from the slide-level indices.
        image_channel_index = image.get_channel_index(
            self.get_channel_identifier(channel_index)
        )
        image_focal_plane_index = image.get_focal_plane_index(
            self.get_focal_plane_offset(focal_plane_index)
        )
        return image.get_slide_region(
            offset=offset,
            size=size,
            channel_index=image_channel_index,
            focal_plane_index=image_focal_plane_index
        )

    def get_slide_region_for_annotation(
        self,
        annotation: hd.sr.Scoord3DContentItem,
        level: int,
        channel_index: int = 0,
        padding: Union[
            float,
            Tuple[float, float],
            Tuple[float, float, float, float]
        ] = 0.0
    ) -> np.ndarray:
        """Get slide region defined by a graphic annotation.

        Parameters
        ----------
        annotation: highdicom.sr.Scoord3DContentItem
            Graphic annotation that defines the region of interest (ROI) in the
            slide coordinate system
        level: int
            Zero-based index into pyramid levels
        channel_index: int, optional
            Zero-based index into channels along the direction defined by
            successive items of the appropriate DICOM attribute of VOLUME
            or THUMBNAIL images.
        padding: Union[int, Tuple[int, int], Tuple[int, int, int, int]], optional
            Padding on each border of the region defined by `annotation`.  If a
            single integer is provided, the value is used to pad all four
            borders with the same number of pixels. If a sequence of length 2
            is provided, the two values are used to pad the left/right (along
            the X axis) and top/bottom (along the Y axis) border, respectively.
            If a sequence of length 4 is provided, the four values are used to
            pad the left (- X axis), top (+ Y axis), right (+ X axis), and
            bottom (- Y axis) borders respectively.

        Returns
        -------
        numpy.ndarray
            Three-dimensional pixel array of shape
            (Rows, Columns, Samples per Pixel) for the requested slide region

        Note
        ----
        The slide coordinate system is defined for the upright standing slide
        such that the X axis corresponds to the short side of the slide and the
        Y axis corresponds to the long side of the slide.
        The rows of the returned pixel array are thus parallel to the X axis of
        the slide coordinate system and the columns parallel to the Y axis of
        the slide coordinate system.

        """  # noqa: E501
        full_padding: Tuple[float, float, float, float]
        if isinstance(padding, (int, float)):
            full_padding = (
                float(padding),
                float(padding),
                float(padding),
                float(padding),
            )
        elif isinstance(padding, tuple):
            if len(padding) == 2:
                full_padding = (
                    float(padding[0]),
                    float(padding[1]),
                    float(padding[0]),
                    float(padding[1]),
                )
            elif len(padding) == 4:
                full_padding = (
                    float(padding[0]),
                    float(padding[1]),
                    float(padding[2]),  # type: ignore
                    float(padding[3]),  # type: ignore
                )
            else:
                raise ValueError(
                    'If argument "padding" is a tuple, its length must be '
                    'either 2 or 4.'
                )
        else:
            raise TypeError(
                'Argument "padding" must be either an integer or a tuple.'
            )

        slide_coordinates: Tuple[float, float]
        size: Tuple[float, float]
        focal_plane_offset: float
        graphic_data = annotation.value
        if annotation.graphic_type == hd.sr.GraphicTypeValues3D.POINT:
            offset = (graphic_data[0, 0], graphic_data[0, 1])
            focal_plane_offset = graphic_data[0, 2]
            size = (0.0, 0.0)
        elif annotation.graphic_type in (
            hd.sr.GraphicTypeValues3D.MULTIPOINT,
            hd.sr.GraphicTypeValues3D.POLYGON,
            hd.sr.GraphicTypeValues3D.POLYLINE,
        ):
            min_point = np.min(graphic_data, axis=0)
            max_point = np.max(graphic_data, axis=0)
            offset = (min_point[0], min_point[1])
            focal_plane_offset = min_point[2]
            size = (max_point[0] - min_point[0], max_point[1] - min_point[1])
        elif annotation.graphic_type == hd.sr.GraphicTypeValues3D.ELLIPSE:
            min_point = [
                np.min([graphic_data[:, 0]]),
                np.min([graphic_data[:, 1]]),
            ]
            max_point = [
                np.max([graphic_data[:, 0]]),
                np.max([graphic_data[:, 1]]),
            ]
            offset = (min_point[0], min_point[1])
            # Points are co-planar per definition
            focal_plane_offset = graphic_data[0, 2]
            size = (max_point[0] - min_point[0], max_point[1] - min_point[1])
        elif annotation.graphic_type == hd.sr.GraphicTypeValues3D.ELLIPSOID:
            raise ValueError(
                'SCOORD3D graphic type "ELLIPSOID" is not supported, '
                'because annotations must be planar and the plane must be '
                'parallel to the slide surface.'
            )
        else:
            raise ValueError(
                'Unknown SCOORD3D graphic type "{annotation.graphic_type}".'
            )

        # TODO: Instead of expecting an exact match, one could select the
        # focal plane closest to the annotated region of interest (ROI).
        # The coordinates of the ROI are expected to be co-planar and parallel
        # to the slide surface but they may not be located on the same plane
        # as any of the available focal planes and may instead need to be
        # projected onto the nearest focal plane.
        focal_plane_index = self.get_focal_plane_index(focal_plane_offset)

        volume_images = self.get_volume_images(
            channel_index=channel_index,
            focal_plane_index=focal_plane_index
        )
        try:
            image = volume_images[level]
        except IndexError:
            raise IndexError(f'Slide does not have level {level}.')
        frame_of_reference_uid = image.metadata.FrameOfReferenceUID
        ref_frame_of_reference_uid = annotation.ReferencedFrameOfReferenceUID
        if ref_frame_of_reference_uid != frame_of_reference_uid:
            raise ValueError(
                'Annotation must be defined in same frame of reference as the '
                f'source images: "{frame_of_reference_uid}".'
            )

        offset = (
            offset[0] - full_padding[0],
            offset[1] + full_padding[1],  # offset increases along Y axis!
        )
        size = (
            size[0] + full_padding[0] + full_padding[2],
            size[1] + full_padding[1] + full_padding[3],
        )

        return self.get_slide_region(
            offset=offset,
            level=level,
            size=size,
            channel_index=channel_index,
            focal_plane_index=focal_plane_index
        )


def find_slides(
    client: DICOMClient,
    study_instance_uid: Optional[str] = None,
    patient_id: Optional[str] = None,
    study_id: Optional[str] = None,
    container_id: Optional[str] = None,
    max_frame_cache_size: int = 6,
    pyramid_tolerance: float = 0.1,
    fail_on_error: bool = True,
    include_derived: bool = True
) -> List[Slide]:
    """Find slides.

    Parameters
    ----------
    client: dicomweb_client.api.DICOMClient
        DICOMweb client
    study_instance_uid: Union[str, None], optional
        DICOM Study Instance UID
    patient_id: Union[str, None], optional
        Patient identifier
    study_id: Union[str, None], optional
        Study identifier
    container_id: Union[str, None], optional
        Specimen container (slide) identifier
    max_frame_cache_size: int, optional
        Maximum number of frames that should be cached per image instance to
        avoid repeated retrieval requests
    pyramid_tolerance: float, optional
        Maximally tolerated distances between the centers of images at
        different pyramid levels in the slide coordinate system in
        millimeter unit
    fail_on_error: bool, optional
        Wether the function should raise an exception in case an error occurs.
        If ``False``, slides will be skipped.
    include_derived: bool, optional
        Whether derived images (DICOM Segmentation or DICOM Parametric Map
        instances) should be considered and included into slides

    Returns
    -------
    List[dicomslide.Slide]
        Digital slides

    """  # noqa: E501
    def bulk_data_uri_handler(
        tag: str,
        vr: str,
        uri: str
    ) -> Union[bytes, None]:
        # Only retrieve ICC Profile
        if tag != '00282000':
            return None
        return client.retrieve_bulkdata(uri)[0]

    # Find VL Whole Slide Microscopy Image instances
    logger.debug('search for slide microscopy image instances')
    search_filters: Dict[str, str] = {}
    if study_instance_uid is not None:
        search_filters['StudyInstanceUID'] = study_instance_uid
    if patient_id is not None:
        search_filters['PatientID'] = patient_id
    if study_id is not None:
        search_filters['StudyID'] = study_id

    # Search for instances per study to speed up subsequent queries.
    if len(search_filters) > 0:
        study_search_results = [
            Dataset.from_json(ds)
            for ds in client.search_for_studies(
                search_filters=search_filters,
                get_remaining=True
            )
        ]
    else:
        study_search_results = [
            Dataset.from_json(ds)
            for ds in client.search_for_studies(get_remaining=True)
        ]
    if len(study_search_results) == 0:
        return []

    lut = defaultdict(list)
    for study in study_search_results:
        current_study_instance_uid = study.StudyInstanceUID
        instance_search_results = []
        # We could search by SOPClassUID directly, but some archives don't
        # support this attribute as a query parameter (although they should
        # according to the standard).
        sm_image_instance_search_results = [
            Dataset.from_json(ds)
            for ds in client.search_for_instances(
                study_instance_uid=current_study_instance_uid,
                search_filters={'Modality': 'SM'}
            )
        ]
        sm_image_instance_search_results = [
            instance
            for instance in sm_image_instance_search_results
            if instance.SOPClassUID == VLWholeSlideMicroscopyImageStorage
        ]
        logger.debug(
            f'found n={len(sm_image_instance_search_results)} '
            'slide microscopy image instances '
            f'for study "{study.StudyInstanceUID}"'
        )
        instance_search_results.extend(sm_image_instance_search_results)

        if include_derived:
            seg_image_instance_search_results = [
                Dataset.from_json(ds)
                for ds in client.search_for_instances(
                    study_instance_uid=current_study_instance_uid,
                    search_filters={'Modality': 'SEG'}
                )
            ]
            seg_image_instance_search_results = [
                instance
                for instance in seg_image_instance_search_results
                if instance.SOPClassUID == SegmentationStorage
            ]
            logger.debug(
                f'found n={len(seg_image_instance_search_results)} '
                'segmentation image instances '
                f'for study "{study.StudyInstanceUID}"'
            )
            instance_search_results.extend(seg_image_instance_search_results)

            pm_image_instance_search_results = [
                Dataset.from_json(ds)
                for ds in client.search_for_instances(
                    study_instance_uid=current_study_instance_uid,
                    search_filters={'Modality': 'OT'}
                )
            ]
            pm_image_instance_search_results = [
                instance
                for instance in sm_image_instance_search_results
                if instance.SOPClassUID == ParametricMapStorage
            ]
            logger.debug(
                f'found n={len(pm_image_instance_search_results)} '
                'parametric map image instances '
                f'for study "{study.StudyInstanceUID}"'
            )
            instance_search_results.extend(pm_image_instance_search_results)
            logger.debug(
                f'found n={len(instance_search_results)} image instances '
                f'for study "{study.StudyInstanceUID}"'
            )

        logger.debug('filter image instances based on metadata')
        for instance in instance_search_results:
            logger.debug(
                'retrieve metadata for '
                f'image instance "{instance.SOPInstanceUID}"'
            )
            metadata = Dataset.from_json(
                client.retrieve_instance_metadata(
                    study_instance_uid=current_study_instance_uid,
                    series_instance_uid=instance.SeriesInstanceUID,
                    sop_instance_uid=instance.SOPInstanceUID,
                ),
                bulk_data_uri_handler=bulk_data_uri_handler
            )

            # These checks should not be necessary, because we are using these
            # attributes to search for studies in the first place. However,
            # some archives get this wrong. Safety first!
            if study_instance_uid is not None:
                if metadata.StudyInstanceUID != study_instance_uid:
                    logger.debug(
                        f'skip image "{metadata.SOPInstanceUID}" because it '
                        'does not match the Study Instance UID '
                        f'"{study_instance_uid}"'
                    )
                    continue
            if patient_id is not None:
                if metadata.PatientID != patient_id:
                    logger.debug(
                        f'skip image "{metadata.SOPInstanceUID}" because it '
                        f'does not match the Patient ID "{patient_id}"'
                    )
                    continue
            if study_id is not None:
                if metadata.StudyID != study_id:
                    logger.debug(
                        f'skip image "{metadata.SOPInstanceUID}" because it '
                        f'does not match the Study ID "{study_id}"'
                    )
                    continue

            try:
                current_container_id = metadata.ContainerIdentifier
            except AttributeError:
                logger.debug(
                    f'skip image "{metadata.SOPInstanceUID}" because it '
                    'does not have a Container Identifier attribute'
                )
                continue

            if container_id is not None:
                if current_container_id != container_id:
                    logger.debug(
                        f'skip image "{metadata.SOPInstanceUID}" because it '
                        f'does not match Container Identifier "{container_id}"'
                    )
                    continue

            try:
                current_frame_of_reference_uid = metadata.FrameOfReferenceUID
            except AttributeError:
                logger.warning(
                    f'skip image "{metadata.SOPInstanceUID}" because it '
                    'does not have a Frame of Reference UID attribute'
                )
                continue

            logger.debug(
                f'assign image "{metadata.SOPInstanceUID}" to slide with '
                f'Study Instance UID "{current_study_instance_uid}", '
                f'Container Identifier "{current_container_id}", and '
                f'Frame of Reference UID "{current_frame_of_reference_uid}"'
            )
            key = (
                current_study_instance_uid,
                current_container_id,
                current_frame_of_reference_uid,
            )
            lut[key].append(metadata)

    logger.debug(f'create n={len(lut)} slide objects')
    found_slides = []
    for _, image_metadata in lut.items():
        ref_image = image_metadata[0]
        logger.debug(
            f'create slide for images of study "{ref_image.StudyInstanceUID}" '
            f'for container "{ref_image.ContainerIdentifier}" '
            f'in frame of reference "{ref_image.FrameOfReferenceUID}"'
        )
        try:
            slide = Slide(
                client=client,
                image_metadata=image_metadata,
                max_frame_cache_size=max_frame_cache_size,
                pyramid_tolerance=pyramid_tolerance
            )
            found_slides.append(slide)
        except Exception as error:
            if fail_on_error:
                raise
            else:
                ref_image = image_metadata[0]
                logger.warning(
                    'failed to create slide for images of study '
                    f'"{ref_image.StudyInstanceUID}" for container '
                    f'"{ref_image.ContainerIdentifier}" in frame of reference '
                    f'"{ref_image.FrameOfReferenceUID}" due to the following '
                    f'error: {error}'
                )

    return found_slides
