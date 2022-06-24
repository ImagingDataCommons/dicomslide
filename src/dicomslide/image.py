import itertools
import logging
from hashlib import sha256
from typing import Tuple, Union

import highdicom as hd
import numpy as np
from dicomweb_client import DICOMClient
from pydicom.dataset import Dataset
from pydicom.uid import (
    ParametricMapStorage,
    SegmentationStorage,
    VLWholeSlideMicroscopyImageStorage,
)

from dicomslide._channel import _get_channel_info
from dicomslide.enum import ChannelTypes
from dicomslide.matrix import TotalPixelMatrix
from dicomslide.tile import compute_frame_positions
from dicomslide.utils import _encode_dataset

logger = logging.getLogger(__name__)


class TiledImage:

    """A tiled DICOM image.

    An instance of the class represents a tiled DICOM image instance and
    provides methods for convenient and efficient access of image metadata and
    pixel data from a DICOMweb server (or another source for which the
    :class:`dicomweb_client.DICOMClient` protocol has been implemented).

    A tiled image is hereby defined as a DICOM image instance that contains
    the Total Pixel Matrix Rows and Total Pixel Matrix Columns attributes.

    The class is designed to be independent of a particular DICOM Information
    Object Definition (IOD) or SOP Class and support various different types of
    DICOM images, including VL Whole Slide Microscopy Image, Segmentation, and
    Parametric Map.

    Each image is associated with one or more
    :class:`dicomslide.TotalPixelMatrix` instances, one for each unique
    combination of channel and focal plane.
    The definition of a channel is specific to a particular IOD. For example,
    in case of VL Whole Slide Microscopy Image, a channel corresponds to an
    optical path, whereas in case of a Segmentation, a channel corresponds to a
    segment.

    Examples
    --------
    >>> image = TiledImage(...)
    >>> print(image.metadata)  # pydicom.Dataset
    >>> print(image.metadata.BitsAllocated)
    >>> print(image.metadata.TotalPixelMatrixRows)
    >>> pixel_matrix = image.get_tota_pixel_matrix(channel_index=0)
    >>> print(pixel_matrix.dtype)
    >>> print(pixel_matrix.shape)
    >>> print(pixel_matrix[:1000, 350:750, :])  # numpy.ndarray

    """

    def __init__(
        self,
        client: DICOMClient,
        image_metadata: Dataset,
        max_frame_cache_size: int = 6
    ) -> None:
        """Construct object.

        Parameters
        ----------
        client: dicomweb_client.api.DICOMClient
            DICOMweb client
        image_metadata: pydicom.dataset.Dataset
            Metadata of a tiled DICOM image
        max_frame_cache_size: int, optional
            Maximum number of frames that should be cached to avoid repeated
            retrieval requests

        Note
        ----
        If `image_metadata` is the metadata of a color image, it should contain
        the ICC Profile element to enable color management. The value of this
        element may be considered bulkdata and therefore may have to be
        retrieved separately over DICOMweb.

        """
        self._client = client
        self._metadata = image_metadata
        # The Total Pixel Matrix Focal Planes attribute must be present in case
        # of Dimension Organization TILED_FULL and may be present otherwise.
        # If it is missing, we need to determine the position of focal planes by
        # either looking up the values in the items in the Per-Frame Functional
        # Groups Sequence attribute or by computing the values if the attribute
        # is not included in the dataset.
        try:
            self._number_of_focal_planes = getattr(
                self._metadata,
                'TotalPixelMatrixFocalPlanes'
            )
            self._frame_positions = None
        except AttributeError:
            self._frame_positions = compute_frame_positions(self._metadata)
            focal_plane_indices = self._frame_positions[3]
            self._number_of_focal_planes = len(np.unique(focal_plane_indices))

        channels, get_channel_identifier = _get_channel_info(self._metadata)
        self._channel_identifiers = tuple([ch.identifier for ch in channels])
        self._number_of_channels = len(self._channel_identifiers)

        iterator = itertools.product(
            range(self._number_of_channels),
            range(self._number_of_focal_planes),
        )
        self._total_pixel_matrix_lut = {
            (channel_index, focal_plane_index): TotalPixelMatrix(
                client=self._client,
                image_metadata=self._metadata,
                channel_index=channel_index,
                focal_plane_index=focal_plane_index,
                max_frame_cache_size=max_frame_cache_size
            )
            for channel_index, focal_plane_index in iterator
        }

        encoded_dataset = _encode_dataset(self._metadata)
        self._quickhash = sha256(encoded_dataset).hexdigest()

        self._ref2pix_transformer: Union[
            hd.spatial.ReferenceToPixelTransformer,
            None
        ] = None

    def __hash__(self) -> int:
        return hash(self._quickhash)

    @property
    def metadata(self) -> Dataset:
        """pydicom.dataset.Dataset: Image metadata"""
        return self._metadata

    @property
    def num_channels(self) -> int:
        """int: Number of channels"""
        return self._number_of_channels

    def get_pixel_indices(
        self,
        offset: Tuple[float, float],
        focal_plane_index: int
    ) -> Tuple[int, int]:
        """Get indices into total pixel matrix for a given slide position.

        Parameters
        ----------
        offset: Tuple[float, float]
            Zero-based (x, y) offset in the slide coordinate system
        focal_plane_index: int
            Zero-based index into focal planes along depth direction from the
            glass slide towards the coverslip in the slide coordinate system
            specified by the Z Offset in Slide Coordinate System attribute.
            Values must be in the range [0, Total Pixel Matrix Focal Planes)

        Returns
        -------
        Tuple[int, int]
            Zero-based (row, column) position in the total pixel matrix

        Note
        ----
        Pixel position may be negativ or extend beyond the size of the total
        pixel matrix if slide position at `offset` does fall into a region on
        the slide that was not imaged.

        """
        focal_plane_offset = self.get_focal_plane_offset(focal_plane_index)
        if self._ref2pix_transformer is None:
            image_orientation = self.metadata.ImageOrientationSlide
            image_origin = self.metadata.TotalPixelMatrixOriginSequence[0]
            image_position = (
                float(image_origin.XOffsetInSlideCoordinateSystem),
                float(image_origin.YOffsetInSlideCoordinateSystem),
                focal_plane_offset / 10**3,
            )
            pixel_spacing = (
                float(
                    self
                    .metadata
                    .SharedFunctionalGroupsSequence[0]
                    .PixelMeasuresSequence[0]
                    .PixelSpacing[0]
                ),
                float(
                    self
                    .metadata
                    .SharedFunctionalGroupsSequence[0]
                    .PixelMeasuresSequence[0]
                    .PixelSpacing[1]
                ),
            )
            self._ref2pix_transformer = hd.spatial.ReferenceToPixelTransformer(
                image_orientation=image_orientation,
                image_position=image_position,
                pixel_spacing=pixel_spacing,
            )
        slide_coordinates = np.array([
            [offset[0], offset[1], focal_plane_offset / 10**3]
        ])
        pixel_indices = self._ref2pix_transformer(slide_coordinates)
        return (
            int(pixel_indices[0, 1]),
            int(pixel_indices[0, 0]),
        )

    def get_rotation(self) -> float:
        """Get angle to rotate image such that it aligns with slide.

        We want to align the image with the slide coordinate system such that
        the axes of the total pixel matrix are aligned with the X and Y axes
        of the slide coordinate system to ensure that spatial coordinates of
        graphic region of interest (ROI) annotations and are aligned with the
        source image region.

        Returns
        -------
        float
            Counterclockwise angle of rotation

        """
        image_orientation = self.metadata.ImageOrientationSlide
        radians = np.arctan2(-image_orientation[3], image_orientation[0])
        degrees = radians * 180.0 / np.pi
        degrees -= 180.0

        # Images are expected to be rotated in plane parallel to the slide
        # surface and the rows and columns of the image are expected to be
        # parallel to the axes of the slide.
        if degrees not in (-0.0, -90.0, -180.0, -270.0):
            logger.warning(
                'encountered unexpected image orientation: '
                f'{image_orientation}: {degrees}'
            )

        return degrees

    def get_channel_index(self, channel_identifier: str) -> int:
        """Get index of a channel.

        The nature of the channel is specific to the SOP Class for the image.
        For example, in case of DICOM VL Whole Slide Microscopy Image, a
        channel is an optical path and in case of a DICOM Segmentation, a
        channel is a segment.

        Parameters
        ----------
        channel_identifier: str
            Identifier of a channel

        Returns
        -------
        int
            Zero-based index into channels along the direction defined by
            successive items of the corresponding attribute.

        Raises
        ------
        ValueError
            When no channel is found for `channel_identifier`

        """
        try:
            index = self._channel_identifiers.index(channel_identifier)
        except ValueError:
            raise ValueError(
                f'Image "{self._metadata.SOPInstanceUID}" does not have an '
                f'optical path with identifier "{channel_identifier}".'
            )
        return index

    @property
    def channel_type(self) -> ChannelTypes:
        """dicomslide.ChannelTypes: type of channels"""
        sop_class_uid = self._metadata.SOPClassUID
        sop_instance_uid = self._metadata.SOPInstanceUID
        lut = {
            VLWholeSlideMicroscopyImageStorage: ChannelTypes.OPTICAL_PATH,
            SegmentationStorage: ChannelTypes.SEGMENT,
            ParametricMapStorage: ChannelTypes.PARAMETER,
        }
        try:
            return lut[sop_class_uid]
        except IndexError:
            raise ValueError(
                'Could not determine channel type for '
                f'image "{sop_instance_uid}" of SOP Class "{sop_class_uid}".'
            )

    def get_channel_identifier(self, channel_index: int) -> str:
        """Get identifier of a channel.

        Parameters
        ----------
        channel_index: int
            Zero-based index into channels along the direction defined by
            successive items of the appropriate DICOM attribute.

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
            return self._channel_identifiers[channel_index]
        except IndexError:
            raise ValueError(
                f'Image "{self._metadata.SOPInstanceUID}" does not have a '
                f'channel {channel_index}.'
            )

    @property
    def num_focal_planes(self) -> int:
        """int: Number of focal planes"""
        return self._number_of_focal_planes

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
            specified by the Z Offset in Slide Coordinate System attribute.
            Values must be in the range [1, Total Pixel Matrix Focal Planes]

        Raises
        ------
        ValueError
            When no focal plane is found for `focal_plane_offset`

        """
        if self._frame_positions is None:
            self._frame_positions = compute_frame_positions(self._metadata)
        slide_positions = self._frame_positions[1]
        focal_plane_indices = self._frame_positions[3]
        index = slide_positions[:, 2] == focal_plane_offset
        if np.sum(index) == 0:
            raise ValueError(
                f'Image "{self._metadata.SOPInstanceUID}" does not have a '
                f'focal plane with z offset {focal_plane_offset}.'
            )
        return focal_plane_indices[index][0]

    def get_focal_plane_offset(self, focal_plane_index: int) -> float:
        """Get z offset in slide coordinate system of a focal plane.

        Parameters
        ----------
        focal_plane_index: int
            Zero-based index into focal planes along depth direction from the
            glass slide towards the coverslip in the slide coordinate system
            specified by the Z Offset in Slide Coordinate System attribute.
            Values must be in the range [0, Total Pixel Matrix Focal Planes).

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
        # TODO: cache focal plane offsets
        origin_item = self._metadata.TotalPixelMatrixOriginSequence[0]
        dim_org_type = hd.DimensionOrganizationTypeValues(
            getattr(
                self._metadata,
                'DimensionOrganizationType',
                hd.DimensionOrganizationTypeValues.TILED_SPARSE.value
            )
        )
        pixel_measures_item = (
            self._metadata
            .SharedFunctionalGroupsSequence[0]
            .PixelMeasuresSequence[0]
        )
        if dim_org_type == hd.DimensionOrganizationTypeValues.TILED_FULL:
            # In case of Dimension Organization Type TILED_FULL, the absolute
            # position of focal planes is not explicitly specified and the
            # plane is assumed to be located at the slide surface.
            z_offset = float(
                getattr(
                    origin_item,
                    'ZOffsetInSlideCoordinateSytem',
                    '0.0'
                )
            )
            spacing_between_slices = float(
                getattr(pixel_measures_item, 'SpacingBetweenSlices', 1.0)
            )
            num_slices = self._metadata.TotalPixelMatrixFocalPlanes
            if focal_plane_index > num_slices:
                raise ValueError(
                    f'Image "{self._metadata.SOPInstanceUID}" does not have a '
                    f'focal plane with index {focal_plane_index}.'
                )
            if num_slices == 1:
                return z_offset
            else:
                # Note that Spacing Between Slices has millimeter unit while
                # the Z Offset in Slide Coordinate System has micrometer unit.
                return z_offset + (spacing_between_slices * num_slices * 10**3)
        else:
            if self._frame_positions is None:
                self._frame_positions = compute_frame_positions(self._metadata)
            slide_positions = self._frame_positions[1]
            focal_plane_indices = self._frame_positions[3]
            index = focal_plane_indices == focal_plane_index
            if np.sum(index) == 0:
                raise ValueError(
                    f'Image "{self._metadata.SOPInstanceUID}" does not have a '
                    f'focal plane with index {focal_plane_index}.'
                )
            return slide_positions[index, 2][0]

    def get_total_pixel_matrix(
        self,
        channel_index: int = 0,
        focal_plane_index: int = 0
    ) -> TotalPixelMatrix:
        """Get total pixel matrix for a given optical path and focal plane.

        Parameters
        ----------
        channel_index: int, optional
            Zero-based index into channels along the direction defined by
            successive items of the appropriate DICOM attribute.
        focal_plane_index: int, optional
            Zero-based index into focal planes along depth direction from the
            glass slide towards the coverslip in the slide coordinate system
            specified by the Z Offset in Slide Coordinate System attribute.
            Values must be in the range [0, Total Pixel Matrix Focal Planes).

        Returns
        -------
        dicomslide.TotalPixelMatrix
            Total Pixel Matrix

        """
        if (
            channel_index < 0 or
            channel_index >= self.num_channels
        ):
            raise ValueError(
                'Argument "channel_index" must be a zero-based index '
                f'in range [0, {self.num_channels}).'
            )
        if (
            focal_plane_index < 0 or
            focal_plane_index >= self.num_focal_planes
        ):
            raise ValueError(
                'Argument "focal_plane_index" must be a zero-based index '
                f'in range [0, {self.num_focal_planes}).'
            )

        key = (channel_index, focal_plane_index)
        try:
            return self._total_pixel_matrix_lut[key]
        except KeyError:
            raise ValueError(
                'Could not find a total pixel matrix for '
                f'optical path {channel_index} and '
                f'focal plane {focal_plane_index}.'
            )
