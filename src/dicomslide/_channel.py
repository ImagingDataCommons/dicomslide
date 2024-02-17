from typing import Callable, List, NamedTuple, Tuple

from pydicom.dataset import Dataset
from pydicom.uid import (
    ParametricMapStorage,
    SegmentationStorage,
    VLWholeSlideMicroscopyImageStorage,
)

from dicomslide.enum import ChannelTypes


class _ChannelInfo(NamedTuple):

    channel_type: ChannelTypes
    channel_identifier: str
    sop_class_uid: str
    sop_instance_uid: str


def _get_channel_info(image: Dataset) -> Tuple[
    List[_ChannelInfo],
    Callable[[Dataset], str]
]:
    """Get information about channels in an image.

    The exact definition of a channel depends on the DICOM Information Object
    Definition (IOD). In case of DICOM VL Whole Slide Microscopy Image, a
    channel is an optical path and individual channels are described by items
    of the Optical Path Sequence attribute. In case of DICOM Segmentation, a
    channel is a segment and individual channels are described by items of the
    Segment Sequence attribute.

    Parameters
    ----------
    image: pydicom.Dataset
        Metadata of a DICOM image

    Returns
    -------
    channel_info: List[dicomslide._channel._ChannelInfo]
        Information for each identified channel
    get_referenced_identifier: Callable[[pydicom.Dataset], str]
        Callable to determine the identifier for an individual frame, given
        an item of either the Shared Functional Groups Sequence or the
        Per-Frame Functional Groups Sequence.

    """
    sop_class_uid = image.SOPClassUID
    sop_instance_uid = image.SOPInstanceUID
    channel_identifiers: Tuple[str, ...]
    get_referenced_identifier: Callable[[Dataset], str]
    channel_type: ChannelTypes
    if sop_class_uid == VLWholeSlideMicroscopyImageStorage:
        def get_referenced_identifier(item: Dataset) -> str:
            # Optical Path Identification Macro
            return str(
                item
                .OpticalPathIdentificationSequence[0]
                .OpticalPathIdentifier
            )

        channel_type = ChannelTypes.OPTICAL_PATH
        channel_identifiers = tuple([
            str(item.OpticalPathIdentifier)
            for item in image.OpticalPathSequence
        ])

    elif sop_class_uid == SegmentationStorage:
        def get_referenced_identifier(item: Dataset) -> str:
            # Segmentation Macro
            return str(
                item
                .SegmentIdentificationSequence[0]
                .ReferencedSegmentNumber
            )

        channel_type = ChannelTypes.SEGMENT
        channel_identifiers = tuple([
            str(item.SegmentNumber)
            for item in image.SegmentSequence
        ])

    elif sop_class_uid == ParametricMapStorage:
        def get_referenced_identifier(item: Dataset) -> str:
            return '|'.join([
                str(item.LUTLabel)
                for item in item.RealWorldValueMappingSequence
            ])

        channel_type = ChannelTypes.PARAMETER
        shared_item = image.SharedFunctionalGroupsSequence[0]
        if hasattr(shared_item, 'RealWorldValueMappingSequence'):
            channel_identifiers = (get_referenced_identifier(shared_item), )
        elif hasattr(image, 'PerFrameFunctionalGroupsSequence'):
            channel_identifiers = tuple(set([
                get_referenced_identifier(item)
                for item in image.PerFrameFunctionalGroupsSequence
            ]))
        else:
            raise ValueError(
                'DICOM Parametric Map must include attribute '
                'Real World Value Mapping Sequence either in the'
                'Functional Groups Sequence or in the Per-Frame Functional '
                'Groups Sequence.'
            )
    else:
        raise ValueError(
            'Unsupported DICOM SOP Class "{image.SOPClassUID}".'
        )
    return (
        [
            _ChannelInfo(
                channel_type,
                identifier,
                sop_class_uid,
                sop_instance_uid
            )
            for identifier in channel_identifiers
        ],
        get_referenced_identifier
    )
