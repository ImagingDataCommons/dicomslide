from io import BytesIO
from typing import Optional, Union

import highdicom as hd
from pydicom.dataset import Dataset
from pydicom.filewriter import dcmwrite
from pydicom.sr.coding import Code

from dicomslide.enum import ImageFlavors


def is_image(dataset: Dataset) -> bool:
    """Determine whether a dataset is an image.

    Parameters
    ----------
    dataset: pydicom.dataset.Dataset
        Dataset

    Returns
    -------
    bool
        Whether dataset is an image

    """
    return all([
        hasattr(dataset, 'BitsAllocated'),
        hasattr(dataset, 'Columns'),
        hasattr(dataset, 'Rows'),
        hasattr(dataset, 'PhotometricInterpretation'),
    ])


def is_tiled_image(dataset: Dataset) -> bool:
    """Determine whether a dataset is a tiled image.

    Parameters
    ----------
    dataset: pydicom.dataset.Dataset
        Dataset

    Returns
    -------
    bool
        Whether dataset is a tiled image

    """
    if is_image(dataset):
        return all([
            hasattr(dataset, 'TotalPixelMatrixColumns'),
            hasattr(dataset, 'TotalPixelMatrixRows'),
        ])
    return False


def is_volume_image(dataset: Dataset) -> bool:
    """Determine whether a dataset is a VOLUME or THUMBNAIL image.

    Parameters
    ----------
    dataset: pydicom.dataset.Dataset
        Dataset

    Returns
    -------
    bool
        Whether dataset is a VOLUME or THUMBNAIL image

    """
    if is_image(dataset):
        try:
            return dataset.ImageType[2] in (
                ImageFlavors.VOLUME.value,
                ImageFlavors.THUMBNAIL.value,
            )
        except IndexError:
            return True
    return False


def is_label_image(dataset: Dataset) -> bool:
    """Determine whether a dataset is a LABEL image.

    Parameters
    ----------
    dataset: pydicom.dataset.Dataset
        Dataset

    Returns
    -------
    bool
        Whether dataset is a LABEL image

    """
    if is_image(dataset):
        try:
            return dataset.ImageType[2] == ImageFlavors.LABEL.value
        except IndexError:
            return False
    return False


def is_overview_image(dataset: Dataset) -> bool:
    """Determine whether a dataset is an OVERVIEW image.

    Parameters
    ----------
    dataset: pydicom.dataset.Dataset
        Dataset

    Returns
    -------
    bool
        Whether dataset is an OVERVIEW image

    """
    if is_image(dataset):
        try:
            return dataset.ImageType[2] == ImageFlavors.OVERVIEW.value
        except IndexError:
            return False
    return False


def _encode_dataset(dataset: Dataset) -> bytes:
    """Encode DICOM dataset.

    Parameters
    ----------
    dataset: pydicom.dataset.Dataset
        Dataset

    Returns
    -------
    bytes
        Binary encoded dataset

    """
    clone = Dataset(dataset)
    clone.is_little_endian = True
    clone.is_implicit_VR = False
    # Only encode the Data Set.
    if hasattr(clone, 'file_meta'):
        del clone.file_meta
    # Some servers may erroursly include these Data Elements into the Data Set
    # instead of the File Meta Information.
    for element in clone.group_dataset(0x0002):
        del clone[element.tag]
    with BytesIO() as fp:
        dcmwrite(fp, clone, write_like_original=True)
        encoded_value = fp.getvalue()
    return encoded_value


def does_optical_path_item_match(
    item: Dataset,
    identifier: Optional[str] = None,
    description: Optional[str] = None,
    illumination_wavelength: Optional[float] = None,
) -> bool:
    """Check whether an optical path item matches.

    Parameters
    ----------
    item: pydicom.Dataset
        Item of the Optical Path Sequence
    identifier: Union[str, None], optional
        Optical path identifier
    description: Union[str, None], optional
        Optical path description
    illumination_wavelength: Union[float, None], optional
        Illumination wave length

    Returns
    -------
    bool
        Whether item matches

    """  # noqa: E501
    matches = []
    if identifier is not None:
        matches.append(item.OpticalPathIdentifier == identifier)
    if description is not None:
        if hasattr(item, 'OpticalPathDescription'):
            matches.append(item.OpticalPathDescription == description)
        else:
            matches.append(False)
    if illumination_wavelength is not None:
        if hasattr(item, 'IlluminationWaveLength'):
            matches.append(
                item.IlluminationWaveLength == illumination_wavelength
            )
        else:
            matches.append(False)
    if len(matches) == 0:
        return True
    return any(matches)


def does_specimen_description_item_match(
    item: Dataset,
    specimen_stain: Optional[Union[hd.sr.CodedConcept, Code]] = None
) -> bool:
    """Check whether a specimen description item matches.

    Parameters
    ----------
    item: pydicom.Dataset
        Item of the Specimen Description Sequence
    specimen_stain: Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept, None], optional
        Specimen stain substance

    Returns
    -------
    bool
        Whether item matches

    """  # noqa: E501
    matches = []
    if specimen_stain is not None:
        description = hd.SpecimenDescription.from_dataset(item)
        is_specimen_staining_described = False
        for step in description.specimen_preparation_steps:
            procedure = step.processing_procedure
            if isinstance(procedure, hd.SpecimenStaining):
                is_specimen_staining_described = True
                substances = [
                    item
                    for item in procedure.substances
                    if isinstance(item, hd.sr.CodedConcept)
                ]
                matches.append(specimen_stain in substances)
        if not is_specimen_staining_described:
            matches.append(False)
    if len(matches) == 0:
        return True
    return any(matches)


def does_segment_item_match(
    item: Dataset,
    number: Optional[int] = None,
    label: Optional[str] = None,
    property_category: Optional[Union[hd.sr.CodedConcept, Code]] = None,
    property_type: Optional[Union[hd.sr.CodedConcept, Code]] = None,
) -> bool:
    """Check whether a segment item matches.

    Parameters
    ----------
    item: pydicom.Dataset
        Item of the Segment Sequence
    number: Union[int, None], optional
        Segment number
    label: Union[int, None], optional
        Segment label
    property_category: Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept, None], optional
        Segmented property category
    property_type: Union[pydicom.sr.coding.Code, highdicom.sr.CodedConcept, None], optional
        Segmented property type

    Returns
    -------
    bool
        Whether item matches

    """  # noqa: E501
    matches = []
    if number is not None:
        matches.append(item.SegmentNumber == number)
    if label is not None:
        matches.append(item.SegmentLabel == label)
    if property_category is not None:
        code_item = hd.sr.CodedConcept.from_dataset(
            item.SegmentedPropertyCategoryCodeSequence[0]
        )
        matches.append(code_item == property_category)
    if property_type is not None:
        code_item = hd.sr.CodedConcept.from_dataset(
            item.SegmentedPropertyTypeCodeSequence[0]
        )
        matches.append(code_item == property_type)
    if len(matches) == 0:
        return True
    return any(matches)
