from io import BytesIO

from pydicom.dataset import Dataset
from pydicom.filewriter import dcmwrite

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
