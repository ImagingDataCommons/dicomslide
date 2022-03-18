from io import BytesIO

from pydicom.dataset import Dataset

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
        return dataset.ImageType[2] in (
            ImageFlavors.VOLUME.value,
            ImageFlavors.THUMBNAIL.value,
        )
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
        return dataset.ImageType[2] == ImageFlavors.LABEL.value
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
        return dataset.ImageType[2] == ImageFlavors.OVERVIEW.value
    return False


def encode_dataset(dataset: Dataset) -> bytes:
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
    with BytesIO() as fp:
        clone.save_as(fp)
        encoded_value = fp.getvalue()
    return encoded_value
