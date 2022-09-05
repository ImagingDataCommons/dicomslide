.. _user-guide:

User guide
==========

Reading whole slide images in DICOM format using the :mod:`dicomslide` package.

Constructing a DICOM Client
---------------------------

Use `dicomweb_client.DICOMfileClient <https://dicomweb-client.readthedocs.io/en/latest/package.html#dicomweb_client.api.DICOMfileClient>`_ to read whole slide images from DICOM files stored on a file system:

.. code-block:: python

    import dicomweb_client

    client = dicomweb_client.DICOMfileClient(url='file:///tmp/images')

Use `dicomweb_client.DICOMwebClient <https://dicomweb-client.readthedocs.io/en/latest/package.html#dicomweb_client.api.DICOMwebClient>`_ to read whole slide images over network using DICOMweb services:

.. code-block:: python

    import dicomweb_client

    client = dicomweb_client.DICOMwebClient(url='http://myserver.com/dicomweb')

.. _user-guide-dicomslide-api:

Reading images using dicomslide API
-----------------------------------

.. code-block:: python

    import dicomslide
    import numpy as np
    from matplotlib import pyplot as plt

    found_slides = dicomslide.find_slides(client, container_id='S22-ABC-123')
    assert len(found_slides) == 1
    slide = found_slides[0]

    print(slide.num_channels)
    print(slide.num_focal_planes)
    print(slide.num_levels)
    print(slide.total_pixel_matrix_dimensions)
    print(slide.downsampling_factors)
    print(slide.label_images)
    print(slide.get_volume_images(channel_index=0, focal_plane_index=0))

    region: np.ndarray = slide.get_image_region(
        pixel_indices=(0, 0),
        level=-1,
        size=(512, 512),
        channel_index=0,
        focal_plane_index=0
    )
    plt.imshow(region)
    plt.show()

.. _user-guide-openslide-api:

Reading images using openslide API
----------------------------------

The library also exposes an `OpenSlide <https://openslide.org/api/python/>`_ interface (:class:`dicomslide.OpenSlide`), which is intended as an API wrapper around a :class:`dicomslide.Slide` instance and to be used as a drop-in replacement for an `openslide.OpenSlide <https://openslide.org/api/python/#openslide.OpenSlide>`_ instance:

.. code-block:: python

    from PIL import Image

    openslide = dicomslide.OpenSlide(slide)

    print(openslide.level_count)
    print(openslide.dimensions)
    print(openslide.level_dimensions)
    print(openslide.level_downsamples)
    print(openslide.properties)
    print(openslide.associated_images)

    thumbnail: Image.Image = openslide.get_thumbnail(size=(50, 100))
    thumbnail.show()

Note that the OpenSlide API only supports 2D color images.
For images with multiple channels or Z-planes, only the standard dicomslide API can be used.
