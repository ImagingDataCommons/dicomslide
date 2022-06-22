[![Build Status](https://github.com/herrmannlab/dicomslide/actions/workflows/run_unit_tests.yml/badge.svg)](https://github.com/herrmannlab/dicomslide/actions)

# dicomslide

A Python library for reading whole slide images in DICOM format from local files or over network via a unified application programming interface.


## Design

The central data structure of the library is the ``dicomslide.Slide`` class, which represents a collection of DICOM VL Whole Slide Microscopy Image instances that share the same Container Identifier and Frame of Reference UID, i.e., that were acquired for the same physical class slide and are spatially aligned.
The interface exposed by the ``Slide`` class abstracts the organization of tiled images that belong to the slide (implemented in class ``dicomslide.TiledImage``) and the associated total pixel matrices (implemented in class ``dicomslide.TotalPixelMatrix``), which form a multi-resolution image pyramid (implemented in class ``dicomslide.Pyramid``).

## Application programming interface

The library leverages the Python [dicomweb-client](https://dicomweb-client.readthedocs.io/en/latest/) library to efficiently search for and retrieve whole slide image data from heterogeneous sources using the interface defined by the [dicomweb_client.DICOMClient](https://dicomweb-client.readthedocs.io/en/latest/package.html#dicomweb_client.api.DICOMClient) protocol.
Importantly, the library does not load the entire images into memory, but dynamically retrieves only the image frames (tiles) that are needed for a requested image region.

The [dicomweb_client.DICOMwebClient](https://dicomweb-client.readthedocs.io/en/latest/package.html#dicomweb_client.api.DICOMwebClient) and [dicomweb_client.DICOMfileClient](https://dicomweb-client.readthedocs.io/en/latest/package.html#dicomweb_client.api.DICOMfileClient) classes both implement that protocol and thereby enable efficient reading of whole slide images in DICOM format from remote archives using the DICOMweb RESTful API and from local DICOM Part10 files, respectively.

```python
import dicomweb_client
import dicomslide
import numpy as np
from matplotlib import pyplot as plt

client = dicomweb_client.DICOMfileClient(base_dir='/tmp/images')

found_slides = dicomslide.find_slides(client, container_id='S22-ABC-123')
assert len(found_slides) == 1
slide = found_slides[0]

print(slide.num_optical_paths)
print(slide.num_focal_planes)
print(slide.num_levels)
print(slide.total_pixel_matrix_dimensions)
print(slide.downsampling_factors)
print(slide.label_images)
print(slide.get_volume_images(optical_path_index=0, focal_plane_index=0))

region: np.ndarray = slide.get_image_region(
    pixel_indices=(0, 0),
    level=-1,
    size=(512, 512),
    optical_path_index=0,
    focal_plane_index=0
)
plt.imshow(region)
plt.show()
```

### OpenSlide API

The library also exposes an [OpenSlide](https://openslide.org/api/python/) interface (implemented in class ``dicomslide.OpenSlide``), which is intended as an API wrapper around an ``dicomslide.Slide`` instance to be used as a drop-in replacement for an [openslide.OpenSlide](https://openslide.org/api/python/#openslide.OpenSlide) instance:

```python
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
```

Note that the OpenSlide API only supports 2D color images.
For images with multiple channels or Z-planes, only the standard dicomslide API can be used.
