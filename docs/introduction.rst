.. _introduction:

Introduction
============

The ``dicomslide`` build distribution provides an application programming interface (API) for querying and retrieving whole slide images in DICOM format from local files or over network via a unified application programming interface.

The :mod:`dicomslide` Python package contains several classes and functions.

Design
------

The :mod:`dicomslide` Python package contains several data structures that abstract whole slide images.
The core data structure of the library is the :class:`dicomslide.Slide` class, which represents a collection of `DICOM VL Whole Slide Microscopy Image <https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_A.32.8.html>`_ instances that share the same Container Identifier and Frame of Reference UID, i.e., that were acquired for the same physical class slide and are spatially aligned.
The interface exposed by the ``dicomslide.Slide`` class abstracts the organization of tiled images that belong to the slide (:class:`dicomslide.TiledImage`) and the associated total pixel matrices (:class:`dicomslide.TotalPixelMatrix`), which form a multi-resolution image pyramid (:class:`dicomslide.Pyramid`).

Application programming interface
---------------------------------

The library leverages the Python `dicomweb-client <https://dicomweb-client.readthedocs.io/en/latest/>`_ library to efficiently search for and retrieve whole slide image data from heterogeneous sources using the interface defined by the `dicomweb_client.DICOMClient <https://dicomweb-client.readthedocs.io/en/latest/package.html#dicomweb_client.api.DICOMClient>`_ protocol.
Importantly, the library does not load the entire images into memory, but dynamically retrieves only the image frames (tiles) that are needed for a requested image region.

The `dicomweb_client.DICOMwebClient <https://dicomweb-client.readthedocs.io/en/latest/package.html#dicomweb_client.api.DICOMwebClient>`_ and `dicomweb_client.DICOMfileClient <https://dicomweb-client.readthedocs.io/en/latest/package.html#dicomweb_client.api.DICOMfileClient>`_ classes both implement that protocol and thereby enable efficient reading of whole slide images in DICOM format from remote archives using the DICOMweb RESTful API (see `DICOM Part 18 <https://dicom.nema.org/medical/dicom/current/output/chtml/part18/PS3.18.html>`_) and from local DICOM files (see `DICOM Part 10 <https://dicom.nema.org/medical/dicom/current/output/chtml/part10/PS3.10.html>`_), respectively.
