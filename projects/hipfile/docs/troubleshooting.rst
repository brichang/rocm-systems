.. meta::
  :description: Troubleshooting hipFile
  :keywords: hipFile, troubleshooting

***************
Troubleshooting
***************

This page provides guidance on troubleshooting common hipFile issues.

Required software
===========================

Use the ``ais-check`` utility to verify the system has the required software
components necessary for hipFile's fastpath. If any of the required components
are missing, hipFile will use the fallback path.

.. code-block:: none

  $ ais-check
  AIS support in:
          Kernel P2PDMA support   : True
          HIP runtime             : True
          amdgpu                  : True

Backing Storage
===============

Currently hipFile's fastpath is only supported on:

 - raw NVMe block devices
 - ext4 on an NVMe block device
 - xfs on an NVMe block device

hipFile will use the fallback path for all other storage.

Alignment and IO size
=====================

Each IO request must meet alignment and size requirements of the underlying file
system and storage device for hipFile to use the fastpath.. ``statx`` can be
used to determine the offset alignment and io size requirements
(``stx_dio_offset_align``), and memory alignment requirement
(``stx_dio_mem_align``) of the storage.

If an IO request does not meet alignment and size requirements, hipFile will use
the fallback path. ``ais-stasts`` can be used to determine if IO requests are
using the fallback path.

IO statistics
=============

The ``ais-stats`` utility can be used to display hipFile IO statistics. These
stats aid clients in determining if hipFile is using the fastpath or the
fallback path.  See `Stats Collection Tool`_ documentation for more information
on using ``ais-stats``.

Performance baseline
====================

``fio``'s ``psync`` engine can be used to establish a peformance baseline for
hipFile. hipFile should be able to achieve similar performance to ``fio``'s
``psync`` engine when running on the same storage device, with similar IO sizes.

For example, to get an idea of the performance expected from hipFile on the filesystem mounted at
``/mnt/storage``, the following command can be used:

.. code-block:: none

    $ fio \
          --name=test \
          --directory /mnt/storage \
          --ioengine=psync \
          --rw=randread \
          --direct=1 \
          --size=128M \
          --bs=1M \
          --time_based \
          --ramp_time=5 \
          --runtime=10 \
          --numjobs=1 \
          --group_reporting

PCIe Topology
=============

Use the ``lstopo`` command to inspect the link speeds and NUMA topology of the
system's PCIe devices. On some systems, bandwidth between the GPU and storage
device may be limited by the PCIe topology.

The following command will display the PCIe topology of the system:

.. code-block:: none

  lstopo --filter core:none --filter group:none --no-caches --no-smt -.ascii

System Log
==========

Inspect the system log for any hipFile/AIS related errors.

For example the following log message from ```amdgpu`` indicates that it was
unable to map a hipFile IO to a supported IO device.

.. code-block:: none

  Jun 01 13:11:39 Sharky kernel: amdgpu: Invalid file path or mount point
  Jun 01 13:11:39 Sharky kernel: amdgpu: Failed to read AIS file: -19

These log messages indicate that IO is being performed to a file on an
unsupported file system or to an unsupported storage device.

Disable the Fallback Path
=========================

When hipFile is unable to use its fastpath, it will issue the IO to the fallback
path. When investigating performance issues, it may be helpful to disable the
fallback path. With the fallback path disabled hipfile will return an error if
an IO requests is unable to be completed using the fastpath.

Disable the fallback path by adding ``HIPFILE_ALLOW_COMPAT_MODE=false`` to the
environment.

For example:

.. code-block:: none

  $ HIPFILE_ALLOW_COMPAT_MODE=false <command>
