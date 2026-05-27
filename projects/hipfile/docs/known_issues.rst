.. meta::
  :description: Known issues with hipFile
  :keywords: hipFile, known issues, limitations

************
Known issues
************

This page documents known issues with hipFile.

Fastpath is disabled on SR-IOV Virtual Functions
================================================

If a GPU has been split into one or more Virtual Functions (VFs) using Single
Root I/O Virtualization (SR-IOV), hipFile's fastpath is disabled. In this case,
hipFile will use the slower fallback path. There is no workaround for this
issue.

Use ``amd-smi`` to determine if your GPU device is a virtual function. If the
GPU device is a virtual function, the GPU's name will include "VF".  The
following is the output from ``amd-smi`` where the GPU device is a virtual
function:

.. code-block:: none

   +------------------------------------------------------------------------------+
   | AMD-SMI 26.2.2+1d06e2956f    amdgpu version: 6.16.13  ROCm version: 7.2.1    |
   | VBIOS version: 00143754                                                      |
   | Platform: Linux Guest                                                        |
   |-------------------------------------+----------------------------------------|
   | BDF                        GPU-Name | Mem-Uti   Temp   UEC       Power-Usage |
   | GPU  HIP-ID  OAM-ID  Partition-Mode | GFX-Uti    Fan               Mem-Usage |
   |=====================================+========================================|
   | 0000:05:00.0 AMD Instinct MI300X VF | 0 %      39 °C   0           142/750 W |
   |   0       0       2        SPX/NPS1 | 0 %        N/A           285/196288 MB |
   +-------------------------------------+----------------------------------------+

If the GPU device is not a virtual function, the GPU name will not include "VF".

.. code-block:: none

   +------------------------------------------------------------------------------+
   | AMD-SMI 26.2.0+021c61fc      amdgpu version: 6.16.6   ROCm version: 7.1.1    |
   | VBIOS version: 00114328                                                      |
   | Platform: Linux Baremetal                                                    |
   |-------------------------------------+----------------------------------------|
   | BDF                        GPU-Name | Mem-Uti   Temp   UEC       Power-Usage |
   | GPU  HIP-ID  OAM-ID  Partition-Mode | GFX-Uti    Fan               Mem-Usage |
   |=====================================+========================================|
   | 0000:a6:00.0    AMD Instinct MI300X | 0 %      46 °C   0           129/750 W |
   |   0       0       2        SPX/NPS1 | 0 %        N/A           283/196592 MB |
   +-------------------------------------+----------------------------------------+
