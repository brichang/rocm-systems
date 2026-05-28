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

High memory utilization with hipFile
====================================

Processes using child processes for IO parallelism may observe each child
process consuming a large amount of memory.

Here is the memory usage of fio using the hipFile engine with four job processes:

.. code-block:: none

    PID  VIRT   RES   SHR Command
   7851 4646M  197M 79364 fio --time_based --runtime=120s --filename=/dev/nvme0n1 --direct=1 --rw=read --bs=8M --name=bw --numjobs=4 --ioengine=libhipfile --rocm_io=hipfile --gpu_dev_ids=0
   7853 4646M  199M 81516 fio --time_based --runtime=120s --filename=/dev/nvme0n1 --direct=1 --rw=read --bs=8M --name=bw --numjobs=4 --ioengine=libhipfile --rocm_io=hipfile --gpu_dev_ids=0
   7852 4646M  197M 79804 fio --time_based --runtime=120s --filename=/dev/nvme0n1 --direct=1 --rw=read --bs=8M --name=bw --numjobs=4 --ioengine=libhipfile --rocm_io=hipfile --gpu_dev_ids=0
   7850 4646M  199M 81504 fio --time_based --runtime=120s --filename=/dev/nvme0n1 --direct=1 --rw=read --bs=8M --name=bw --numjobs=4 --ioengine=libhipfile --rocm_io=hipfile --gpu_dev_ids=0

Here is the memory usage of fio using the psync engine with four job processes:

.. code-block:: none

    PID VIRT   RES   SHR Command
   8021 237M 18004  1640 fio --time_based --runtime=120s --filename=/dev/nvme0n1 --direct=1 --rw=read --bs=8M --name=bw --numjobs=4 --ioengine=psync
   8022 237M 18004  1640 fio --time_based --runtime=120s --filename=/dev/nvme0n1 --direct=1 --rw=read --bs=8M --name=bw --numjobs=4 --ioengine=psync
   8023 237M 18044  1680 fio --time_based --runtime=120s --filename=/dev/nvme0n1 --direct=1 --rw=read --bs=8M --name=bw --numjobs=4 --ioengine=psync
   8024 237M 18040  1676 fio --time_based --runtime=120s --filename=/dev/nvme0n1 --direct=1 --rw=read --bs=8M --name=bw --numjobs=4 --ioengine=psync

Some of the increased memory usage can be attributed to the HIP runtime
dynamically loading the LLVM/Clang compiler stack to compile device kernels
on-demand. When using child processes for IO parallelism, each child process
will load the LLVM/Clang compiler stack separately, leading to increased memory
usage.

Using threads for IO parallelism instead of child processes can help reduce
overall memory usage as the HIP runtime and its dependencies will be shared
among threads.

GPU Reset on RDNA4 GPUs
=======================

hipFile's fallback path may trigger GPU resets on RDNA4 GPUs. The fallback path
is used when IO does not meet the requirements to use the fastpath.

From the systemd journal:

.. code-block:: none

  kernel: amdgpu 0000:f3:00.0: amdgpu: MES might be in unrecoverable state, issue a GPU reset
  kernel: amdgpu 0000:f3:00.0: amdgpu: Suspending all queues failed
  kernel: amdgpu 0000:f3:00.0: amdgpu: Failed to evict process queues
  kernel: amdgpu: Failed to quiesce KFD
  kernel: amdgpu 0000:f3:00.0: amdgpu: GPU reset begin!. Source:  3

To work around this issue, avoid the use of the fallback path on RDNA4 GPUs. Add
``HIPFILE_ALLOW_COMPAT_MODE=0`` to the environment to disable the fallback path.
IOs that would have used the fallback path will fail with
``hipFileInternalError`` instead.

This issue has not been observed on other GPU architectures.
