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

Lower than expected IOPS with small IO sizes
============================================

hipFile delivers lower than expected IOPS with small IO sizes (less than
32KiB) and many threads/processes. This is caused by lock contention when
pinning and unpinning GPU buffers. With small IO sizes, threads pin and unpin GPU
buffers rapidly. A lock must be held while a buffer is pinned/unpinned. With
small IO sizes, GPU buffers are pinned and unpinned rapidly. With many threads,
this leads to high lock contention. As IO sizes increase beyond 32KiB, more time
is spent in the IO stack which increases the time between pin and unpin operations.
Contention on the lock is reduced and performance is inline with expectations.

Poor performance within QEMU virtual machines
=============================================

Note: This is not a hipFile issue, but a common mistake with PCIe passthrough
which affects hipFile performance.

Within a QEMU virtual machine, if NVMe <-> GPU throughput is lower than
expected, PCIe devices may not be correctly passed into the virtual machine.
Ensure that each PCIe device (GPU, NVMe, ...) passed into the virtual machine is
attached to its own root port. PCIe devices attached directly to the root bus
will be unable to fully utilize the hardware's PCIe bandwidth.

The following is the output of ``lspci -tv`` and a snippet from `TransferBench
<https://github.com/ROCm/TransferBench>`_'s output on a virtual machine where
the GPU is attached directly to the root bus.
GPU <-> CPU bandwidth is significantly lower than expected.

.. code-block:: none

  $ lspci -tv
  -[0000:00]-+-00.0  Intel Corporation 82G33/G31/P35/P31 Express DRAM Controller
             +-01.0  Device 1234:1111
             +-02.0  Red Hat, Inc. Virtio block device
             +-03.0  Red Hat, Inc. Virtio network device
             +-04.0  KIOXIA Corporation NVMe SSD Controller XG8
             +-11.0  Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 [Radeon RX 7900 XT/7900 XTX/7900M]
             +-11.1  Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 HDMI/DP Audio
             +-11.2  Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 USB
             +-11.3  Advanced Micro Devices, Inc. [AMD/ATI] Device 7444
             +-1f.0  Intel Corporation 82801IB (ICH9) LPC Interface Controller
             +-1f.2  Intel Corporation 82801IR/IO/IH (ICH9R/DO/DH) 6 port SATA Controller [AHCI mode]
             \-1f.3  Intel Corporation 82801I (ICH9 Family) SMBus Controller

  $ TransferBench p2p
  ... snip ...
                             CPU->CPU  CPU->GPU  GPU->CPU  GPU->GPU
  Averages (During UniDir):       N/A      5.48      7.17       N/A

When the PCIe devices are attached to their own root ports, PCIe bandwidth
between devices is in line with bare metal performance. Below is the output of
``lspci -tv`` and a snippet of TransferBench's output on a virtual machine where
a GPU and NVMe drive are attached to their own root ports.

.. code-block:: none

  $ lspci -tv
  -[0000:00]-+-00.0  Intel Corporation 82G33/G31/P35/P31 Express DRAM Controller
             +-01.0  Device 1234:1111
             +-02.0  Red Hat, Inc. Virtio block device
             +-03.0  Red Hat, Inc. Virtio network device
             +-04.0-[01]----00.0  KIOXIA Corporation NVMe SSD Controller XG8
             +-05.0-[02]--+-00.0  Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 [Radeon RX 7900 XT/7900 XTX/7900M]
             |            +-00.1  Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 HDMI/DP Audio
             |            +-00.2  Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 USB
             |            \-00.3  Advanced Micro Devices, Inc. [AMD/ATI] Device 7444
             +-1f.0  Intel Corporation 82801IB (ICH9) LPC Interface Controller
             +-1f.2  Intel Corporation 82801IR/IO/IH (ICH9R/DO/DH) 6 port SATA Controller [AHCI mode]
             \-1f.3  Intel Corporation 82801I (ICH9 Family) SMBus Controller

  $ TransferBench p2p
  ... snip ...
                             CPU->CPU  CPU->GPU  GPU->CPU  GPU->GPU
  Averages (During UniDir):       N/A     21.24     28.00       N/A

Below is an example QEMU command where GPU and NVMe devices are attached to
their own root ports:

.. code-block:: none

  ./qemu-system-x86_64 \
      -machine q35,accel=kvm,kernel_irqchip=on \
      -cpu host,topoext=on,migratable=off \
      -smp 32,sockets=1,dies=2,cores=8,threads=2 \
      -m 96G \
      -drive file=disk.qcow2,if=none,id=disk0,format=qcow2,cache=none,aio=io_uring,discard=unmap \
      -device virtio-blk-pci,drive=disk0,id=virtio-disk0 \
      -netdev user,id=net0 \
      -device virtio-net-pci,netdev=net0,id=nic0 \
      -device pcie-root-port,id=pcie.1,bus=pcie.0,chassis=1,slot=1 \
      -device vfio-pci,bus=pcie.1,host=0000:01:00.0 \
      -device pcie-root-port,id=pcie.2,bus=pcie.0,chassis=2,slot=2,multifunction=on \
      -device vfio-pci,bus=pcie.2,host=0000:83:00.0,addr=0.0,multifunction=on \
      -device vfio-pci,bus=pcie.2,host=0000:83:00.1,addr=0.1 \
      -device vfio-pci,bus=pcie.2,host=0000:83:00.2,addr=0.2 \
      -device vfio-pci,bus=pcie.2,host=0000:83:00.3,addr=0.3

From the QEMU command above, the NVMe device is passed through with:

.. code-block:: none

      -device pcie-root-port,id=pcie.1,bus=pcie.0,chassis=1,slot=1 \
      -device vfio-pci,bus=pcie.1,host=0000:01:00.0 \

From the QEMU command above, the GPU device is passed through with:

.. code-block:: none

      -device pcie-root-port,id=pcie.2,bus=pcie.0,chassis=2,slot=2,multifunction=on \
      -device vfio-pci,bus=pcie.2,host=0000:83:00.0,addr=0.0,multifunction=on \
      -device vfio-pci,bus=pcie.2,host=0000:83:00.1,addr=0.1 \
      -device vfio-pci,bus=pcie.2,host=0000:83:00.2,addr=0.2 \
      -device vfio-pci,bus=pcie.2,host=0000:83:00.3,addr=0.3

See QEMU's documentation for more information about PCIe passthrough.
