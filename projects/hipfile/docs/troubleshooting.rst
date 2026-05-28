.. meta::
  :description: Troubleshooting hipFile
  :keywords: hipFile, troubleshooting

***************
Troubleshooting
***************

This page provides guidance on troubleshooting common hipFile issues.

Poor NVMe <-> GPU throughput within QEMU virtual machines
=========================================================

If NVMe <-> GPU throughput is lower than expected, PCIe devices may not be
correctly passed into the virtual machine. Ensure that each PCIe device (GPU,
NVMe) passed into a QEMU virtual machine is attached to its own root port. PCIe
devices attached directly to a QEMU virtual machine's root bus will be unable to
fully utilize the PCIe bandwidth of the hardware.

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

See QEMU's documentation for more information on PCIe passthrough.
