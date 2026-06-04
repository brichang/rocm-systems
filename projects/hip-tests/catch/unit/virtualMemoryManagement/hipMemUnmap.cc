/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipMemUnmap hipMemUnmap
 * @{
 * @ingroup VirtualMemoryManagementTest
 * `hipError_t hipMemUnmap (void* ptr, size_t size)` -
 * Unmap memory allocation of a given address range.
 */


#include <hip_test_common.hh>

#include <chrono>

#include "hip_vmm_common.hh"

constexpr int N = (1 << 13);

/**
 * Test Description
 * ------------------------
 *    - Negative Tests
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemUnmap.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
HIP_TEST_CASE(Unit_hipMemUnmap_negative) {
  size_t granularity = 0;
  size_t buffer_size = N * sizeof(int);
  int deviceId = 0;
  hipDevice_t device;

  CTX_CREATE();
  HIP_CHECK(hipDeviceGet(&device, deviceId));
  checkVMMSupported(device);

  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;  // Current Devices

  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);
  size_t size_mem = ((granularity + buffer_size - 1) / granularity) * granularity;

  hipMemGenericAllocationHandle_t handle;
  void* ptrA;
  // Allocate physical memory
  HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));
  // Allocate virtual address range
  HIP_CHECK(hipMemAddressReserve(&ptrA, size_mem, 0, 0, 0));
  HIP_CHECK(hipMemMap(ptrA, size_mem, 0, handle, 0));

  SECTION("nullptr to ptrA") { REQUIRE(hipMemUnmap(nullptr, size_mem) == hipErrorInvalidValue); }

  SECTION("pass zero to size") { REQUIRE(hipMemUnmap(ptrA, 0) == hipErrorInvalidValue); }

  SECTION("unmap a smaller size") {
    REQUIRE(hipMemUnmap(ptrA, (size_mem - 1)) == hipErrorInvalidValue);
  }

  HIP_CHECK(hipMemUnmap(ptrA, size_mem));
  HIP_CHECK(hipMemAddressFree(ptrA, size_mem));
  HIP_CHECK(hipMemRelease(handle));
  CTX_DESTROY();
}

HIP_TEST_CASE(Unit_hipMemUnmap_Capture) {
  CTX_CREATE();
  size_t granularity = 0;
  constexpr size_t kBufferSize = N * sizeof(int);
  int device_id = 0;
  hipDevice_t device;

  HIP_CHECK(hipDeviceGet(&device, device_id));
  checkVMMSupported(device);

  hipMemAllocationProp allocation_prop{};
  allocation_prop.type = hipMemAllocationTypePinned;
  allocation_prop.location.type = hipMemLocationTypeDevice;
  allocation_prop.location.id = device;

  HIP_CHECK(hipMemGetAllocationGranularity(&granularity, &allocation_prop,
                                           hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);
  size_t mem_size = ((granularity + kBufferSize - 1) / granularity) * granularity;

  hipMemGenericAllocationHandle_t allocation_handle;
  void* device_ptr = nullptr;
  HIP_CHECK(hipMemCreate(&allocation_handle, mem_size, &allocation_prop, 0));
  HIP_CHECK(hipMemAddressReserve(&device_ptr, mem_size, 0, nullptr, 0));
  HIP_CHECK(hipMemMap(device_ptr, mem_size, 0, allocation_handle, 0));
  HIP_CHECK(hipMemRelease(allocation_handle));

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  GENERATE_CAPTURE();
  BEGIN_CAPTURE(stream);
  HIP_CHECK(hipMemUnmap(device_ptr, mem_size));
  END_CAPTURE(stream);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipMemAddressFree(device_ptr, mem_size));
  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *    - Characterization test for UnmapMemObjBookkeeping. After
 * hipMemUnmap, the sub_obj must be removed from MemObjMap and the
 * vaddr<->phys cross-link torn down. Observable: a subsequent
 * hipMemRetainAllocationHandle on the (still-reserved) VA must
 * return hipErrorInvalidValue. Pins the RemoveMemObj + cross-link
 * clear + sub_obj.release() steps of the unmap bookkeeping helper.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemUnmap.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
HIP_TEST_CASE(Unit_hipMemUnmap_Bookkeeping_CrossLinksTornDown) {
  HIP_CHECK(hipFree(0));
  size_t granularity = 0;
  size_t buffer_size = N * sizeof(int);
  int deviceId = 0;
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, deviceId));
  checkVMMSupported(device);
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);
  size_t size_mem = ((granularity + buffer_size - 1) / granularity) * granularity;

  hipMemGenericAllocationHandle_t handle;
  void* ptr = nullptr;
  HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));
  HIP_CHECK(hipMemAddressReserve(&ptr, size_mem, 0, 0, 0));
  HIP_CHECK(hipMemMap(ptr, size_mem, 0, handle, 0));

  // Sanity: cross-link is wired pre-unmap.
  hipMemGenericAllocationHandle_t retrieved = nullptr;
  HIP_CHECK(hipMemRetainAllocationHandle(&retrieved, ptr));
  REQUIRE(retrieved == handle);
  HIP_CHECK(hipMemRelease(retrieved));

  HIP_CHECK(hipMemUnmap(ptr, size_mem));

  // After unmap: MemObjMap::RemoveMemObj must have run and the
  // cross-link must be torn down. retainAllocationHandle on the
  // still-reserved VA must now fail.
  REQUIRE(hipMemRetainAllocationHandle(&retrieved, ptr) == hipErrorInvalidValue);

  HIP_CHECK(hipMemRelease(handle));
  HIP_CHECK(hipMemAddressFree(ptr, size_mem));
}

/**
 * Test Description
 * ------------------------
 *    - Characterization test for the map/unmap/remap cycle through
 * the bookkeeping helper. Pin that after an unmap the same VA can
 * be re-mapped (potentially with a different handle) and the
 * cross-link gets rewired to the new handle. Demonstrates the
 * unmap helper fully clears the prior MemObjMap entry + cross-links
 * so a fresh map sees a clean slot.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemUnmap.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
HIP_TEST_CASE(Unit_hipMemUnmap_Bookkeeping_RemapRewiresCrossLinks) {
  HIP_CHECK(hipFree(0));
  size_t granularity = 0;
  size_t buffer_size = N * sizeof(int);
  int deviceId = 0;
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, deviceId));
  checkVMMSupported(device);
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);
  size_t size_mem = ((granularity + buffer_size - 1) / granularity) * granularity;

  hipMemGenericAllocationHandle_t handle1, handle2;
  HIP_CHECK(hipMemCreate(&handle1, size_mem, &prop, 0));
  HIP_CHECK(hipMemCreate(&handle2, size_mem, &prop, 0));

  void* ptr = nullptr;
  HIP_CHECK(hipMemAddressReserve(&ptr, size_mem, 0, 0, 0));

  HIP_CHECK(hipMemMap(ptr, size_mem, 0, handle1, 0));
  hipMemGenericAllocationHandle_t retrieved = nullptr;
  HIP_CHECK(hipMemRetainAllocationHandle(&retrieved, ptr));
  REQUIRE(retrieved == handle1);
  HIP_CHECK(hipMemRelease(retrieved));

  HIP_CHECK(hipMemUnmap(ptr, size_mem));

  // Helper cleared the slot -- remap with different handle must succeed
  // and the cross-link must point at handle2 (not the stale handle1).
  HIP_CHECK(hipMemMap(ptr, size_mem, 0, handle2, 0));
  HIP_CHECK(hipMemRetainAllocationHandle(&retrieved, ptr));
  REQUIRE(retrieved == handle2);
  HIP_CHECK(hipMemRelease(retrieved));

  HIP_CHECK(hipMemUnmap(ptr, size_mem));
  HIP_CHECK(hipMemRelease(handle1));
  HIP_CHECK(hipMemRelease(handle2));
  HIP_CHECK(hipMemAddressFree(ptr, size_mem));
}

/**
 * Test Description
 * ------------------------
 *    - Direct-path TDD test for the sub-buffer loop's "abort on first
 * error" contract in the new hipMemUnmap (Commit 5 of refactor). Per
 * design: on Device::virtualUnmap returning false for any sub-buffer,
 * the loop aborts immediately and returns hipErrorInvalidValue, and
 * ga->release() is NOT called for the failing sub-buffer (release-only-
 * on-success). We cannot directly synthesize a mid-range bad VA from a
 * black-box test: hipMemUnmap rejects whole-range invalid input at
 * ValidateSubBufferCoverage (hip_vm.cpp:433) before the sub-buffer
 * loop is entered, so the abort-on-first-error path inside the loop
 * itself isn't reachable via the public API without intentional state
 * corruption. Skipping with a documented reason; will be re-enabled
 * once Commit 5 lands and we can drive a failure through a test seam
 * or via the device-driver returning an error on a partially-torn
 * range.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemUnmap.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
HIP_TEST_CASE(Unit_hipMemUnmap_DirectPath_AbortsOnFirstError) {
  HIP_SKIP_TEST(
      "Cannot reach the sub-buffer loop's abort-on-first-error branch "
      "from a black-box test -- ValidateSubBufferCoverage rejects bad "
      "input before the loop runs. Re-enable after Commit 5's failure-"
      "injection seam exists.");
}

/**
 * Long-running kernel used by the PAL in-flight-work test. Issues
 * enough iterations of a trivial RMW that the GPU is still busy by
 * the time the host issues hipMemUnmap.
 */
static __global__ void pal_unmap_slow_kernel(int* buf, int iters, int marker) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int v = 0;
  for (int k = 0; k < iters; ++k) {
    v += k;
  }
  buf[i] = marker + (v & 0);  // marker, with a data dep so the loop isn't DCE'd
}

/**
 * PAL-direct-path test helper -- duplicated here from hipMemMap.cc so
 * each translation unit can detect PAL without cross-file linkage. See
 * the comment in hipMemMap.cc on palDeviceDetected() for the platform
 * inference rationale.
 */
static bool palDeviceDetectedUnmap() {
#if defined(_WIN32) && defined(HT_AMD)
  return true;
#else
  return false;
#endif
}

/**
 * Test Description
 * ------------------------
 *    - PAL-direct-path TDD test for the "unmap waits for in-flight
 * work on this device" contract (Commit 3 of refactor, with broader
 * coverage in Task #10 / Commit 6d). Per design, pal::Device::virtualUnmap
 * must take NullStream's execution() lock and call WaitForIdleCompute
 * + WaitForIdleSdma on THIS DEVICE before issuing
 * IQueue::RemapVirtualMemoryPages. Observable from a black-box test:
 * launch a long-running kernel that writes a sentinel into the mapped
 * region on a non-default stream, immediately call hipMemUnmap (no
 * hipStreamSynchronize first), then verify the kernel has been
 * observably completed by the time hipMemUnmap returns -- the mapped
 * VA is torn down on success, so we instead inspect via a separate
 * mapping after the fact.
 *
 *  Simpler observable form used here: after hipMemUnmap returns, the
 * stream the kernel was launched on must report no in-flight work
 * (hipStreamQuery == hipSuccess). If virtualUnmap raced the kernel
 * rather than waiting, the kernel could still be in flight and
 * hipStreamQuery would return hipErrorNotReady. Today this exercises
 * the existing PAL VirtualMapCommand path (which already calls
 * WaitForIdleCompute/Sdma) and is expected to PASS on PAL hardware.
 * After Commit 3, the direct path must preserve this. Skipped when
 * no PAL device is present.
 *
 *  Black-box gaps explicitly NOT tested here:
 *  - That WaitForIdle is called on THIS device only (no peer-device
 *    WaitForIdle). Internal -- not observable via public API.
 *  - That NullStream's VirtualGPU is reused (no new queue created).
 *    Internal -- not observable via public API.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemUnmap.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 *    - PAL backend (Windows AMD-Pro stack)
 */
HIP_TEST_CASE(Unit_hipMemUnmap_PalDirectPath_WaitsForInFlightKernel) {
  if (!palDeviceDetectedUnmap()) {
    HIP_SKIP_TEST(
        "Skipping: PAL device not present in this build "
        "(ROCCLR_ENABLE_PAL=OFF on HSA-only Linux). Test exists for "
        "PAL CI and documents the contract.");
    return;
  }
  HIP_CHECK(hipFree(0));
  size_t granularity = 0;
  size_t buffer_size = N * sizeof(int);
  int deviceId = 0;
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, deviceId));
  checkVMMSupported(device);
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);
  size_t size_mem = ((granularity + buffer_size - 1) / granularity) * granularity;

  hipMemGenericAllocationHandle_t handle;
  void* ptr = nullptr;
  HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));
  HIP_CHECK(hipMemAddressReserve(&ptr, size_mem, 0, 0, 0));
  HIP_CHECK(hipMemMap(ptr, size_mem, 0, handle, 0));

  hipMemAccessDesc accessDesc{};
  accessDesc.location.type = hipMemLocationTypeDevice;
  accessDesc.location.id = device;
  accessDesc.flags = hipMemAccessFlagsProtReadWrite;
  HIP_CHECK(hipMemSetAccess(ptr, size_mem, &accessDesc, 1));

  // Launch a deliberately long kernel on a non-default stream so the
  // default stream's implicit sync doesn't drain it.
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  constexpr int kSlowIters = 1 << 20;
  pal_unmap_slow_kernel<<<dim3(N / threadsPerBlk), dim3(threadsPerBlk), 0, stream>>>(
      reinterpret_cast<int*>(ptr), kSlowIters, 42);

  // Immediately request unmap without an explicit sync. The PAL
  // virtualUnmap path must wait for in-flight compute work touching
  // this device before tearing down the page-table mapping.
  HIP_CHECK(hipMemUnmap(ptr, size_mem));

  // If virtualUnmap correctly waited (WaitForIdleCompute), the stream
  // must report no pending work. If it raced the kernel, the kernel
  // could still be in flight and hipStreamQuery returns
  // hipErrorNotReady.
  hipError_t q = hipStreamQuery(stream);
  REQUIRE(q == hipSuccess);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipMemRelease(handle));
  HIP_CHECK(hipMemAddressFree(ptr, size_mem));
}

/**
 * End doxygen group VirtualMemoryManagementTest.
 * @}
 */
