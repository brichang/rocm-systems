// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file shared_infra_test.cpp
/// @brief Phase B unit tests: addr_calc, MMA execution, wavefront context, CU factory.

#include "rocjitsu/isa/arch/amdgpu/cdna1/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna2/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/addr_calc.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/isa.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/operand.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/smem.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/sop1.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/sop2.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/sopk.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/vds.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/vflat.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/vglobal.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/vop1.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/vop3.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/vop3p.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/vscratch.h"
#include "rocjitsu/isa/arch/amdgpu/rdna2/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/isa.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_flat.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h"
#include "rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h"
#include "rocjitsu/isa/arch/amdgpu/shared/ds_transpose.h"
#include "rocjitsu/isa/arch/amdgpu/shared/mma_exec.h"
#include "rocjitsu/isa/arch/amdgpu/shared/transcendental.h"
#include "rocjitsu/isa/isa_traits.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l1_vector_cache.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/memory_pipeline.h"

#include "simdojo/sim/simulation.h"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace {

using namespace rocjitsu;

void clear_vgprs(amdgpu::ComputeUnitCore &cu, uint32_t base, uint32_t regs) {
  for (uint32_t reg = 0; reg < regs; ++reg) {
    for (uint32_t lane = 0; lane < amdgpu::WMMA_WAVE32; ++lane)
      cu.write_vgpr(base + reg, lane, 0);
  }
}

void write_packed_f16(amdgpu::ComputeUnitCore &cu, uint32_t base, const amdgpu::InputLoc &loc,
                      float value) {
  const uint32_t reg = base + loc.vgpr_offset;
  const uint32_t shift = loc.sub_element * 16;
  const uint32_t old = cu.read_vgpr(reg, loc.lane);
  const uint32_t packed = static_cast<uint32_t>(util::f32_to_f16(value)) << shift;
  cu.write_vgpr(reg, loc.lane, (old & ~(0xFFFFu << shift)) | packed);
}

void write_packed_bits(amdgpu::ComputeUnitCore &cu, uint32_t base, const amdgpu::InputLoc &loc,
                       uint32_t value) {
  const uint32_t reg = base + loc.vgpr_offset;
  const uint32_t mask = loc.data_bits == 32 ? 0xFFFFFFFFu : ((1u << loc.data_bits) - 1u);
  const uint32_t shifted_mask = mask << loc.bit_offset;
  const uint32_t old = cu.read_vgpr(reg, loc.lane);
  cu.write_vgpr(reg, loc.lane, (old & ~shifted_mask) | ((value & mask) << loc.bit_offset));
}

std::unique_ptr<amdgpu::ComputeUnitCore> make_gfx1250_cu(amdgpu::GpuMemory &mem,
                                                         amdgpu::L2Cache &l2, const char *name,
                                                         uint32_t vgprs_per_wf = 128) {
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_GFX1250;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 128;
  cfg.vgprs_per_wf = vgprs_per_wf;
  cfg.lds_size_kb = 64;
  return amdgpu::ComputeUnitCore::create(name, cfg, &mem, &l2);
}

template <typename InstT>
void run_gfx1250_smem_load(amdgpu::ComputeUnitCore &cu, amdgpu::Wavefront &wf, uint32_t dst,
                           uint32_t offset) {
  gfx1250::SmemMachineInst raw{};
  raw.sbase = 2;
  raw.sdata = dst;
  raw.soffset = gfx1250::OPR_SMEM_OFFSET_NULL;
  raw.ioffset = offset;
  auto inst = std::make_unique<InstT>(reinterpret_cast<const gfx1250::MachineInst *>(&raw));
  inst->execute_impl(wf);
  amdgpu::ScalarMemPipeline pipeline(&cu.l1_scalar());
  pipeline.issue(inst.release(), wf);
}

void write_global_u32(amdgpu::GpuMemory &mem, uint64_t addr, uint32_t value) {
  for (uint32_t byte = 0; byte < sizeof(value); ++byte)
    mem.write8(addr + byte, static_cast<uint8_t>(value >> (byte * 8)));
}

uint32_t read_global_u32(amdgpu::GpuMemory &mem, uint64_t addr) {
  uint32_t value = 0;
  for (uint32_t byte = 0; byte < sizeof(value); ++byte)
    value |= static_cast<uint32_t>(mem.read8(addr + byte)) << (byte * 8);
  return value;
}

template <typename InstT, typename RawT>
void expect_gfx1250_monitor_load_request(uint32_t num_elems) {
  amdgpu::GpuMemory mem("gfx1250_monitor_load_request_mem");
  amdgpu::L2Cache l2("gfx1250_monitor_load_request_l2");
  auto cu = make_gfx1250_cu(mem, l2, "gfx1250_monitor_load_request_cu");
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 128);
  ASSERT_NE(wf, nullptr);

  constexpr uint32_t kVaddr = 4;
  constexpr uint32_t kVdst = 12;
  constexpr uint64_t kAddr = 0x100000;

  const uint32_t vb = wf->vgpr_alloc().base;
  cu->write_vgpr(vb + kVaddr, 0, static_cast<uint32_t>(kAddr));
  cu->write_vgpr(vb + kVaddr + 1, 0, static_cast<uint32_t>(kAddr >> 32));
  wf->set_exec(1ULL);

  RawT raw{};
  raw.saddr = gfx1250::OPR_SREG_NULL;
  raw.vaddr = kVaddr;
  raw.vdst = kVdst;
  auto inst = std::make_unique<InstT>(reinterpret_cast<const gfx1250::MachineInst *>(&raw));
  inst->execute_impl(*wf);

  const auto *d = inst->template data_as<amdgpu::VectorMemState>();
  ASSERT_NE(d, nullptr);
  EXPECT_TRUE(d->is_load);
  EXPECT_EQ(d->elem_size, 4u);
  EXPECT_EQ(d->num_elems, num_elems);
  EXPECT_EQ(d->wait_counter_type, amdgpu::WaitCounterType::LOADCNT);
  EXPECT_EQ(d->dst_reg_base, vb + kVdst);
  EXPECT_EQ(d->per_lane_addr[0], kAddr);
}

template <typename InstT> uint32_t run_gfx1250_sop2(uint32_t src0, uint32_t src1) {
  amdgpu::GpuMemory mem("sop2_mem");
  amdgpu::L2Cache l2("sop2_l2");
  auto cu = make_gfx1250_cu(mem, l2, "sop2_cu");
  auto *wf = cu->dispatch_wf(0, 0, 128, 32);

  constexpr uint32_t kSrc0 = 2;
  constexpr uint32_t kSrc1 = 3;
  constexpr uint32_t kDst = 4;
  const uint32_t sb = wf->sgpr_alloc().base;
  cu->write_sgpr(sb + kSrc0, src0);
  cu->write_sgpr(sb + kSrc1, src1);

  gfx1250::Sop2MachineInst raw{};
  raw.ssrc0 = kSrc0;
  raw.ssrc1 = kSrc1;
  raw.sdst = kDst;
  InstT inst(reinterpret_cast<const gfx1250::MachineInst *>(&raw));
  inst.execute_impl(*wf);
  return cu->read_sgpr(sb + kDst);
}

template <typename InstT>
std::array<uint32_t, 3> run_gfx1250_vop3_add_minmax(const std::array<uint32_t, 2> &src0,
                                                    const std::array<uint32_t, 2> &src1,
                                                    const std::array<uint32_t, 2> &src2) {
  amdgpu::GpuMemory mem("gfx1250_vop3_add_minmax_mem");
  amdgpu::L2Cache l2("gfx1250_vop3_add_minmax_l2");
  auto cu = make_gfx1250_cu(mem, l2, "gfx1250_vop3_add_minmax_cu");
  auto *wf = cu->dispatch_wf(0, 0, 128, 128);

  constexpr uint32_t kSrc0 = 1;
  constexpr uint32_t kSrc1 = 2;
  constexpr uint32_t kSrc2 = 3;
  constexpr uint32_t kDst = 4;
  constexpr uint32_t kVgprSrcBase = 256;
  const uint32_t vb = wf->vgpr_alloc().base;

  for (uint32_t lane = 0; lane < 2; ++lane) {
    cu->write_vgpr(vb + kSrc0, lane, src0[lane]);
    cu->write_vgpr(vb + kSrc1, lane, src1[lane]);
    cu->write_vgpr(vb + kSrc2, lane, src2[lane]);
    cu->write_vgpr(vb + kDst, lane, 0xdeadbeefu);
  }
  cu->write_vgpr(vb + kDst, 2, 0xfeedfaceu);
  wf->set_exec(0x3);

  gfx1250::Vop3MachineInst raw{};
  raw.vdst = kDst;
  raw.src0 = kVgprSrcBase + kSrc0;
  raw.src1 = kVgprSrcBase + kSrc1;
  raw.src2 = kVgprSrcBase + kSrc2;
  InstT inst(reinterpret_cast<const gfx1250::MachineInst *>(&raw));
  inst.execute_impl(*wf);

  return {cu->read_vgpr(vb + kDst, 0), cu->read_vgpr(vb + kDst, 1), cu->read_vgpr(vb + kDst, 2)};
}

template <typename InstT>
std::array<uint32_t, 4> run_gfx1250_vop3_ashr_pk(const std::array<uint32_t, 3> &src0,
                                                 const std::array<uint32_t, 3> &src1,
                                                 const std::array<uint32_t, 3> &src2) {
  amdgpu::GpuMemory mem("gfx1250_vop3_ashr_pk_mem");
  amdgpu::L2Cache l2("gfx1250_vop3_ashr_pk_l2");
  auto cu = make_gfx1250_cu(mem, l2, "gfx1250_vop3_ashr_pk_cu");
  auto *wf = cu->dispatch_wf(0, 0, 128, 128);

  constexpr uint32_t kSrc0 = 1;
  constexpr uint32_t kSrc1 = 2;
  constexpr uint32_t kSrc2 = 3;
  constexpr uint32_t kDst = 4;
  constexpr uint32_t kVgprSrcBase = 256;
  const uint32_t vb = wf->vgpr_alloc().base;

  for (uint32_t lane = 0; lane < 3; ++lane) {
    cu->write_vgpr(vb + kSrc0, lane, src0[lane]);
    cu->write_vgpr(vb + kSrc1, lane, src1[lane]);
    cu->write_vgpr(vb + kSrc2, lane, src2[lane]);
    cu->write_vgpr(vb + kDst, lane, 0xdeadbeefu);
  }
  cu->write_vgpr(vb + kDst, 3, 0xfeedfaceu);
  wf->set_exec(0x7);

  gfx1250::Vop3MachineInst raw{};
  raw.vdst = kDst;
  raw.src0 = kVgprSrcBase + kSrc0;
  raw.src1 = kVgprSrcBase + kSrc1;
  raw.src2 = kVgprSrcBase + kSrc2;
  InstT inst(reinterpret_cast<const gfx1250::MachineInst *>(&raw));
  inst.execute_impl(*wf);

  return {cu->read_vgpr(vb + kDst, 0), cu->read_vgpr(vb + kDst, 1), cu->read_vgpr(vb + kDst, 2),
          cu->read_vgpr(vb + kDst, 3)};
}

template <typename InstT, typename RawT>
std::array<uint32_t, 5> run_gfx1250_cvt_norm_f16(const std::array<uint16_t, 4> &src,
                                                 uint32_t encoded_dst) {
  amdgpu::GpuMemory mem("gfx1250_cvt_norm_mem");
  amdgpu::L2Cache l2("gfx1250_cvt_norm_l2");
  auto cu = make_gfx1250_cu(mem, l2, "gfx1250_cvt_norm_cu");
  auto *wf = cu->dispatch_wf(0, 0, 128, 128);

  constexpr uint32_t kSrc = 1;
  constexpr uint32_t kVgprSrcBase = 256;
  const uint32_t dst_reg = encoded_dst;
  const uint32_t vb = wf->vgpr_alloc().base;

  for (uint32_t lane = 0; lane < src.size(); ++lane) {
    cu->write_vgpr(vb + kSrc, lane, src[lane]);
    cu->write_vgpr(vb + dst_reg, lane, 0xaaaa5555u);
  }
  cu->write_vgpr(vb + dst_reg, 4, 0xfeedfaceu);
  wf->set_exec(0xf);

  RawT raw{};
  raw.vdst = encoded_dst;
  raw.src0 = kVgprSrcBase + kSrc;
  InstT inst(reinterpret_cast<const gfx1250::MachineInst *>(&raw));
  inst.execute_impl(*wf);

  return {cu->read_vgpr(vb + dst_reg, 0), cu->read_vgpr(vb + dst_reg, 1),
          cu->read_vgpr(vb + dst_reg, 2), cu->read_vgpr(vb + dst_reg, 3),
          cu->read_vgpr(vb + dst_reg, 4)};
}

template <typename InstT, typename RawT>
std::array<uint32_t, 5> run_gfx1250_cvt_f16_fp8_bf8(const std::array<uint8_t, 4> &src,
                                                    uint32_t encoded_dst) {
  amdgpu::GpuMemory mem("gfx1250_cvt_f16_fp8_bf8_mem");
  amdgpu::L2Cache l2("gfx1250_cvt_f16_fp8_bf8_l2");
  auto cu = make_gfx1250_cu(mem, l2, "gfx1250_cvt_f16_fp8_bf8_cu");
  auto *wf = cu->dispatch_wf(0, 0, 128, 128);

  constexpr uint32_t kSrc = 1;
  constexpr uint32_t kVgprSrcBase = 256;
  const uint32_t vb = wf->vgpr_alloc().base;

  for (uint32_t lane = 0; lane < src.size(); ++lane) {
    cu->write_vgpr(vb + kSrc, lane, src[lane]);
    cu->write_vgpr(vb + encoded_dst, lane, 0xaaaa5555u);
  }
  cu->write_vgpr(vb + encoded_dst, 4, 0xfeedfaceu);
  wf->set_exec(0xf);

  RawT raw{};
  raw.vdst = encoded_dst;
  raw.src0 = kVgprSrcBase + kSrc;
  InstT inst(reinterpret_cast<const gfx1250::MachineInst *>(&raw));
  inst.execute_impl(*wf);

  return {cu->read_vgpr(vb + encoded_dst, 0), cu->read_vgpr(vb + encoded_dst, 1),
          cu->read_vgpr(vb + encoded_dst, 2), cu->read_vgpr(vb + encoded_dst, 3),
          cu->read_vgpr(vb + encoded_dst, 4)};
}

template <typename InstT, typename RawT>
std::array<uint64_t, 5> run_gfx1250_cvt_pk_f32_fp8_bf8(const std::array<uint16_t, 4> &src,
                                                       uint32_t encoded_dst) {
  amdgpu::GpuMemory mem("gfx1250_cvt_pk_f32_fp8_bf8_mem");
  amdgpu::L2Cache l2("gfx1250_cvt_pk_f32_fp8_bf8_l2");
  auto cu = make_gfx1250_cu(mem, l2, "gfx1250_cvt_pk_f32_fp8_bf8_cu");
  auto *wf = cu->dispatch_wf(0, 0, 128, 128);

  constexpr uint32_t kSrc = 1;
  constexpr uint32_t kVgprSrcBase = 256;
  const uint32_t vb = wf->vgpr_alloc().base;

  for (uint32_t lane = 0; lane < src.size(); ++lane) {
    cu->write_vgpr(vb + kSrc, lane, src[lane]);
    cu->write_vgpr(vb + encoded_dst, lane, 0xaaaa5555u);
    cu->write_vgpr(vb + encoded_dst + 1, lane, 0xbbbb6666u);
  }
  cu->write_vgpr(vb + encoded_dst, 4, 0xfeedfaceu);
  cu->write_vgpr(vb + encoded_dst + 1, 4, 0xcafebabeu);
  wf->set_exec(0xf);

  RawT raw{};
  raw.vdst = encoded_dst;
  raw.src0 = kVgprSrcBase + kSrc;
  InstT inst(reinterpret_cast<const gfx1250::MachineInst *>(&raw));
  inst.execute_impl(*wf);

  auto lane_value = [&](uint32_t lane) {
    return static_cast<uint64_t>(cu->read_vgpr(vb + encoded_dst + 1, lane)) << 32 |
           cu->read_vgpr(vb + encoded_dst, lane);
  };
  return {lane_value(0), lane_value(1), lane_value(2), lane_value(3), lane_value(4)};
}

template <typename InstT, typename RawT>
std::array<uint32_t, 5> run_gfx1250_cvt_pk_f16_fp8_bf8(const std::array<uint16_t, 4> &src,
                                                       uint32_t encoded_dst) {
  amdgpu::GpuMemory mem("gfx1250_cvt_pk_f16_fp8_bf8_mem");
  amdgpu::L2Cache l2("gfx1250_cvt_pk_f16_fp8_bf8_l2");
  auto cu = make_gfx1250_cu(mem, l2, "gfx1250_cvt_pk_f16_fp8_bf8_cu");
  auto *wf = cu->dispatch_wf(0, 0, 128, 128);

  constexpr uint32_t kSrc = 1;
  constexpr uint32_t kVgprSrcBase = 256;
  const uint32_t vb = wf->vgpr_alloc().base;

  for (uint32_t lane = 0; lane < src.size(); ++lane) {
    cu->write_vgpr(vb + kSrc, lane, src[lane]);
    cu->write_vgpr(vb + encoded_dst, lane, 0xaaaa5555u);
  }
  cu->write_vgpr(vb + encoded_dst, 4, 0xfeedfaceu);
  wf->set_exec(0xf);

  RawT raw{};
  raw.vdst = encoded_dst;
  raw.src0 = kVgprSrcBase + kSrc;
  InstT inst(reinterpret_cast<const gfx1250::MachineInst *>(&raw));
  inst.execute_impl(*wf);

  return {cu->read_vgpr(vb + encoded_dst, 0), cu->read_vgpr(vb + encoded_dst, 1),
          cu->read_vgpr(vb + encoded_dst, 2), cu->read_vgpr(vb + encoded_dst, 3),
          cu->read_vgpr(vb + encoded_dst, 4)};
}

template <size_t Count, uint32_t Bits>
std::array<uint32_t, (Count * Bits + 31) / 32>
pack_lowp_codes(const std::array<uint32_t, Count> &codes) {
  std::array<uint32_t, (Count * Bits + 31) / 32> words{};
  const uint32_t mask = (1u << Bits) - 1u;
  for (uint32_t index = 0; index < Count; ++index) {
    const uint32_t bit = index * Bits;
    const uint32_t word = bit / 32;
    const uint32_t shift = bit & 31u;
    const uint32_t code = codes[index] & mask;
    words[word] |= code << shift;
    if (shift + Bits > 32u)
      words[word + 1] |= code >> (32u - shift);
  }
  return words;
}

template <typename InstT, size_t SrcWords, size_t DstWords>
std::array<uint32_t, DstWords>
run_gfx1250_cvt_scale_lane0(const std::array<uint32_t, SrcWords> &src_words, float scale) {
  amdgpu::GpuMemory mem("gfx1250_cvt_scale_mem");
  amdgpu::L2Cache l2("gfx1250_cvt_scale_l2");
  auto cu = make_gfx1250_cu(mem, l2, "gfx1250_cvt_scale_cu");
  auto *wf = cu->dispatch_wf(0, 0, 128, 128);

  constexpr uint32_t kSrc = 1;
  constexpr uint32_t kScale = 24;
  constexpr uint32_t kDst = 32;
  constexpr uint32_t kVgprSrcBase = 256;
  const uint32_t vb = wf->vgpr_alloc().base;
  clear_vgprs(*cu, vb, 64);

  for (uint32_t word = 0; word < SrcWords; ++word)
    cu->write_vgpr(vb + kSrc + word, 0, src_words[word]);
  cu->write_vgpr(vb + kScale, 0, std::bit_cast<uint32_t>(scale));
  for (uint32_t word = 0; word < DstWords; ++word)
    cu->write_vgpr(vb + kDst + word, 0, 0xdeadbeefu);
  wf->set_exec(1);

  gfx1250::Vop3MachineInst raw{};
  raw.vdst = kDst;
  raw.src0 = kVgprSrcBase + kSrc;
  raw.src1 = kVgprSrcBase + kScale;
  InstT inst(reinterpret_cast<const gfx1250::MachineInst *>(&raw));
  inst.execute_impl(*wf);

  std::array<uint32_t, DstWords> result{};
  for (uint32_t word = 0; word < DstWords; ++word)
    result[word] = cu->read_vgpr(vb + kDst + word, 0);
  return result;
}

template <typename InstT>
std::array<uint32_t, 5>
run_gfx1250_cvt_pk_fp8_bf8_f32(const std::array<std::array<float, 2>, 4> &src,
                               uint32_t encoded_dst) {
  amdgpu::GpuMemory mem("gfx1250_cvt_pk_fp8_bf8_f32_mem");
  amdgpu::L2Cache l2("gfx1250_cvt_pk_fp8_bf8_f32_l2");
  auto cu = make_gfx1250_cu(mem, l2, "gfx1250_cvt_pk_fp8_bf8_f32_cu");
  auto *wf = cu->dispatch_wf(0, 0, 128, 128);

  constexpr uint32_t kSrc0 = 1;
  constexpr uint32_t kSrc1 = 2;
  constexpr uint32_t kVgprSrcBase = 256;
  const uint32_t vb = wf->vgpr_alloc().base;

  for (uint32_t lane = 0; lane < src.size(); ++lane) {
    cu->write_vgpr(vb + kSrc0, lane, std::bit_cast<uint32_t>(src[lane][0]));
    cu->write_vgpr(vb + kSrc1, lane, std::bit_cast<uint32_t>(src[lane][1]));
    cu->write_vgpr(vb + encoded_dst, lane, 0xaaaa5555u);
  }
  cu->write_vgpr(vb + encoded_dst, 4, 0xfeedfaceu);
  wf->set_exec(0xf);

  gfx1250::Vop3MachineInst raw{};
  raw.vdst = encoded_dst;
  raw.src0 = kVgprSrcBase + kSrc0;
  raw.src1 = kVgprSrcBase + kSrc1;
  InstT inst(reinterpret_cast<const gfx1250::MachineInst *>(&raw));
  inst.execute_impl(*wf);

  return {cu->read_vgpr(vb + encoded_dst, 0), cu->read_vgpr(vb + encoded_dst, 1),
          cu->read_vgpr(vb + encoded_dst, 2), cu->read_vgpr(vb + encoded_dst, 3),
          cu->read_vgpr(vb + encoded_dst, 4)};
}

template <typename InstT>
std::array<uint32_t, 5>
run_gfx1250_cvt_pk_fp8_bf8_f16(const std::array<std::array<float, 2>, 4> &src,
                               uint32_t encoded_dst) {
  amdgpu::GpuMemory mem("gfx1250_cvt_pk_fp8_bf8_f16_mem");
  amdgpu::L2Cache l2("gfx1250_cvt_pk_fp8_bf8_f16_l2");
  auto cu = make_gfx1250_cu(mem, l2, "gfx1250_cvt_pk_fp8_bf8_f16_cu");
  auto *wf = cu->dispatch_wf(0, 0, 128, 128);

  constexpr uint32_t kSrc = 1;
  constexpr uint32_t kVgprSrcBase = 256;
  const uint32_t vb = wf->vgpr_alloc().base;

  for (uint32_t lane = 0; lane < src.size(); ++lane) {
    uint32_t lo = util::f32_to_f16(src[lane][0]);
    uint32_t hi = util::f32_to_f16(src[lane][1]);
    cu->write_vgpr(vb + kSrc, lane, lo | (hi << 16));
    cu->write_vgpr(vb + encoded_dst, lane, 0xaaaa5555u);
  }
  cu->write_vgpr(vb + encoded_dst, 4, 0xfeedfaceu);
  wf->set_exec(0xf);

  gfx1250::Vop3MachineInst raw{};
  raw.vdst = encoded_dst;
  raw.src0 = kVgprSrcBase + kSrc;
  InstT inst(reinterpret_cast<const gfx1250::MachineInst *>(&raw));
  inst.execute_impl(*wf);

  return {cu->read_vgpr(vb + encoded_dst, 0), cu->read_vgpr(vb + encoded_dst, 1),
          cu->read_vgpr(vb + encoded_dst, 2), cu->read_vgpr(vb + encoded_dst, 3),
          cu->read_vgpr(vb + encoded_dst, 4)};
}

template <typename InstT, typename RawT>
std::array<uint32_t, 5> run_gfx1250_unary16(const std::array<uint16_t, 4> &src,
                                            uint32_t encoded_dst) {
  amdgpu::GpuMemory mem("gfx1250_unary16_mem");
  amdgpu::L2Cache l2("gfx1250_unary16_l2");
  auto cu = make_gfx1250_cu(mem, l2, "gfx1250_unary16_cu");
  auto *wf = cu->dispatch_wf(0, 0, 128, 128);

  constexpr uint32_t kSrc = 1;
  constexpr uint32_t kVgprSrcBase = 256;
  const uint32_t vb = wf->vgpr_alloc().base;

  for (uint32_t lane = 0; lane < src.size(); ++lane) {
    cu->write_vgpr(vb + kSrc, lane, src[lane]);
    cu->write_vgpr(vb + encoded_dst, lane, 0xaaaa5555u);
  }
  cu->write_vgpr(vb + encoded_dst, 4, 0xfeedfaceu);
  wf->set_exec(0xf);

  RawT raw{};
  raw.vdst = encoded_dst;
  raw.src0 = kVgprSrcBase + kSrc;
  InstT inst(reinterpret_cast<const gfx1250::MachineInst *>(&raw));
  inst.execute_impl(*wf);

  return {cu->read_vgpr(vb + encoded_dst, 0), cu->read_vgpr(vb + encoded_dst, 1),
          cu->read_vgpr(vb + encoded_dst, 2), cu->read_vgpr(vb + encoded_dst, 3),
          cu->read_vgpr(vb + encoded_dst, 4)};
}

template <typename InstT, typename RawT>
std::array<uint32_t, 5> run_gfx1250_unary32(const std::array<uint32_t, 4> &src,
                                            uint32_t encoded_dst) {
  amdgpu::GpuMemory mem("gfx1250_unary32_mem");
  amdgpu::L2Cache l2("gfx1250_unary32_l2");
  auto cu = make_gfx1250_cu(mem, l2, "gfx1250_unary32_cu");
  auto *wf = cu->dispatch_wf(0, 0, 128, 128);

  constexpr uint32_t kSrc = 1;
  constexpr uint32_t kVgprSrcBase = 256;
  const uint32_t vb = wf->vgpr_alloc().base;

  for (uint32_t lane = 0; lane < src.size(); ++lane) {
    cu->write_vgpr(vb + kSrc, lane, src[lane]);
    cu->write_vgpr(vb + encoded_dst, lane, 0xdeadbeefu);
  }
  cu->write_vgpr(vb + encoded_dst, 4, 0xfeedfaceu);
  wf->set_exec(0xf);

  RawT raw{};
  raw.vdst = encoded_dst;
  raw.src0 = kVgprSrcBase + kSrc;
  InstT inst(reinterpret_cast<const gfx1250::MachineInst *>(&raw));
  inst.execute_impl(*wf);

  return {cu->read_vgpr(vb + encoded_dst, 0), cu->read_vgpr(vb + encoded_dst, 1),
          cu->read_vgpr(vb + encoded_dst, 2), cu->read_vgpr(vb + encoded_dst, 3),
          cu->read_vgpr(vb + encoded_dst, 4)};
}

template <typename Vop1T, typename Vop3T, typename ExpectedFn>
void expect_gfx1250_unary16_pair(const std::array<uint16_t, 4> &src, ExpectedFn expected,
                                 uint32_t vop1_dst, uint32_t vop3_dst) {
  const auto vop1 = run_gfx1250_unary16<Vop1T, gfx1250::Vop1MachineInst>(src, vop1_dst);
  const auto vop3 = run_gfx1250_unary16<Vop3T, gfx1250::Vop3MachineInst>(src, vop3_dst);

  for (uint32_t lane = 0; lane < src.size(); ++lane) {
    EXPECT_EQ(vop1[lane], expected(src[lane]));
    EXPECT_EQ(vop3[lane], 0xaaaa0000u | expected(src[lane]));
  }
  EXPECT_EQ(vop1[4], 0xfeedfaceu);
  EXPECT_EQ(vop3[4], 0xfeedfaceu);
}

template <typename Vop1T, typename Vop3T, typename ExpectedFn>
void expect_gfx1250_unary32_pair(const std::array<uint32_t, 4> &src, ExpectedFn expected,
                                 uint32_t vop1_dst, uint32_t vop3_dst) {
  const auto vop1 = run_gfx1250_unary32<Vop1T, gfx1250::Vop1MachineInst>(src, vop1_dst);
  const auto vop3 = run_gfx1250_unary32<Vop3T, gfx1250::Vop3MachineInst>(src, vop3_dst);

  for (uint32_t lane = 0; lane < src.size(); ++lane) {
    EXPECT_EQ(vop1[lane], expected(src[lane]));
    EXPECT_EQ(vop3[lane], expected(src[lane]));
  }
  EXPECT_EQ(vop1[4], 0xfeedfaceu);
  EXPECT_EQ(vop3[4], 0xfeedfaceu);
}

template <typename InstT, typename RawT>
std::array<uint32_t, 5> run_gfx1250_cvt_off_f32_i4(const std::array<uint32_t, 4> &src,
                                                   uint32_t encoded_dst) {
  amdgpu::GpuMemory mem("gfx1250_cvt_off_f32_i4_mem");
  amdgpu::L2Cache l2("gfx1250_cvt_off_f32_i4_l2");
  auto cu = make_gfx1250_cu(mem, l2, "gfx1250_cvt_off_f32_i4_cu");
  auto *wf = cu->dispatch_wf(0, 0, 128, 128);

  constexpr uint32_t kSrc = 1;
  constexpr uint32_t kVgprSrcBase = 256;
  const uint32_t vb = wf->vgpr_alloc().base;

  for (uint32_t lane = 0; lane < src.size(); ++lane) {
    cu->write_vgpr(vb + kSrc, lane, src[lane]);
    cu->write_vgpr(vb + encoded_dst, lane, 0xdeadbeefu);
  }
  cu->write_vgpr(vb + encoded_dst, 4, 0xfeedfaceu);
  wf->set_exec(0xf);

  RawT raw{};
  raw.vdst = encoded_dst;
  raw.src0 = kVgprSrcBase + kSrc;
  InstT inst(reinterpret_cast<const gfx1250::MachineInst *>(&raw));
  inst.execute_impl(*wf);

  return {cu->read_vgpr(vb + encoded_dst, 0), cu->read_vgpr(vb + encoded_dst, 1),
          cu->read_vgpr(vb + encoded_dst, 2), cu->read_vgpr(vb + encoded_dst, 3),
          cu->read_vgpr(vb + encoded_dst, 4)};
}

float run_swmmac_f16_row0_col0(uint32_t index_word, uint32_t index_key) {
  amdgpu::GpuMemory mem("swmmac_mem");
  amdgpu::L2Cache l2("swmmac_l2");

  auto cu = make_gfx1250_cu(mem, l2, "swmmac_cu");
  if (!cu)
    return std::numeric_limits<float>::quiet_NaN();
  auto *wf = cu->dispatch_wf(0, 0, 128, 128);
  if (!wf)
    return std::numeric_limits<float>::quiet_NaN();

  const uint32_t vb = wf->vgpr_alloc().base;
  clear_vgprs(*cu, vb, 128);

  const uint32_t dst = vb;
  const uint32_t a_base = vb + 16;
  const uint32_t b_base = vb + 48;
  const uint32_t index_base = vb + 96;

  write_packed_f16(*cu, a_base, amdgpu::wmma_input_loc(16, 32, 0, 0, 16), 2.0f);
  write_packed_f16(*cu, a_base, amdgpu::wmma_input_loc(16, 32, 0, 1, 16), 3.0f);
  write_packed_f16(*cu, b_base, amdgpu::wmma_input_loc(16, 64, 0, 0, 16), 5.0f);
  write_packed_f16(*cu, b_base, amdgpu::wmma_input_loc(16, 64, 0, 1, 16), 7.0f);
  write_packed_f16(*cu, b_base, amdgpu::wmma_input_loc(16, 64, 0, 2, 16), 11.0f);
  write_packed_f16(*cu, b_base, amdgpu::wmma_input_loc(16, 64, 0, 3, 16), 13.0f);
  cu->write_vgpr(index_base, 0, index_word);

  amdgpu::exec_swmmac_f32(*cu, 16, 16, 64, 16, dst, a_base, b_base, dst, index_base,
                          /*index_bits=*/16, index_key, amdgpu::extract_f16, amdgpu::extract_f16);

  const auto out = amdgpu::wmma_output_loc_32(16, 16, 0, 0);
  return std::bit_cast<float>(cu->read_vgpr(dst + out.reg, out.lane));
}

float run_v_wmma_f16_high_bank_row0_col0() {
  amdgpu::GpuMemory mem("wmma_f16_mem");
  amdgpu::L2Cache l2("wmma_f16_l2");
  auto cu = make_gfx1250_cu(mem, l2, "wmma_f16_cu", /*vgprs_per_wf=*/512);
  if (!cu)
    return std::numeric_limits<float>::quiet_NaN();
  auto *wf = cu->dispatch_wf(0, 0, 128, 512);
  if (!wf)
    return std::numeric_limits<float>::quiet_NaN();

  const uint32_t vb = wf->vgpr_alloc().base;
  clear_vgprs(*cu, vb, 512);

  constexpr uint32_t kBank = 1;
  constexpr uint32_t kDstLow = 8;
  constexpr uint32_t kALow = 32;
  constexpr uint32_t kBLow = 64;
  const uint32_t dst = vb + (kBank << 8) + kDstLow;
  const uint32_t a_base = vb + (kBank << 8) + kALow;
  const uint32_t b_base = vb + (kBank << 8) + kBLow;

  write_packed_f16(*cu, a_base, amdgpu::wmma_input_loc(16, 32, 0, 0, 16), 2.0f);
  write_packed_f16(*cu, a_base, amdgpu::wmma_input_loc(16, 32, 0, 1, 16), 3.0f);
  write_packed_f16(*cu, b_base, amdgpu::wmma_input_loc(16, 32, 0, 0, 16), 5.0f);
  write_packed_f16(*cu, b_base, amdgpu::wmma_input_loc(16, 32, 0, 1, 16), 7.0f);

  wf->set_vgpr_msb_mode((kBank << 0) | (kBank << 2) | (kBank << 6));
  gfx1250::Vop3pMachineInst raw{};
  raw.vdst = kDstLow;
  raw.src0 = 256 + kALow;
  raw.src1 = 256 + kBLow;
  raw.src2 = 128;
  gfx1250::VWmmaF3216x16x32F16Vop3p inst(reinterpret_cast<const gfx1250::MachineInst *>(&raw));
  inst.execute_impl(*wf);

  const auto out = amdgpu::wmma_output_loc_32(16, 16, 0, 0);
  return std::bit_cast<float>(cu->read_vgpr(dst + out.reg, out.lane));
}

using IntExtractFn = int32_t (*)(amdgpu::ComputeUnitCore &, uint32_t, const amdgpu::InputLoc &);

uint32_t run_wmma_i32_iu8_row0_col0(IntExtractFn extract_a, IntExtractFn extract_b, bool clamp) {
  amdgpu::GpuMemory mem("wmma_i32_mem");
  amdgpu::L2Cache l2("wmma_i32_l2");
  auto cu = make_gfx1250_cu(mem, l2, "wmma_i32_cu");
  if (!cu)
    return 0;
  auto *wf = cu->dispatch_wf(0, 0, 128, 128);
  if (!wf)
    return 0;

  const uint32_t vb = wf->vgpr_alloc().base;
  clear_vgprs(*cu, vb, 128);

  const uint32_t dst = vb;
  const uint32_t a_base = vb + 16;
  const uint32_t b_base = vb + 48;
  write_packed_bits(*cu, a_base, amdgpu::wmma_input_loc(16, 64, 0, 0, 8), 0xFF);
  write_packed_bits(*cu, b_base, amdgpu::wmma_input_loc(16, 64, 0, 0, 8), 2);

  amdgpu::exec_wmma_i32(*cu, 16, 16, 64, 8, dst, a_base, b_base, dst, extract_a, extract_b, clamp,
                        /*const_acc=*/0);

  const auto out = amdgpu::wmma_output_loc_32(16, 16, 0, 0);
  return cu->read_vgpr(dst + out.reg, out.lane);
}

uint32_t run_swmmac_i32_iu8_row0_col0(IntExtractFn extract_a, IntExtractFn extract_b) {
  amdgpu::GpuMemory mem("swmmac_i32_mem");
  amdgpu::L2Cache l2("swmmac_i32_l2");
  auto cu = make_gfx1250_cu(mem, l2, "swmmac_i32_cu");
  if (!cu)
    return 0;
  auto *wf = cu->dispatch_wf(0, 0, 128, 128);
  if (!wf)
    return 0;

  const uint32_t vb = wf->vgpr_alloc().base;
  clear_vgprs(*cu, vb, 128);

  const uint32_t dst = vb;
  const uint32_t a_base = vb + 16;
  const uint32_t b_base = vb + 48;
  const uint32_t index_base = vb + 96;
  const auto a_loc = amdgpu::wmma_input_loc(16, 64, 0, 0, 8);
  write_packed_bits(*cu, a_base, a_loc, 0xFF);
  write_packed_bits(*cu, b_base, amdgpu::wmma_input_loc(16, 128, 0, 0, 8), 2);
  cu->write_vgpr(index_base, a_loc.lane, 4u);

  amdgpu::exec_swmmac_i32(*cu, 16, 16, 128, 8, dst, a_base, b_base, dst, index_base,
                          /*index_bits=*/32, /*index_key=*/0, extract_a, extract_b,
                          /*clamp=*/false, /*const_acc=*/0);

  const auto out = amdgpu::wmma_output_loc_32(16, 16, 0, 0);
  return cu->read_vgpr(dst + out.reg, out.lane);
}

uint32_t run_dot4_i32_iu8_lane0(uint32_t neg_bits) {
  amdgpu::GpuMemory mem("dot4_iu8_mem");
  amdgpu::L2Cache l2("dot4_iu8_l2");
  auto cu = make_gfx1250_cu(mem, l2, "dot4_iu8_cu");
  if (!cu)
    return 0;
  auto *wf = cu->dispatch_wf(0, 0, 128, 128);
  if (!wf)
    return 0;

  const uint32_t vb = wf->vgpr_alloc().base;
  clear_vgprs(*cu, vb, 8);
  cu->write_vgpr(vb + 1, 0, 0xFF027F80u);
  cu->write_vgpr(vb + 2, 0, 0x0403FE02u);
  cu->write_vgpr(vb + 3, 0, 11u);

  gfx1250::Vop3pMachineInst raw{};
  raw.vdst = 0;
  raw.src0 = 256 + 1;
  raw.src1 = 256 + 2;
  raw.src2 = 256 + 3;
  raw.neg = neg_bits;
  gfx1250::VDot4I32Iu8Vop3p inst(reinterpret_cast<const gfx1250::MachineInst *>(&raw));
  inst.execute_impl(*wf);
  return cu->read_vgpr(vb, 0);
}

uint32_t run_dot8_i32_iu4_lane0(uint32_t neg_bits) {
  amdgpu::GpuMemory mem("dot8_iu4_mem");
  amdgpu::L2Cache l2("dot8_iu4_l2");
  auto cu = make_gfx1250_cu(mem, l2, "dot8_iu4_cu");
  if (!cu)
    return 0;
  auto *wf = cu->dispatch_wf(0, 0, 128, 128);
  if (!wf)
    return 0;

  const uint32_t vb = wf->vgpr_alloc().base;
  clear_vgprs(*cu, vb, 8);
  cu->write_vgpr(vb + 1, 0, 0x3A10F278u);
  cu->write_vgpr(vb + 2, 0, 0x876543E2u);
  cu->write_vgpr(vb + 3, 0, 13u);

  gfx1250::Vop3pMachineInst raw{};
  raw.vdst = 0;
  raw.src0 = 256 + 1;
  raw.src1 = 256 + 2;
  raw.src2 = 256 + 3;
  raw.neg = neg_bits;
  gfx1250::VDot8I32Iu4Vop3p inst(reinterpret_cast<const gfx1250::MachineInst *>(&raw));
  inst.execute_impl(*wf);
  return cu->read_vgpr(vb, 0);
}

uint32_t pack_bf16_pair(float lo, float hi) {
  return static_cast<uint32_t>(util::f32_to_bf16(lo)) |
         (static_cast<uint32_t>(util::f32_to_bf16(hi)) << 16);
}

uint32_t pack_f16_pair(float lo, float hi) {
  return static_cast<uint32_t>(util::f32_to_f16(lo)) |
         (static_cast<uint32_t>(util::f32_to_f16(hi)) << 16);
}

uint32_t pack_u16_pair(uint16_t lo, uint16_t hi) {
  return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}

uint32_t pack_i16_pair(int16_t lo, int16_t hi) {
  return pack_u16_pair(static_cast<uint16_t>(lo), static_cast<uint16_t>(hi));
}

template <typename InstT> uint32_t run_gfx1250_pk_bf16_binop_lane0(uint32_t src0, uint32_t src1) {
  amdgpu::GpuMemory mem("pk_bf16_mem");
  amdgpu::L2Cache l2("pk_bf16_l2");
  auto cu = make_gfx1250_cu(mem, l2, "pk_bf16_cu");
  if (!cu)
    return 0;
  auto *wf = cu->dispatch_wf(0, 0, 128, 128);
  if (!wf)
    return 0;

  const uint32_t vb = wf->vgpr_alloc().base;
  clear_vgprs(*cu, vb, 8);
  cu->write_vgpr(vb + 1, 0, src0);
  cu->write_vgpr(vb + 2, 0, src1);

  gfx1250::Vop3pMachineInst raw{};
  raw.vdst = 0;
  raw.src0 = 256 + 1;
  raw.src1 = 256 + 2;
  raw.opsel_hi = 0x3;
  InstT inst(reinterpret_cast<const gfx1250::MachineInst *>(&raw));
  inst.execute_impl(*wf);
  return cu->read_vgpr(vb, 0);
}

uint32_t run_gfx1250_pk_fma_bf16_lane0(uint32_t src0, uint32_t src1, uint32_t src2) {
  amdgpu::GpuMemory mem("pk_fma_bf16_mem");
  amdgpu::L2Cache l2("pk_fma_bf16_l2");
  auto cu = make_gfx1250_cu(mem, l2, "pk_fma_bf16_cu");
  if (!cu)
    return 0;
  auto *wf = cu->dispatch_wf(0, 0, 128, 128);
  if (!wf)
    return 0;

  const uint32_t vb = wf->vgpr_alloc().base;
  clear_vgprs(*cu, vb, 8);
  cu->write_vgpr(vb + 1, 0, src0);
  cu->write_vgpr(vb + 2, 0, src1);
  cu->write_vgpr(vb + 3, 0, src2);

  gfx1250::Vop3pMachineInst raw{};
  raw.vdst = 0;
  raw.src0 = 256 + 1;
  raw.src1 = 256 + 2;
  raw.src2 = 256 + 3;
  raw.opsel_hi = 0x3;
  raw.pad_14 = 1;
  gfx1250::VPkFmaBf16Vop3p inst(reinterpret_cast<const gfx1250::MachineInst *>(&raw));
  inst.execute_impl(*wf);
  return cu->read_vgpr(vb, 0);
}

template <typename InstT>
uint32_t run_gfx1250_pk_ternary_lane0(uint32_t src0, uint32_t src1, uint32_t src2) {
  amdgpu::GpuMemory mem("pk_ternary_mem");
  amdgpu::L2Cache l2("pk_ternary_l2");
  auto cu = make_gfx1250_cu(mem, l2, "pk_ternary_cu");
  if (!cu)
    return 0;
  auto *wf = cu->dispatch_wf(0, 0, 128, 128);
  if (!wf)
    return 0;

  const uint32_t vb = wf->vgpr_alloc().base;
  clear_vgprs(*cu, vb, 8);
  cu->write_vgpr(vb + 1, 0, src0);
  cu->write_vgpr(vb + 2, 0, src1);
  cu->write_vgpr(vb + 3, 0, src2);

  gfx1250::Vop3pMachineInst raw{};
  raw.vdst = 0;
  raw.src0 = 256 + 1;
  raw.src1 = 256 + 2;
  raw.src2 = 256 + 3;
  raw.opsel_hi = 0x3;
  raw.pad_14 = 1;
  InstT inst(reinterpret_cast<const gfx1250::MachineInst *>(&raw));
  inst.execute_impl(*wf);
  return cu->read_vgpr(vb, 0);
}

// ---------------------------------------------------------------------------
// Concept and trait verification (compile-time)
// ---------------------------------------------------------------------------

static_assert(GpuIsa<cdna3::Isa>);
static_assert(GpuIsa<rdna4::Isa>);
static_assert(GpuIsa<gfx1250::Isa>);
static_assert(HasAccVgpr<cdna3::Isa>);
static_assert(!HasAccVgpr<rdna4::Isa>);
static_assert(!HasAccVgpr<gfx1250::Isa>);
static_assert(HasMonolithicWaitcnt<cdna3::Isa>);
static_assert(!HasMonolithicWaitcnt<rdna4::Isa>);
static_assert(!HasMonolithicWaitcnt<gfx1250::Isa>);

// RDNA3/3.5 retain monolithic S_WAITCNT (GFX11 layout).
static_assert(HasMonolithicWaitcnt<rdna3::Isa>);

// RDNA2 supports Wave64 (WF_SIZE_MAX inherited as 64).
static_assert(rdna2::Isa::WF_SIZE_MAX == 64);
static_assert(gfx1250::Isa::WF_SIZE == 32);
static_assert(gfx1250::Isa::WF_SIZE_MAX == 32);
static_assert(gfx1250::Isa::MAX_VGPRS_PER_WF == 256);
static_assert(gfx1250::Isa::MAX_ADDRESSABLE_VGPRS_PER_WF == 1024);
static_assert(supports_wave_size<gfx1250::Isa>(32));
static_assert(!supports_wave_size<gfx1250::Isa>(64));

TEST(Gfx1250VectorArithmeticTest, AddMinMaxWrapsBeforeSignedOrUnsignedClamp) {
  const auto add_max_i32 = run_gfx1250_vop3_add_minmax<gfx1250::VAddMaxI32Vop3>(
      {0x7fffffffu, 0xfffffffbu}, {1u, 2u}, {0u, 0xfffffff6u});
  EXPECT_EQ(add_max_i32[0], 0u);
  EXPECT_EQ(add_max_i32[1], 0xfffffffdu);
  EXPECT_EQ(add_max_i32[2], 0xfeedfaceu);

  const auto add_min_i32 = run_gfx1250_vop3_add_minmax<gfx1250::VAddMinI32Vop3>(
      {0x7fffffffu, 0xfffffffbu}, {1u, 2u}, {0u, 0xfffffff6u});
  EXPECT_EQ(add_min_i32[0], 0x80000000u);
  EXPECT_EQ(add_min_i32[1], 0xfffffff6u);
  EXPECT_EQ(add_min_i32[2], 0xfeedfaceu);

  const auto add_max_u32 = run_gfx1250_vop3_add_minmax<gfx1250::VAddMaxU32Vop3>(
      {0xffffffffu, 10u}, {2u, 20u}, {5u, 25u});
  EXPECT_EQ(add_max_u32[0], 5u);
  EXPECT_EQ(add_max_u32[1], 30u);
  EXPECT_EQ(add_max_u32[2], 0xfeedfaceu);

  const auto add_min_u32 = run_gfx1250_vop3_add_minmax<gfx1250::VAddMinU32Vop3>(
      {0xffffffffu, 10u}, {2u, 20u}, {5u, 25u});
  EXPECT_EQ(add_min_u32[0], 1u);
  EXPECT_EQ(add_min_u32[1], 25u);
  EXPECT_EQ(add_min_u32[2], 0xfeedfaceu);
}

TEST(Gfx1250VectorArithmeticTest, PackedBf16ArithmeticPacksGeneratedResults) {
  EXPECT_EQ(run_gfx1250_pk_bf16_binop_lane0<gfx1250::VPkAddBf16Vop3p>(pack_bf16_pair(1.5f, -2.0f),
                                                                      pack_bf16_pair(2.25f, 0.5f)),
            pack_bf16_pair(3.75f, -1.5f));

  EXPECT_EQ(run_gfx1250_pk_bf16_binop_lane0<gfx1250::VPkMulBf16Vop3p>(pack_bf16_pair(1.5f, -2.0f),
                                                                      pack_bf16_pair(2.0f, 4.0f)),
            pack_bf16_pair(3.0f, -8.0f));

  const float nan = std::numeric_limits<float>::quiet_NaN();
  EXPECT_EQ(run_gfx1250_pk_bf16_binop_lane0<gfx1250::VPkMinNumBf16Vop3p>(
                pack_bf16_pair(nan, -4.0f), pack_bf16_pair(5.0f, 3.0f)),
            pack_bf16_pair(5.0f, -4.0f));
  EXPECT_EQ(run_gfx1250_pk_bf16_binop_lane0<gfx1250::VPkMaxNumBf16Vop3p>(
                pack_bf16_pair(nan, -4.0f), pack_bf16_pair(5.0f, 3.0f)),
            pack_bf16_pair(5.0f, 3.0f));

  EXPECT_EQ(run_gfx1250_pk_fma_bf16_lane0(pack_bf16_pair(2.0f, -2.0f), pack_bf16_pair(3.0f, 4.0f),
                                          pack_bf16_pair(1.0f, 0.5f)),
            pack_bf16_pair(7.0f, -7.5f));
}

TEST(Gfx1250VectorArithmeticTest, PackedMin3Max3SelectsComponentwiseExtrema) {
  const uint32_t signed_src0 = pack_i16_pair(-10, 30000);
  const uint32_t signed_src1 = pack_i16_pair(5, -20000);
  const uint32_t signed_src2 = pack_i16_pair(-7, 1234);
  EXPECT_EQ(
      run_gfx1250_pk_ternary_lane0<gfx1250::VPkMin3I16Vop3p>(signed_src0, signed_src1, signed_src2),
      pack_i16_pair(-10, -20000));
  EXPECT_EQ(
      run_gfx1250_pk_ternary_lane0<gfx1250::VPkMax3I16Vop3p>(signed_src0, signed_src1, signed_src2),
      pack_i16_pair(5, 30000));

  const uint32_t unsigned_src0 = pack_u16_pair(10, 60000);
  const uint32_t unsigned_src1 = pack_u16_pair(5, 20000);
  const uint32_t unsigned_src2 = pack_u16_pair(7, 65535);
  EXPECT_EQ(run_gfx1250_pk_ternary_lane0<gfx1250::VPkMin3U16Vop3p>(unsigned_src0, unsigned_src1,
                                                                   unsigned_src2),
            pack_u16_pair(5, 20000));
  EXPECT_EQ(run_gfx1250_pk_ternary_lane0<gfx1250::VPkMax3U16Vop3p>(unsigned_src0, unsigned_src1,
                                                                   unsigned_src2),
            pack_u16_pair(10, 65535));

  const float nan = std::numeric_limits<float>::quiet_NaN();
  EXPECT_EQ(run_gfx1250_pk_ternary_lane0<gfx1250::VPkMin3NumF16Vop3p>(
                pack_f16_pair(nan, -4.0f), pack_f16_pair(5.0f, 3.0f), pack_f16_pair(-2.0f, -1.0f)),
            pack_f16_pair(-2.0f, -4.0f));
  EXPECT_EQ(run_gfx1250_pk_ternary_lane0<gfx1250::VPkMax3NumF16Vop3p>(
                pack_f16_pair(nan, -4.0f), pack_f16_pair(5.0f, 3.0f), pack_f16_pair(-2.0f, -1.0f)),
            pack_f16_pair(5.0f, 3.0f));
}

TEST(Gfx1250VectorArithmeticTest, AshrPkI8I32PacksShiftedSaturatedBytes) {
  const auto signed_pk = run_gfx1250_vop3_ashr_pk<gfx1250::VAshrPkI8I32Vop3>(
      {0x00000100u, 0xffffffffu, 0x00000004u}, {0xffffff00u, 0x000000ffu, 0x80000000u},
      {1u, 0u, 33u});
  EXPECT_EQ(signed_pk[0], 0x807fu);
  EXPECT_EQ(signed_pk[1], 0x7fffu);
  EXPECT_EQ(signed_pk[2], 0x8002u);
  EXPECT_EQ(signed_pk[3], 0xfeedfaceu);

  const auto unsigned_pk = run_gfx1250_vop3_ashr_pk<gfx1250::VAshrPkU8I32Vop3>(
      {0x00000100u, 0xffffffffu, 0x00000004u}, {0xffffff00u, 0x000000ffu, 0x80000000u},
      {1u, 0u, 33u});
  EXPECT_EQ(unsigned_pk[0], 0x0080u);
  EXPECT_EQ(unsigned_pk[1], 0xff00u);
  EXPECT_EQ(unsigned_pk[2], 0x0002u);
  EXPECT_EQ(unsigned_pk[3], 0xfeedfaceu);
}

TEST(Gfx1250VectorArithmeticTest, CvtNormF16SaturatesAndZeroExtendsGeneratedResults) {
  const uint16_t half = util::f32_to_f16(0.5f);
  const uint16_t negative_overflow = util::f32_to_f16(-1.5f);
  const uint16_t positive_overflow = util::f32_to_f16(2.0f);
  const uint16_t nan = util::f32_to_f16(std::numeric_limits<float>::quiet_NaN());

  const auto i16_vop1 =
      run_gfx1250_cvt_norm_f16<gfx1250::VCvtNormI16F16Vop1, gfx1250::Vop1MachineInst>(
          {half, negative_overflow, nan, positive_overflow}, 4);
  EXPECT_EQ(i16_vop1[0], 0x3fffu);
  EXPECT_EQ(i16_vop1[1], 0x8000u);
  EXPECT_EQ(i16_vop1[2], 0x0000u);
  EXPECT_EQ(i16_vop1[3], 0x7fffu);
  EXPECT_EQ(i16_vop1[4], 0xfeedfaceu);

  const auto u16_vop1 =
      run_gfx1250_cvt_norm_f16<gfx1250::VCvtNormU16F16Vop1, gfx1250::Vop1MachineInst>(
          {half, negative_overflow, nan, positive_overflow}, 5);
  EXPECT_EQ(u16_vop1[0], 0x7fffu);
  EXPECT_EQ(u16_vop1[1], 0x0000u);
  EXPECT_EQ(u16_vop1[2], 0x0000u);
  EXPECT_EQ(u16_vop1[3], 0xffffu);
  EXPECT_EQ(u16_vop1[4], 0xfeedfaceu);

  const auto i16_vop3 =
      run_gfx1250_cvt_norm_f16<gfx1250::VCvtNormI16F16Vop3, gfx1250::Vop3MachineInst>(
          {half, negative_overflow, nan, positive_overflow}, 6);
  EXPECT_EQ(i16_vop3[0], 0x3fffu);
  EXPECT_EQ(i16_vop3[1], 0x8000u);
  EXPECT_EQ(i16_vop3[2], 0x0000u);
  EXPECT_EQ(i16_vop3[3], 0x7fffu);
  EXPECT_EQ(i16_vop3[4], 0xfeedfaceu);

  const auto u16_vop3 =
      run_gfx1250_cvt_norm_f16<gfx1250::VCvtNormU16F16Vop3, gfx1250::Vop3MachineInst>(
          {half, negative_overflow, nan, positive_overflow}, 7);
  EXPECT_EQ(u16_vop3[0], 0x7fffu);
  EXPECT_EQ(u16_vop3[1], 0x0000u);
  EXPECT_EQ(u16_vop3[2], 0x0000u);
  EXPECT_EQ(u16_vop3[3], 0xffffu);
  EXPECT_EQ(u16_vop3[4], 0xfeedfaceu);
}

TEST(Gfx1250VectorArithmeticTest, CvtF16Fp8Bf8ZeroExtendsGeneratedHalfResults) {
  const std::array<uint8_t, 4> fp8_src = {0x00u, 0x38u, 0xb8u, 0x7fu};
  const auto expected_fp8 = [](uint8_t value) -> uint32_t {
    return util::f32_to_f16(util::fp8_e4m3_to_f32(value));
  };

  const auto fp8_vop1 =
      run_gfx1250_cvt_f16_fp8_bf8<gfx1250::VCvtF16Fp8Vop1, gfx1250::Vop1MachineInst>(fp8_src, 8);
  EXPECT_EQ(fp8_vop1[0], expected_fp8(fp8_src[0]));
  EXPECT_EQ(fp8_vop1[1], expected_fp8(fp8_src[1]));
  EXPECT_EQ(fp8_vop1[2], expected_fp8(fp8_src[2]));
  EXPECT_EQ(fp8_vop1[3], expected_fp8(fp8_src[3]));
  EXPECT_EQ(fp8_vop1[4], 0xfeedfaceu);

  const auto fp8_vop3 =
      run_gfx1250_cvt_f16_fp8_bf8<gfx1250::VCvtF16Fp8Vop3, gfx1250::Vop3MachineInst>(fp8_src, 9);
  EXPECT_EQ(fp8_vop3[0], expected_fp8(fp8_src[0]));
  EXPECT_EQ(fp8_vop3[1], expected_fp8(fp8_src[1]));
  EXPECT_EQ(fp8_vop3[2], expected_fp8(fp8_src[2]));
  EXPECT_EQ(fp8_vop3[3], expected_fp8(fp8_src[3]));
  EXPECT_EQ(fp8_vop3[4], 0xfeedfaceu);

  const std::array<uint8_t, 4> bf8_src = {0x00u, 0x3cu, 0xbcu, 0x7cu};
  const auto expected_bf8 = [](uint8_t value) -> uint32_t {
    return util::f32_to_f16(util::bf8_e5m2_to_f32(value));
  };

  const auto bf8_vop1 =
      run_gfx1250_cvt_f16_fp8_bf8<gfx1250::VCvtF16Bf8Vop1, gfx1250::Vop1MachineInst>(bf8_src, 10);
  EXPECT_EQ(bf8_vop1[0], expected_bf8(bf8_src[0]));
  EXPECT_EQ(bf8_vop1[1], expected_bf8(bf8_src[1]));
  EXPECT_EQ(bf8_vop1[2], expected_bf8(bf8_src[2]));
  EXPECT_EQ(bf8_vop1[3], expected_bf8(bf8_src[3]));
  EXPECT_EQ(bf8_vop1[4], 0xfeedfaceu);

  const auto bf8_vop3 =
      run_gfx1250_cvt_f16_fp8_bf8<gfx1250::VCvtF16Bf8Vop3, gfx1250::Vop3MachineInst>(bf8_src, 11);
  EXPECT_EQ(bf8_vop3[0], expected_bf8(bf8_src[0]));
  EXPECT_EQ(bf8_vop3[1], expected_bf8(bf8_src[1]));
  EXPECT_EQ(bf8_vop3[2], expected_bf8(bf8_src[2]));
  EXPECT_EQ(bf8_vop3[3], expected_bf8(bf8_src[3]));
  EXPECT_EQ(bf8_vop3[4], 0xfeedfaceu);
}

TEST(Gfx1250VectorArithmeticTest, CvtPkF32F16Fp8Bf8UnpacksBothPackedElements) {
  auto pack_f32 = [](float lo, float hi) {
    return static_cast<uint64_t>(std::bit_cast<uint32_t>(lo)) |
           (static_cast<uint64_t>(std::bit_cast<uint32_t>(hi)) << 32);
  };
  auto pack_f16 = [](float lo, float hi) {
    return static_cast<uint32_t>(util::f32_to_f16(lo)) |
           (static_cast<uint32_t>(util::f32_to_f16(hi)) << 16);
  };

  const std::array<uint16_t, 4> fp8_src = {0x3800u, 0x7fb8u, 0xc040u, 0xf878u};
  const std::array<uint16_t, 4> bf8_src = {0xbc3cu, 0xfc7cu, 0x3c00u, 0x7fbcu};

  std::array<uint64_t, 5> expected_fp8_f32{};
  std::array<uint64_t, 5> expected_bf8_f32{};
  std::array<uint32_t, 5> expected_fp8_f16{};
  std::array<uint32_t, 5> expected_bf8_f16{};
  for (uint32_t lane = 0; lane < fp8_src.size(); ++lane) {
    const auto fp8_lo = util::fp8_e4m3_to_f32(static_cast<uint8_t>(fp8_src[lane] & 0xffu));
    const auto fp8_hi = util::fp8_e4m3_to_f32(static_cast<uint8_t>(fp8_src[lane] >> 8));
    const auto bf8_lo = util::bf8_e5m2_to_f32(static_cast<uint8_t>(bf8_src[lane] & 0xffu));
    const auto bf8_hi = util::bf8_e5m2_to_f32(static_cast<uint8_t>(bf8_src[lane] >> 8));
    expected_fp8_f32[lane] = pack_f32(fp8_lo, fp8_hi);
    expected_bf8_f32[lane] = pack_f32(bf8_lo, bf8_hi);
    expected_fp8_f16[lane] = pack_f16(fp8_lo, fp8_hi);
    expected_bf8_f16[lane] = pack_f16(bf8_lo, bf8_hi);
  }
  expected_fp8_f32[4] = 0xcafebabefeedfaceull;
  expected_bf8_f32[4] = 0xcafebabefeedfaceull;
  expected_fp8_f16[4] = 0xfeedfaceu;
  expected_bf8_f16[4] = 0xfeedfaceu;

  EXPECT_EQ((run_gfx1250_cvt_pk_f32_fp8_bf8<gfx1250::VCvtPkF32Fp8Vop1, gfx1250::Vop1MachineInst>(
                fp8_src, 12)),
            expected_fp8_f32);
  EXPECT_EQ((run_gfx1250_cvt_pk_f32_fp8_bf8<gfx1250::VCvtPkF32Fp8Vop3, gfx1250::Vop3MachineInst>(
                fp8_src, 14)),
            expected_fp8_f32);
  EXPECT_EQ((run_gfx1250_cvt_pk_f32_fp8_bf8<gfx1250::VCvtPkF32Bf8Vop1, gfx1250::Vop1MachineInst>(
                bf8_src, 16)),
            expected_bf8_f32);
  EXPECT_EQ((run_gfx1250_cvt_pk_f32_fp8_bf8<gfx1250::VCvtPkF32Bf8Vop3, gfx1250::Vop3MachineInst>(
                bf8_src, 18)),
            expected_bf8_f32);

  EXPECT_EQ((run_gfx1250_cvt_pk_f16_fp8_bf8<gfx1250::VCvtPkF16Fp8Vop1, gfx1250::Vop1MachineInst>(
                fp8_src, 20)),
            expected_fp8_f16);
  EXPECT_EQ((run_gfx1250_cvt_pk_f16_fp8_bf8<gfx1250::VCvtPkF16Fp8Vop3, gfx1250::Vop3MachineInst>(
                fp8_src, 21)),
            expected_fp8_f16);
  EXPECT_EQ((run_gfx1250_cvt_pk_f16_fp8_bf8<gfx1250::VCvtPkF16Bf8Vop1, gfx1250::Vop1MachineInst>(
                bf8_src, 22)),
            expected_bf8_f16);
  EXPECT_EQ((run_gfx1250_cvt_pk_f16_fp8_bf8<gfx1250::VCvtPkF16Bf8Vop3, gfx1250::Vop3MachineInst>(
                bf8_src, 23)),
            expected_bf8_f16);
}

TEST(Gfx1250VectorArithmeticTest, CvtPkFp8Bf8OutputUsesRnePacking) {
  EXPECT_EQ(util::f32_to_fp8_e4m3_rne(1.1875f), 0x3au);
  EXPECT_EQ(util::f32_to_bf8_e5m2_rne(1.375f), 0x3eu);
  EXPECT_EQ(util::f32_to_fp8_e4m3_rne(464.0f), 0x7eu);
  EXPECT_EQ(util::fp8_e4m3_to_f32(0x7eu), 448.0f);
  EXPECT_TRUE(std::isnan(util::fp8_e4m3_to_f32(0x7fu)));

  const std::array<std::array<float, 2>, 4> fp8_src = {{{1.0625f, 1.1875f},
                                                        {-1.1875f, -1.0625f},
                                                        {448.0f, 464.0f},
                                                        {0.0009765625f, -0.0009765625f}}};
  const std::array<std::array<float, 2>, 4> bf8_src = {
      {{1.125f, 1.375f}, {-1.375f, -1.125f}, {57344.0f, 61440.0f}, {0.0f, -0.0f}}};

  auto pack_fp8 = [](float lo, float hi) {
    return static_cast<uint32_t>(util::f32_to_fp8_e4m3_rne(lo)) |
           (static_cast<uint32_t>(util::f32_to_fp8_e4m3_rne(hi)) << 8);
  };
  auto pack_bf8 = [](float lo, float hi) {
    return static_cast<uint32_t>(util::f32_to_bf8_e5m2_rne(lo)) |
           (static_cast<uint32_t>(util::f32_to_bf8_e5m2_rne(hi)) << 8);
  };
  auto half_value = [](float value) { return util::f16_to_f32(util::f32_to_f16(value)); };

  std::array<uint32_t, 5> expected_fp8_f32{};
  std::array<uint32_t, 5> expected_bf8_f32{};
  std::array<uint32_t, 5> expected_fp8_f16{};
  std::array<uint32_t, 5> expected_bf8_f16{};
  for (uint32_t lane = 0; lane < fp8_src.size(); ++lane) {
    expected_fp8_f32[lane] = pack_fp8(fp8_src[lane][0], fp8_src[lane][1]);
    expected_bf8_f32[lane] = pack_bf8(bf8_src[lane][0], bf8_src[lane][1]);
    expected_fp8_f16[lane] = pack_fp8(half_value(fp8_src[lane][0]), half_value(fp8_src[lane][1]));
    expected_bf8_f16[lane] = pack_bf8(half_value(bf8_src[lane][0]), half_value(bf8_src[lane][1]));
  }
  expected_fp8_f32[4] = 0xfeedfaceu;
  expected_bf8_f32[4] = 0xfeedfaceu;
  expected_fp8_f16[4] = 0xfeedfaceu;
  expected_bf8_f16[4] = 0xfeedfaceu;

  EXPECT_EQ((run_gfx1250_cvt_pk_fp8_bf8_f32<gfx1250::VCvtPkFp8F32Vop3>(fp8_src, 24)),
            expected_fp8_f32);
  EXPECT_EQ((run_gfx1250_cvt_pk_fp8_bf8_f32<gfx1250::VCvtPkBf8F32Vop3>(bf8_src, 25)),
            expected_bf8_f32);
  EXPECT_EQ((run_gfx1250_cvt_pk_fp8_bf8_f16<gfx1250::VCvtPkFp8F16Vop3>(fp8_src, 26)),
            expected_fp8_f16);
  EXPECT_EQ((run_gfx1250_cvt_pk_fp8_bf8_f16<gfx1250::VCvtPkBf8F16Vop3>(bf8_src, 27)),
            expected_bf8_f16);
}

TEST(Gfx1250VectorArithmeticTest, CvtScalePackedLowPrecisionUsesScaleAndPackedLayout) {
  const std::array<uint32_t, 8> fp4_codes = {0x0u, 0x1u, 0x2u, 0x3u, 0x8u, 0x9u, 0xau, 0xfu};
  const auto fp4_words = pack_lowp_codes<8, 4>(fp4_codes);
  std::array<uint32_t, 8> expected_fp4_f32{};
  for (uint32_t index = 0; index < fp4_codes.size(); ++index) {
    const float value = util::fp4_e2m1_to_f32(static_cast<uint8_t>(fp4_codes[index])) * 2.0f;
    expected_fp4_f32[index] = std::bit_cast<uint32_t>(value);
  }
  EXPECT_EQ((run_gfx1250_cvt_scale_lane0<gfx1250::VCvtScalePk8F32Fp4Vop3, 1, 8>(fp4_words, 2.0f)),
            expected_fp4_f32);

  const std::array<uint32_t, 16> bf6_codes = {0x00u, 0x08u, 0x0cu, 0x10u, 0x14u, 0x18u,
                                              0x1cu, 0x1fu, 0x20u, 0x28u, 0x2cu, 0x30u,
                                              0x34u, 0x38u, 0x3cu, 0x3fu};
  const auto bf6_words = pack_lowp_codes<16, 6>(bf6_codes);
  std::array<uint32_t, 8> expected_bf6_bf16{};
  for (uint32_t word = 0; word < expected_bf6_bf16.size(); ++word) {
    const float lo = util::bf6_e3m2_to_f32(static_cast<uint8_t>(bf6_codes[word * 2])) * 0.5f;
    const float hi = util::bf6_e3m2_to_f32(static_cast<uint8_t>(bf6_codes[word * 2 + 1])) * 0.5f;
    expected_bf6_bf16[word] = pack_bf16_pair(lo, hi);
  }
  EXPECT_EQ((run_gfx1250_cvt_scale_lane0<gfx1250::VCvtScalePk16Bf16Bf6Vop3, 3, 8>(bf6_words, 0.5f)),
            expected_bf6_bf16);

  const std::array<float, 8> f32_values = {0.0f, 0.5f, 1.0f, 1.5f, -0.5f, -1.0f, -2.0f, -3.0f};
  std::array<uint32_t, 8> f32_words{};
  std::array<uint32_t, 8> expected_fp4_codes{};
  for (uint32_t index = 0; index < f32_values.size(); ++index) {
    f32_words[index] = std::bit_cast<uint32_t>(f32_values[index]);
    expected_fp4_codes[index] = util::f32_to_fp4_e2m1_rne(f32_values[index] * 0.5f);
  }
  EXPECT_EQ(
      (run_gfx1250_cvt_scale_lane0<gfx1250::VCvtScalef32Pk8Fp4F32Vop3, 8, 1>(f32_words, 0.5f)),
      (pack_lowp_codes<8, 4>(expected_fp4_codes)));

  const std::array<float, 16> bf16_values = {0.0f, 0.5f, 1.0f, 2.0f,  -0.5f, -1.0f, -2.0f, 3.0f,
                                             4.0f, 6.0f, 8.0f, 12.0f, -4.0f, -6.0f, -8.0f, -12.0f};
  std::array<uint32_t, 8> bf16_src_words{};
  std::array<uint32_t, 16> expected_bf6_codes{};
  for (uint32_t word = 0; word < bf16_src_words.size(); ++word) {
    bf16_src_words[word] = pack_bf16_pair(bf16_values[word * 2], bf16_values[word * 2 + 1]);
    const float lo = util::bf16_to_f32(util::f32_to_bf16(bf16_values[word * 2])) * 0.5f;
    const float hi = util::bf16_to_f32(util::f32_to_bf16(bf16_values[word * 2 + 1])) * 0.5f;
    expected_bf6_codes[word * 2] = util::f32_to_bf6_e3m2_rne(lo);
    expected_bf6_codes[word * 2 + 1] = util::f32_to_bf6_e3m2_rne(hi);
  }
  EXPECT_EQ((run_gfx1250_cvt_scale_lane0<gfx1250::VCvtScalef32Pk16Bf6Bf16Vop3, 8, 3>(bf16_src_words,
                                                                                     0.5f)),
            (pack_lowp_codes<16, 6>(expected_bf6_codes)));
}

TEST(Gfx1250VectorArithmeticTest, CosBf16UsesSharedScaledCosine) {
  const std::array<uint16_t, 4> src = {0x0000u, 0x3f00u, 0x3f80u, 0xbf00u};
  const auto expected = [](uint16_t value) -> uint32_t {
    return util::f32_to_bf16(amdgpu::transcendental::cos_f32(util::bf16_to_f32(value)));
  };

  const auto vop1 = run_gfx1250_unary16<gfx1250::VCosBf16Vop1, gfx1250::Vop1MachineInst>(src, 8);
  EXPECT_EQ(vop1[0], expected(src[0]));
  EXPECT_EQ(vop1[1], expected(src[1]));
  EXPECT_EQ(vop1[2], expected(src[2]));
  EXPECT_EQ(vop1[3], expected(src[3]));
  EXPECT_EQ(vop1[4], 0xfeedfaceu);

  const auto vop3 = run_gfx1250_unary16<gfx1250::VCosBf16Vop3, gfx1250::Vop3MachineInst>(src, 9);
  EXPECT_EQ(vop3[0], 0xaaaa0000u | expected(src[0]));
  EXPECT_EQ(vop3[1], 0xaaaa0000u | expected(src[1]));
  EXPECT_EQ(vop3[2], 0xaaaa0000u | expected(src[2]));
  EXPECT_EQ(vop3[3], 0xaaaa0000u | expected(src[3]));
  EXPECT_EQ(vop3[4], 0xfeedfaceu);
}

TEST(Gfx1250VectorArithmeticTest, Bf16TranscendentalsUseSharedHelpers) {
  const std::array<uint16_t, 4> src = {
      util::f32_to_bf16(0.25f),
      util::f32_to_bf16(0.5f),
      util::f32_to_bf16(1.0f),
      util::f32_to_bf16(2.0f),
  };
  const auto expect_bf16 = [](uint16_t value, auto fn) -> uint32_t {
    return util::f32_to_bf16(fn(util::bf16_to_f32(value)));
  };

  expect_gfx1250_unary16_pair<gfx1250::VRcpBf16Vop1, gfx1250::VRcpBf16Vop3>(
      src, [&](uint16_t value) { return expect_bf16(value, amdgpu::transcendental::rcp_f32); }, 12,
      13);
  expect_gfx1250_unary16_pair<gfx1250::VSqrtBf16Vop1, gfx1250::VSqrtBf16Vop3>(
      src, [&](uint16_t value) { return expect_bf16(value, amdgpu::transcendental::sqrt_f32); }, 14,
      15);
  expect_gfx1250_unary16_pair<gfx1250::VRsqBf16Vop1, gfx1250::VRsqBf16Vop3>(
      src, [&](uint16_t value) { return expect_bf16(value, amdgpu::transcendental::rsq_f32); }, 16,
      17);
  expect_gfx1250_unary16_pair<gfx1250::VLogBf16Vop1, gfx1250::VLogBf16Vop3>(
      src, [&](uint16_t value) { return expect_bf16(value, amdgpu::transcendental::log_f32); }, 18,
      19);
  expect_gfx1250_unary16_pair<gfx1250::VExpBf16Vop1, gfx1250::VExpBf16Vop3>(
      src, [&](uint16_t value) { return expect_bf16(value, amdgpu::transcendental::exp_f32); }, 20,
      21);
  expect_gfx1250_unary16_pair<gfx1250::VSinBf16Vop1, gfx1250::VSinBf16Vop3>(
      src, [&](uint16_t value) { return expect_bf16(value, amdgpu::transcendental::sin_f32); }, 22,
      23);
  expect_gfx1250_unary16_pair<gfx1250::VTanhBf16Vop1, gfx1250::VTanhBf16Vop3>(
      src, [&](uint16_t value) { return expect_bf16(value, amdgpu::transcendental::tanh_f32); }, 24,
      25);
}

TEST(Gfx1250VectorArithmeticTest, TanhF32F16UsesSharedHelper) {
  const std::array<uint32_t, 4> f32_src = {
      std::bit_cast<uint32_t>(0.0f),
      std::bit_cast<uint32_t>(0.5f),
      std::bit_cast<uint32_t>(-1.0f),
      std::bit_cast<uint32_t>(2.0f),
  };
  expect_gfx1250_unary32_pair<gfx1250::VTanhF32Vop1, gfx1250::VTanhF32Vop3>(
      f32_src,
      [](uint32_t value) {
        return std::bit_cast<uint32_t>(
            amdgpu::transcendental::tanh_f32(std::bit_cast<float>(value)));
      },
      26, 27);

  const std::array<uint16_t, 4> f16_src = {
      util::f32_to_f16(0.0f),
      util::f32_to_f16(0.5f),
      util::f32_to_f16(-1.0f),
      util::f32_to_f16(2.0f),
  };
  expect_gfx1250_unary16_pair<gfx1250::VTanhF16Vop1, gfx1250::VTanhF16Vop3>(
      f16_src,
      [](uint16_t value) -> uint32_t {
        return util::f32_to_f16(amdgpu::transcendental::tanh_f32(util::f16_to_f32(value)));
      },
      28, 29);
}

TEST(Gfx1250VectorArithmeticTest, CvtOffF32I4UsesSignedNibbleOffsetTable) {
  const std::array<uint32_t, 4> src = {0u, 7u, 8u, 0xffffffffu};
  const auto expected = [](uint32_t value) -> uint32_t {
    int32_t nibble = static_cast<int32_t>(value & 0xfu);
    if (nibble & 0x8)
      nibble -= 16;
    return std::bit_cast<uint32_t>(static_cast<float>(nibble) * 0.0625f);
  };

  const auto vop1 =
      run_gfx1250_cvt_off_f32_i4<gfx1250::VCvtOffF32I4Vop1, gfx1250::Vop1MachineInst>(src, 10);
  EXPECT_EQ(vop1[0], expected(src[0]));
  EXPECT_EQ(vop1[1], expected(src[1]));
  EXPECT_EQ(vop1[2], expected(src[2]));
  EXPECT_EQ(vop1[3], expected(src[3]));
  EXPECT_EQ(vop1[4], 0xfeedfaceu);

  const auto vop3 =
      run_gfx1250_cvt_off_f32_i4<gfx1250::VCvtOffF32I4Vop3, gfx1250::Vop3MachineInst>(src, 11);
  EXPECT_EQ(vop3[0], expected(src[0]));
  EXPECT_EQ(vop3[1], expected(src[1]));
  EXPECT_EQ(vop3[2], expected(src[2]));
  EXPECT_EQ(vop3[3], expected(src[3]));
  EXPECT_EQ(vop3[4], 0xfeedfaceu);
}

TEST(Gfx1250True16Test, VMovB16E32HighDestinationMergesHalf) {
  amdgpu::GpuMemory mem("gfx1250_mov_b16_mem");
  amdgpu::L2Cache l2("gfx1250_mov_b16_l2");
  auto cu = make_gfx1250_cu(mem, l2, "gfx1250_mov_b16_cu");
  ASSERT_NE(cu, nullptr);
  auto *wf = cu->dispatch_wf(0, 0, 128, 128);
  ASSERT_NE(wf, nullptr);

  const uint32_t vb = wf->vgpr_alloc().base;
  cu->write_vgpr(vb + 0, 0, 0xaaaa5555u);
  cu->write_vgpr(vb + 1, 0, 0x00003c00u);

  gfx1250::Vop1MachineInst raw{};
  raw.src0 = 256 + 1;
  raw.op = 28;
  raw.pad_16 = 0;
  raw.vdst = 128;
  raw.encoding = 0x7c;
  gfx1250::VMovB16Vop1 inst(reinterpret_cast<const gfx1250::MachineInst *>(&raw));

  inst.execute_impl(*wf);

  EXPECT_EQ(cu->read_vgpr(vb + 0, 0), 0x3c005555u);
}

TEST(Gfx1250True16Test, Vop3B16OpsSelectSourceHalvesAndMergeDestination) {
  amdgpu::GpuMemory mem("gfx1250_vop3_b16_mem");
  amdgpu::L2Cache l2("gfx1250_vop3_b16_l2");
  auto cu = make_gfx1250_cu(mem, l2, "gfx1250_vop3_b16_cu");
  ASSERT_NE(cu, nullptr);
  auto *wf = cu->dispatch_wf(0, 0, 128, 128);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);

  const uint32_t vb = wf->vgpr_alloc().base;

  cu->write_vgpr(vb + 0, 0, 0x55551234u);
  cu->write_vgpr(vb + 1, 0, 0xabcd0001u);
  cu->write_vgpr(vb + 2, 0, 0x22220034u);
  gfx1250::Vop3MachineInst or_raw{};
  or_raw.vdst = 0;
  or_raw.src0 = 256 + 1;
  or_raw.src1 = 256 + 2;
  or_raw.opsel = 0x8u | 0x1u; // dst.h, src0.h, src1.l
  gfx1250::VOrB16Vop3 or_inst(reinterpret_cast<const gfx1250::MachineInst *>(&or_raw));

  or_inst.execute_impl(*wf);

  EXPECT_EQ(cu->read_vgpr(vb + 0, 0), 0xabfd1234u);

  cu->write_vgpr(vb + 0, 0, 0xaaaa5678u);
  cu->write_vgpr(vb + 1, 0, 0x003c0001u);
  gfx1250::Vop3MachineInst shl_raw{};
  shl_raw.vdst = 0;
  shl_raw.src0 = 128 + 8;
  shl_raw.src1 = 256 + 1;
  shl_raw.opsel = 0x8u | 0x2u; // dst.h, src1.h
  gfx1250::VLshlrevB16Vop3 shl_inst(reinterpret_cast<const gfx1250::MachineInst *>(&shl_raw));

  shl_inst.execute_impl(*wf);

  EXPECT_EQ(cu->read_vgpr(vb + 0, 0), 0x3c005678u);
}

TEST(Gfx1250HwregTest, IbSts2ReportsNoClusters) {
  amdgpu::GpuMemory mem("gfx1250_ib_sts2_mem");
  amdgpu::L2Cache l2("gfx1250_ib_sts2_l2");
  auto cu = make_gfx1250_cu(mem, l2, "gfx1250_ib_sts2_cu");
  ASSERT_NE(cu, nullptr);
  auto *wf = cu->dispatch_wf(0, 0, 128, 128);
  ASSERT_NE(wf, nullptr);

  constexpr uint32_t kIbSts2 = 28;
  constexpr uint32_t kOffset = 6;
  constexpr uint32_t kSize = 4;
  gfx1250::SopkMachineInst raw{};
  raw.simm16 = kIbSts2 | (kOffset << 6) | ((kSize - 1) << 11);
  raw.sdst = 4;
  raw.op = 18;
  raw.encoding = 0xb;
  gfx1250::SGetregB32Sopk inst(reinterpret_cast<const gfx1250::MachineInst *>(&raw));

  cu->write_sgpr(wf->sgpr_alloc().base + 4, 0xdeadbeefu);
  inst.execute_impl(*wf);

  EXPECT_EQ(cu->read_sgpr(wf->sgpr_alloc().base + 4), 0u);
}

TEST(Gfx1250OperandTest, Scalar64WriteToNullIsNoop) {
  amdgpu::GpuMemory mem("gfx1250_null_operand_mem");
  amdgpu::L2Cache l2("gfx1250_null_operand_l2");
  auto cu = make_gfx1250_cu(mem, l2, "gfx1250_null_operand_cu");
  ASSERT_NE(cu, nullptr);
  auto *wf = cu->dispatch_wf(0, 0, 128, 128);
  ASSERT_NE(wf, nullptr);

  const uint32_t sbase = wf->sgpr_alloc().base;
  cu->write_sgpr(sbase, 0x12345678u);
  cu->write_sgpr(sbase + 1, 0x9abcdef0u);

  gfx1250::Operand null_dst(64, gfx1250::OperandType::OPR_SDST, 124);
  EXPECT_NO_THROW(null_dst.write_scalar64(*wf, 0xfedcba9876543210ULL));
  EXPECT_EQ(cu->read_sgpr(sbase), 0x12345678u);
  EXPECT_EQ(cu->read_sgpr(sbase + 1), 0x9abcdef0u);
}

TEST(Gfx1250OperandTest, FlatScratchSourceReadsWaveScratchBase) {
  amdgpu::GpuMemory mem("gfx1250_flat_scratch_source_mem");
  amdgpu::L2Cache l2("gfx1250_flat_scratch_source_l2");
  auto cu = make_gfx1250_cu(mem, l2, "gfx1250_flat_scratch_source_cu");
  ASSERT_NE(cu, nullptr);
  auto *wf = cu->dispatch_wf(0, 0, 128, 128);
  ASSERT_NE(wf, nullptr);

  constexpr uint64_t kScratchBase = 0x12345678abcdef00ULL;
  wf->set_scratch_base(kScratchBase);

  gfx1250::Operand scratch_lo(32, gfx1250::OperandType::OPR_SRC, 230);
  gfx1250::Operand scratch_hi(32, gfx1250::OperandType::OPR_SRC, 231);
  gfx1250::Operand scratch_base(64, gfx1250::OperandType::OPR_SRC, 230);

  EXPECT_TRUE(gfx1250::Isa::simd_capable_value(gfx1250::OperandType::OPR_SRC, 230));
  EXPECT_TRUE(gfx1250::Isa::simd_capable_value(gfx1250::OperandType::OPR_SRC, 231));
  EXPECT_EQ(scratch_lo.read_scalar(*wf), static_cast<uint32_t>(kScratchBase));
  EXPECT_EQ(scratch_hi.read_scalar(*wf), static_cast<uint32_t>(kScratchBase >> 32));
  EXPECT_EQ(scratch_lo.read_lane(*wf, 0), static_cast<uint32_t>(kScratchBase));
  EXPECT_EQ(scratch_hi.read_lane(*wf, 0), static_cast<uint32_t>(kScratchBase >> 32));
  EXPECT_EQ(scratch_base.read_scalar64(*wf), kScratchBase);
  EXPECT_EQ(scratch_base.read_lane64(*wf, 0), kScratchBase);
}

// CDNA1 has no AccVGPRs; CDNA2/3/4 have 256.
static_assert(cdna1::Isa::MAX_ACC_VGPRS_PER_WF == 0);
static_assert(cdna2::Isa::MAX_ACC_VGPRS_PER_WF == 256);
static_assert(cdna3::Isa::MAX_ACC_VGPRS_PER_WF == 256);

// ---------------------------------------------------------------------------
// MFMA register layout tests
// ---------------------------------------------------------------------------

TEST(MfmaExecTest, InputLocF32_32x32) {
  // v_mfma_f32_32x32x1f32: M=32, K=1, B=1, f32 inputs.
  // lanes_per_block = 64 / (32 * 1) = 2, elems_per_group = 1 / 2 = 0 -> special case.
  // Actually for M=32,K=2,B=1: lanes_per_block = 64/(32*1) = 2, elems = 2/2 = 1.
  // Use M=4, K=4, B=4 which is v_mfma_f32_4x4x4f16 (valid shape).
  // lanes_per_block = 64 / (4 * 4) = 4, elems_per_group = 4/4 = 1.
  auto loc = amdgpu::input_loc(4, 4, 4, /*i=*/2, /*k=*/0, /*b=*/0, 32);
  EXPECT_EQ(loc.vgpr_offset, 0u);
  EXPECT_EQ(loc.lane, 2u); // b*dim + ... = 0*4 + (0/1)*4*4 + 2 = 2
  EXPECT_EQ(loc.sub_element, 0u);
}

TEST(MfmaExecTest, InputLocF16_16x16) {
  // 16x16x16 with 1 block, f16 inputs: each lane holds 16 * 2B = 32B = 8 dwords.
  // lanes_per_block = 64 / (16 * 1) = 4
  // elems_per_group = 16 / 4 = 4
  // For i=0, k=0, b=0: local=0%4=0, lane=0*16+0*16*1+0=0, per_dword=2
  // vgpr_offset = 0/2 = 0, sub_element = 0%2 = 0
  auto loc = amdgpu::input_loc(16, 16, 1, 0, 0, 0, 16);
  EXPECT_EQ(loc.vgpr_offset, 0u);
  EXPECT_EQ(loc.lane, 0u);
  EXPECT_EQ(loc.sub_element, 0u);

  // k=1: local=1, vgpr_offset = 1/2 = 0, sub_element = 1
  auto loc1 = amdgpu::input_loc(16, 16, 1, 0, 1, 0, 16);
  EXPECT_EQ(loc1.vgpr_offset, 0u);
  EXPECT_EQ(loc1.sub_element, 1u);
}

TEST(MfmaExecTest, OutputLoc32_4x4) {
  // 4x4 matrix, block 0: reg = column index, lane = row index.
  auto loc = amdgpu::output_loc_32(4, 4, /*col=*/2, /*row=*/1, /*b=*/0);
  EXPECT_EQ(loc.reg, 2u);
  EXPECT_EQ(loc.lane, 1u);
}

TEST(MfmaExecTest, WmmaInputLocF16_16x16x32) {
  auto first = amdgpu::wmma_input_loc(16, 32, /*i=*/0, /*k=*/0, 16);
  EXPECT_EQ(first.vgpr_offset, 0u);
  EXPECT_EQ(first.lane, 0u);
  EXPECT_EQ(first.sub_element, 0u);

  auto last_low_half = amdgpu::wmma_input_loc(16, 32, /*i=*/0, /*k=*/15, 16);
  EXPECT_EQ(last_low_half.vgpr_offset, 7u);
  EXPECT_EQ(last_low_half.lane, 0u);
  EXPECT_EQ(last_low_half.sub_element, 1u);

  auto first_high_half = amdgpu::wmma_input_loc(16, 32, /*i=*/0, /*k=*/16, 16);
  EXPECT_EQ(first_high_half.vgpr_offset, 0u);
  EXPECT_EQ(first_high_half.lane, 16u);
  EXPECT_EQ(first_high_half.sub_element, 0u);
}

TEST(MfmaExecTest, WmmaInputLocF4_32x16x128) {
  auto a_last = amdgpu::wmma_input_loc(32, 128, /*i=*/31, /*k=*/127, 4);
  EXPECT_EQ(a_last.vgpr_offset, 15u);
  EXPECT_EQ(a_last.lane, 31u);
  EXPECT_EQ(a_last.sub_element, 7u);
  EXPECT_EQ(a_last.bit_offset, 28u);

  auto b_group_switch = amdgpu::wmma_input_loc(16, 128, /*i=*/15, /*k=*/64, 4);
  EXPECT_EQ(b_group_switch.vgpr_offset, 0u);
  EXPECT_EQ(b_group_switch.lane, 31u);
  EXPECT_EQ(b_group_switch.sub_element, 0u);
  EXPECT_EQ(b_group_switch.bit_offset, 0u);
}

TEST(MfmaExecTest, WmmaOutputLoc32) {
  auto first = amdgpu::wmma_output_loc_32(16, 16, /*row=*/0, /*col=*/0);
  EXPECT_EQ(first.reg, 0u);
  EXPECT_EQ(first.lane, 0u);

  auto last_low_half = amdgpu::wmma_output_loc_32(16, 16, /*row=*/7, /*col=*/15);
  EXPECT_EQ(last_low_half.reg, 7u);
  EXPECT_EQ(last_low_half.lane, 15u);

  auto first_high_half = amdgpu::wmma_output_loc_32(16, 16, /*row=*/8, /*col=*/0);
  EXPECT_EQ(first_high_half.reg, 0u);
  EXPECT_EQ(first_high_half.lane, 16u);

  auto wide = amdgpu::wmma_output_loc_32(32, 16, /*row=*/31, /*col=*/15);
  EXPECT_EQ(wide.reg, 15u);
  EXPECT_EQ(wide.lane, 31u);
}

TEST(MfmaExecTest, SwmmacSparseDirectIndexEntries) {
  EXPECT_EQ(amdgpu::swmmac_dense_k(/*index_set=*/4, /*compressed_k=*/0,
                                   /*local_compressed_k=*/0),
            0u);
  EXPECT_EQ(amdgpu::swmmac_dense_k(/*index_set=*/4, /*compressed_k=*/1,
                                   /*local_compressed_k=*/1),
            1u);
  EXPECT_EQ(amdgpu::swmmac_dense_k(/*index_set=*/14, /*compressed_k=*/0,
                                   /*local_compressed_k=*/0),
            2u);
  EXPECT_EQ(amdgpu::swmmac_dense_k(/*index_set=*/14, /*compressed_k=*/1,
                                   /*local_compressed_k=*/1),
            3u);
  EXPECT_EQ(amdgpu::swmmac_dense_k(/*index_set=*/2u << 4, /*compressed_k=*/2,
                                   /*local_compressed_k=*/2),
            6u);
  EXPECT_EQ(amdgpu::swmmac_dense_k_from_word(/*index_word=*/0x00000002u,
                                             /*compressed_k=*/0,
                                             /*local_compressed_k=*/16),
            2u);
}

TEST(MfmaExecTest, SwmmacReadsFullIndexSets) {
  amdgpu::GpuMemory mem("swmmac_index_mem");
  amdgpu::L2Cache l2("swmmac_index_l2");
  auto cu = make_gfx1250_cu(mem, l2, "swmmac_index_cu");
  ASSERT_NE(cu, nullptr);
  auto *wf = cu->dispatch_wf(0, 0, 128, 128);
  ASSERT_NE(wf, nullptr);

  const uint32_t base = wf->vgpr_alloc().base;
  cu->write_vgpr(base, 0, 0xAAAAAAAAu);
  cu->write_vgpr(base + 1, 0, 0x55555555u);

  EXPECT_EQ(amdgpu::read_swmmac_index_set(*cu, base, 0, /*index_entries=*/16,
                                          /*index_key=*/0),
            0xAAAAAAAAull);
  EXPECT_EQ(amdgpu::read_swmmac_index_set(*cu, base, 0, /*index_entries=*/16,
                                          /*index_key=*/1),
            0xAAAAull);
  EXPECT_EQ(amdgpu::read_swmmac_index_set(*cu, base, 0, /*index_entries=*/32,
                                          /*index_key=*/0),
            0x55555555AAAAAAAAull);
  EXPECT_EQ(amdgpu::read_swmmac_index_set(*cu, base, 0, /*index_entries=*/32,
                                          /*index_key=*/1),
            0x55555555ull);
}

TEST(MfmaExecTest, SwmmacReadsIndexWordForReductionGroup) {
  amdgpu::GpuMemory mem("swmmac_index_window_mem");
  amdgpu::L2Cache l2("swmmac_index_window_l2");
  auto cu = make_gfx1250_cu(mem, l2, "swmmac_index_window_cu");
  ASSERT_NE(cu, nullptr);
  auto *wf = cu->dispatch_wf(0, 0, 128, 128);
  ASSERT_NE(wf, nullptr);

  const uint32_t base = wf->vgpr_alloc().base;
  cu->write_vgpr(base, 0, 0xAAAAAAAAu);
  cu->write_vgpr(base + 1, 0, 0x55555555u);

  EXPECT_EQ(amdgpu::read_swmmac_index_word(*cu, base, 0, /*index_entries=*/32,
                                           /*index_key=*/0, /*k_group=*/0),
            0xAAAAAAAAu);
  EXPECT_EQ(amdgpu::read_swmmac_index_word(*cu, base, 0, /*index_entries=*/32,
                                           /*index_key=*/0, /*k_group=*/1),
            0x55555555u);
  EXPECT_EQ(amdgpu::read_swmmac_index_word(*cu, base, 0, /*index_entries=*/32,
                                           /*index_key=*/1, /*k_group=*/0),
            0x55555555u);
}

TEST(MfmaExecTest, SwmmacF16UsesSparseMetadata) {
  EXPECT_FLOAT_EQ(run_swmmac_f16_row0_col0(/*index_word=*/4u, /*index_key=*/0), 31.0f);
  EXPECT_FLOAT_EQ(run_swmmac_f16_row0_col0(/*index_word=*/14u, /*index_key=*/0), 61.0f);
}

TEST(MfmaExecTest, SwmmacF16HonorsIndexKey) {
  EXPECT_FLOAT_EQ(run_swmmac_f16_row0_col0(/*index_word=*/14u << 16, /*index_key=*/1), 61.0f);
}

TEST(MfmaExecTest, WmmaF16UsesVgprMsbHighBankOperands) {
  EXPECT_FLOAT_EQ(run_v_wmma_f16_high_bank_row0_col0(), 31.0f);
}

TEST(CacheCoherenceTest, VectorLoadRefetchesRemoteWriteThroughStore) {
  amdgpu::GpuMemory mem("coherent_mem");
  amdgpu::L2Cache l2_a("l2_a");
  amdgpu::L2Cache l2_b("l2_b");
  l2_a.set_backing_memory(&mem);
  l2_b.set_backing_memory(&mem);
  amdgpu::L1VectorCache l1_a(&l2_a);
  amdgpu::L1VectorCache l1_b(&l2_b);

  constexpr uint64_t kAddr = 0x5c03200000ULL;
  std::array<uint64_t, 64> addrs{};
  addrs[0] = kAddr;

  std::array<uint8_t, 64 * sizeof(uint32_t)> load_buf{};
  l1_b.load(addrs.data(), 1, sizeof(uint32_t), 1, load_buf.data(), amdgpu::Mtype::RW,
            /*non_temporal=*/false);
  uint32_t observed = 0;
  std::memcpy(&observed, load_buf.data(), sizeof(observed));
  ASSERT_EQ(observed, 0u);

  constexpr uint32_t kValue = 0x12345678u;
  std::array<uint8_t, 64 * sizeof(uint32_t)> store_buf{};
  std::memcpy(store_buf.data(), &kValue, sizeof(kValue));
  l1_a.store(addrs.data(), 1, sizeof(uint32_t), 1, store_buf.data(), amdgpu::Mtype::RW,
             /*non_temporal=*/false);

  load_buf.fill(0);
  l1_b.load(addrs.data(), 1, sizeof(uint32_t), 1, load_buf.data(), amdgpu::Mtype::RW,
            /*non_temporal=*/false);
  std::memcpy(&observed, load_buf.data(), sizeof(observed));
  EXPECT_EQ(observed, kValue);
}

TEST(TransposeLoadTest, TrB16TransposesHalfwordsWithinEightLaneGroups) {
  constexpr uint32_t kWaveSize = 32;
  constexpr uint32_t kNumElems = 4;
  constexpr uint32_t kBytesPerLane = kNumElems * 4;
  constexpr uint32_t kHalfwordsPerLane = kBytesPerLane / 2;
  std::vector<uint8_t> data(kWaveSize * kBytesPerLane);

  auto write_u16 = [&](uint32_t lane, uint32_t halfword, uint16_t value) {
    const uint32_t offset = lane * kBytesPerLane + halfword * 2;
    data[offset] = static_cast<uint8_t>(value & 0xffu);
    data[offset + 1] = static_cast<uint8_t>(value >> 8);
  };
  auto read_u16 = [&](uint32_t lane, uint32_t halfword) {
    const uint32_t offset = lane * kBytesPerLane + halfword * 2;
    return static_cast<uint16_t>(data[offset] | (static_cast<uint16_t>(data[offset + 1]) << 8));
  };

  for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
    for (uint32_t halfword = 0; halfword < kHalfwordsPerLane; ++halfword)
      write_u16(lane, halfword, static_cast<uint16_t>(lane * 0x10u + halfword));
  }

  amdgpu::transpose_b16(data, kNumElems, kWaveSize);

  for (uint32_t group_start = 0; group_start < kWaveSize; group_start += kHalfwordsPerLane) {
    for (uint32_t halfword = 0; halfword < kHalfwordsPerLane; ++halfword) {
      const uint32_t dest_lane = group_start + halfword;
      for (uint32_t lane_in_group = 0; lane_in_group < kHalfwordsPerLane; ++lane_in_group) {
        EXPECT_EQ(read_u16(dest_lane, lane_in_group),
                  static_cast<uint16_t>((group_start + lane_in_group) * 0x10u + halfword));
      }
    }
  }
}

TEST(MfmaExecTest, IntegerExtractorsSignOrZeroExtendPackedValues) {
  amdgpu::GpuMemory mem("integer_extract_mem");
  amdgpu::L2Cache l2("integer_extract_l2");
  auto cu = make_gfx1250_cu(mem, l2, "integer_extract_cu");
  ASSERT_NE(cu, nullptr);
  auto *wf = cu->dispatch_wf(0, 0, 128, 128);
  ASSERT_NE(wf, nullptr);

  const uint32_t base = wf->vgpr_alloc().base;
  cu->write_vgpr(base, 0, 0x0000FF8Fu);
  EXPECT_EQ(amdgpu::extract_u8(*cu, base, amdgpu::InputLoc{0, 0, 1, 8, 8}), 0xFF);
  EXPECT_EQ(amdgpu::extract_i8(*cu, base, amdgpu::InputLoc{0, 0, 1, 8, 8}), -1);
  EXPECT_EQ(amdgpu::extract_u4(*cu, base, amdgpu::InputLoc{0, 0, 0, 0, 4}), 0xF);
  EXPECT_EQ(amdgpu::extract_i4(*cu, base, amdgpu::InputLoc{0, 0, 0, 0, 4}), -1);
  EXPECT_EQ(amdgpu::extract_u4(*cu, base, amdgpu::InputLoc{0, 0, 1, 4, 4}), 0x8);
  EXPECT_EQ(amdgpu::extract_i4(*cu, base, amdgpu::InputLoc{0, 0, 1, 4, 4}), -8);
}

TEST(PackedDotExecTest, Dot4I32Iu8UsesOperandSignednessModifiers) {
  EXPECT_EQ(run_dot4_i32_iu8_lane0(/*neg_bits=*/0), 33551u);
  EXPECT_EQ(run_dot4_i32_iu8_lane0(/*neg_bits=*/1), 32015u);
  EXPECT_EQ(run_dot4_i32_iu8_lane0(/*neg_bits=*/2), 1039u);
  EXPECT_EQ(run_dot4_i32_iu8_lane0(/*neg_bits=*/3), 0xFFFFFE0Fu);
}

TEST(PackedDotExecTest, Dot8I32Iu4UsesOperandSignednessModifiers) {
  EXPECT_EQ(run_dot8_i32_iu4_lane0(/*neg_bits=*/0), 293u);
  EXPECT_EQ(run_dot8_i32_iu4_lane0(/*neg_bits=*/1), 85u);
  EXPECT_EQ(run_dot8_i32_iu4_lane0(/*neg_bits=*/2), 133u);
  EXPECT_EQ(run_dot8_i32_iu4_lane0(/*neg_bits=*/3), 0xFFFFFFB5u);
}

TEST(MfmaExecTest, WmmaI32Iu8DefaultsToUnsignedOperands) {
  EXPECT_EQ(run_wmma_i32_iu8_row0_col0(amdgpu::extract_u8, amdgpu::extract_u8,
                                       /*clamp=*/false),
            510u);
}

TEST(MfmaExecTest, WmmaI32Iu8SignedModifiersSignExtendOperands) {
  EXPECT_EQ(run_wmma_i32_iu8_row0_col0(amdgpu::extract_i8, amdgpu::extract_u8,
                                       /*clamp=*/false),
            0xFFFFFFFEu);
}

TEST(MfmaExecTest, WmmaI32ClampPreservesInRangeSignedResults) {
  EXPECT_EQ(run_wmma_i32_iu8_row0_col0(amdgpu::extract_i8, amdgpu::extract_u8,
                                       /*clamp=*/true),
            0xFFFFFFFEu);
}

TEST(MfmaExecTest, WmmaI32ClampSaturatesToSignedI32Range) {
  EXPECT_EQ(amdgpu::pack_i32_acc(static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1,
                                 /*clamp=*/true),
            static_cast<uint32_t>(std::numeric_limits<int32_t>::max()));
  EXPECT_EQ(amdgpu::pack_i32_acc(static_cast<int64_t>(std::numeric_limits<int32_t>::min()) - 1,
                                 /*clamp=*/true),
            static_cast<uint32_t>(std::numeric_limits<int32_t>::min()));
}

TEST(MfmaExecTest, SwmmacI32Iu8UsesSparseMetadataAndUnsignedOperands) {
  EXPECT_EQ(run_swmmac_i32_iu8_row0_col0(amdgpu::extract_u8, amdgpu::extract_u8), 510u);
}

TEST(MfmaExecTest, ResolveAccConstant) {
  // Encoding value 0-255 = inline constant. The callback should be invoked.
  uint32_t const_acc = 0;
  uint32_t result = amdgpu::resolve_acc<amdgpu::AccMode::Unified>(
      /*vb=*/100, /*dst=*/200, /*src2_ev=*/128, const_acc, [&]() -> uint32_t { return 42u; });
  EXPECT_EQ(const_acc, 42u);
  EXPECT_EQ(result, 200u); // Returns dst when constant.
}

TEST(MfmaExecTest, ResolveAccVgpr) {
  // Encoding value 256-511 = VGPR.
  uint32_t const_acc = 0;
  uint32_t result = amdgpu::resolve_acc<amdgpu::AccMode::Unified>(
      /*vb=*/100, /*dst=*/200, /*src2_ev=*/260, const_acc, [&]() -> uint32_t { return 99u; });
  EXPECT_EQ(const_acc, amdgpu::ACC_FROM_VGPR);
  EXPECT_EQ(result, 100u + 4u); // vb + (260 - 256)
}

TEST(MfmaExecTest, ResolveAccAccVgpr) {
  // Encoding value 768-1023 = AccVGPR (unified alias).
  uint32_t const_acc = 0;
  uint32_t result = amdgpu::resolve_acc<amdgpu::AccMode::Unified>(
      /*vb=*/100, /*dst=*/200, /*src2_ev=*/770, const_acc, [&]() -> uint32_t { return 99u; });
  EXPECT_EQ(const_acc, amdgpu::ACC_FROM_VGPR);
  EXPECT_EQ(result, 100u + 256u + 2u); // vb + ACC_VGPR_OFFSET + (770 - 768)
}

// ---------------------------------------------------------------------------
// CU factory tests — verify all AMDGPU ISAs can be instantiated
// ---------------------------------------------------------------------------

class CuFactoryTest : public ::testing::TestWithParam<rj_code_arch_t> {};

TEST_P(CuFactoryTest, CreatesSuccessfully) {
  auto arch = GetParam();
  amdgpu::GpuMemory mem("test_mem");
  amdgpu::L2Cache l2("test_l2");

  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = arch;
  cfg.num_wf_slots = 2;
  cfg.sgprs_per_wf = 102;
  cfg.vgprs_per_wf = 256;
  cfg.lds_size_kb = 64;

  auto cu = amdgpu::ComputeUnitCore::create("test_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);
  EXPECT_EQ(cu->arch(), arch);
}

INSTANTIATE_TEST_SUITE_P(AllIsas, CuFactoryTest,
                         ::testing::Values(ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2,
                                           ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4,
                                           ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2,
                                           ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
                                           ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_GFX1250));

// ---------------------------------------------------------------------------
// DPP permutation tests
// ---------------------------------------------------------------------------

TEST(DppPermuteTest, QuadPerm) {
  using namespace amdgpu::dpp;
  // quad_perm(1,0,3,2) = swap pairs within each quad
  // Encoding: lane0->1, lane1->0, lane2->3, lane3->2
  // = (1 << 0) | (0 << 2) | (3 << 4) | (2 << 6) = 0xB1
  bool oob = false;
  EXPECT_EQ(dpp_permute(0xB1, 0, 64, oob), 1);
  EXPECT_FALSE(oob);
  EXPECT_EQ(dpp_permute(0xB1, 1, 64, oob), 0);
  EXPECT_FALSE(oob);
  EXPECT_EQ(dpp_permute(0xB1, 2, 64, oob), 3);
  EXPECT_FALSE(oob);
  EXPECT_EQ(dpp_permute(0xB1, 3, 64, oob), 2);
  EXPECT_FALSE(oob);
  // Quad boundary: lane 4 starts a new quad, same permutation.
  EXPECT_EQ(dpp_permute(0xB1, 4, 64, oob), 5);
  EXPECT_FALSE(oob);
  EXPECT_EQ(dpp_permute(0xB1, 5, 64, oob), 4);
  EXPECT_FALSE(oob);
}

TEST(DppPermuteTest, RowShr1) {
  using namespace amdgpu::dpp;
  bool oob = false;
  // row_shr 1 = 0x111: data shifts right, so lane K reads from lane K-1.
  EXPECT_EQ(dpp_permute(0x111, 1, 64, oob), 0);
  EXPECT_FALSE(oob);
  EXPECT_EQ(dpp_permute(0x111, 15, 64, oob), 14);
  EXPECT_FALSE(oob);
  // Lane 0 (first in row) goes OOB (no lane -1).
  oob = false;
  dpp_permute(0x111, 0, 64, oob);
  EXPECT_TRUE(oob);
}

TEST(DppPermuteTest, RowShl1) {
  using namespace amdgpu::dpp;
  bool oob = false;
  // row_shl 1 = 0x101: data shifts left, so lane K reads from lane K+1.
  EXPECT_EQ(dpp_permute(0x101, 0, 64, oob), 1);
  EXPECT_FALSE(oob);
  EXPECT_EQ(dpp_permute(0x101, 14, 64, oob), 15);
  EXPECT_FALSE(oob);
  // Lane 15 (last in row) goes OOB (no lane 16 in this row).
  oob = false;
  dpp_permute(0x101, 15, 64, oob);
  EXPECT_TRUE(oob);
}

TEST(DppPermuteTest, RowMirror) {
  using namespace amdgpu::dpp;
  bool oob = false;
  // row_mirror = 0x140: reverse lane order within a row.
  EXPECT_EQ(dpp_permute(0x140, 0, 64, oob), 15);
  EXPECT_EQ(dpp_permute(0x140, 15, 64, oob), 0);
  EXPECT_EQ(dpp_permute(0x140, 7, 64, oob), 8);
  // Second row.
  EXPECT_EQ(dpp_permute(0x140, 16, 64, oob), 31);
}

TEST(DppPermuteTest, RowXmask) {
  using namespace amdgpu::dpp;
  bool oob = false;
  // row_xmask with mask=1 = 0x151: XOR lane offset with 1 (swap adjacent pairs).
  EXPECT_EQ(dpp_permute(0x151, 0, 64, oob), 1);
  EXPECT_EQ(dpp_permute(0x151, 1, 64, oob), 0);
  EXPECT_EQ(dpp_permute(0x151, 2, 64, oob), 3);
  EXPECT_EQ(dpp_permute(0x151, 3, 64, oob), 2);
}

TEST(DppPermuteTest, DppRead) {
  using namespace amdgpu::dpp;
  // Set up 64 source values: src[i] = i * 10.
  uint32_t src[64];
  for (int i = 0; i < 64; ++i)
    src[i] = i * 10;

  // row_shr 1: lane 1 reads from lane 0.
  uint32_t val = dpp_read(src, 1, 64, 0x111, 0xF, 0xF, 1, 999);
  EXPECT_EQ(val, 0u); // src[0] = 0

  // Lane 5 reads from lane 4 (src[4] = 40).
  val = dpp_read(src, 5, 64, 0x111, 0xF, 0xF, 1, 999);
  EXPECT_EQ(val, 40u);

  // Lane 0 goes OOB, bound_ctrl=1 -> returns 0.
  val = dpp_read(src, 0, 64, 0x111, 0xF, 0xF, 1, 999);
  EXPECT_EQ(val, 0u);

  // Lane 0 goes OOB, bound_ctrl=0 -> returns old_val.
  val = dpp_read(src, 0, 64, 0x111, 0xF, 0xF, 0, 999);
  EXPECT_EQ(val, 999u);

  // Row mask disables row 0 (bits [3:0], row0 = lanes 0-15).
  val = dpp_read(src, 5, 64, 0x111, 0xE, 0xF, 1, 999);
  EXPECT_EQ(val, 999u); // row 0 masked -> old_val

  // Bank mask disables bank 1 (lanes 4-7 within each row).
  val = dpp_read(src, 5, 64, 0x111, 0xF, 0xD, 1, 999);
  EXPECT_EQ(val, 999u); // bank 1 disabled -> old_val

  // Unmasked lane in row 1: lane 17 reads from lane 16.
  val = dpp_read(src, 17, 64, 0x111, 0xF, 0xF, 1, 999);
  EXPECT_EQ(val, 160u); // src[16] = 160
}

// ---------------------------------------------------------------------------
// SDWA tests
// ---------------------------------------------------------------------------

TEST(SdwaTest, SrcSelect) {
  using namespace amdgpu::sdwa;
  uint32_t val = 0xDEADBEEF;

  EXPECT_EQ(sdwa_src_select(val, BYTE_0, false), 0xEFu);
  EXPECT_EQ(sdwa_src_select(val, BYTE_1, false), 0xBEu);
  EXPECT_EQ(sdwa_src_select(val, BYTE_2, false), 0xADu);
  EXPECT_EQ(sdwa_src_select(val, BYTE_3, false), 0xDEu);
  EXPECT_EQ(sdwa_src_select(val, WORD_0, false), 0xBEEFu);
  EXPECT_EQ(sdwa_src_select(val, WORD_1, false), 0xDEADu);
  EXPECT_EQ(sdwa_src_select(val, DWORD, false), val);

  // Sign extension.
  EXPECT_EQ(sdwa_src_select(0x00000080, BYTE_0, true), 0xFFFFFF80u);
  EXPECT_EQ(sdwa_src_select(0x00000080, BYTE_0, false), 0x80u);
  EXPECT_EQ(sdwa_src_select(0x00008000, WORD_0, true), 0xFFFF8000u);
}

TEST(SdwaTest, DstMerge) {
  using namespace amdgpu::sdwa;
  // Write result byte 0x42 into BYTE_1, zero-pad rest.
  uint32_t merged = sdwa_dst_merge(0x42, 0xAAAAAAAA, BYTE_1, UNUSED_PAD);
  EXPECT_EQ(merged, 0x00004200u);

  // Preserve unused bytes.
  merged = sdwa_dst_merge(0x42, 0xAABBCCDD, BYTE_1, UNUSED_PRESERVE);
  EXPECT_EQ(merged, 0xAABB42DDu);

  // Full dword: just return result.
  merged = sdwa_dst_merge(0x12345678, 0xAAAAAAAA, DWORD, UNUSED_PAD);
  EXPECT_EQ(merged, 0x12345678u);
}

// ---------------------------------------------------------------------------
// gfx1250 address calculation tests
// ---------------------------------------------------------------------------

TEST(Gfx1250AddrCalcTest, GlobalScaleOffsetScalesVaddrOnly) {
  amdgpu::GpuMemory mem("gfx1250_scale_offset_mem");
  amdgpu::L2Cache l2("gfx1250_scale_offset_l2");
  auto cu = make_gfx1250_cu(mem, l2, "gfx1250_scale_offset_cu");
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 128);
  ASSERT_NE(wf, nullptr);

  constexpr uint64_t kBase = 0x200000;
  constexpr uint32_t kSaddr = 4;
  cu->write_sgpr(wf->sgpr_alloc().base + kSaddr, static_cast<uint32_t>(kBase));
  cu->write_sgpr(wf->sgpr_alloc().base + kSaddr + 1, static_cast<uint32_t>(kBase >> 32));

  const uint32_t vbase = wf->vgpr_alloc().base;
  cu->write_vgpr(vbase, 0, 1);
  cu->write_vgpr(vbase, 1, 3);
  wf->set_exec(0x3);

  gfx1250::VglobalMachineInst inst{};
  inst.saddr = kSaddr;
  inst.vaddr = 0;
  inst.ioffset = 2;
  inst.scale_offset = 1;

  amdgpu::VectorMemState d(amdgpu::GLOBAL_MEM);
  d.elem_size = 4;
  d.num_elems = 2;
  gfx1250::flat_calculate_addresses(inst, *wf, d);

  EXPECT_EQ(d.per_lane_addr[0], kBase + 1 * 8 + 2);
  EXPECT_EQ(d.per_lane_addr[1], kBase + 3 * 8 + 2);
}

TEST(Gfx1250AddrCalcTest, GlobalNullSaddrUsesFullVaddrPair) {
  amdgpu::GpuMemory mem("gfx1250_global_null_saddr_mem");
  amdgpu::L2Cache l2("gfx1250_global_null_saddr_l2");
  auto cu = make_gfx1250_cu(mem, l2, "gfx1250_global_null_saddr_cu");
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 128);
  ASSERT_NE(wf, nullptr);

  const uint32_t vbase = wf->vgpr_alloc().base;
  cu->write_vgpr(vbase, 0, 0x2000);
  cu->write_vgpr(vbase + 1, 0, 0x0001);
  wf->set_exec(0x1);

  gfx1250::VglobalMachineInst inst{};
  inst.saddr = gfx1250::OPR_SREG_NULL;
  inst.vaddr = 0;
  inst.ioffset = 0x20;

  amdgpu::VectorMemState d(amdgpu::GLOBAL_MEM);
  d.elem_size = 16;
  d.num_elems = 1;
  gfx1250::flat_calculate_addresses(inst, *wf, d);

  EXPECT_EQ(d.per_lane_addr[0], 0x1'0000'2020ULL);
}

TEST(Gfx1250AddrCalcTest, GlobalSaddrZeroUsesScalarBaseAndVaddrOffset) {
  amdgpu::GpuMemory mem("gfx1250_global_saddr_zero_mem");
  amdgpu::L2Cache l2("gfx1250_global_saddr_zero_l2");
  auto cu = make_gfx1250_cu(mem, l2, "gfx1250_global_saddr_zero_cu");
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 128);
  ASSERT_NE(wf, nullptr);

  constexpr uint64_t kBase = 0x2'0000'0000ULL;
  const uint32_t sbase = wf->sgpr_alloc().base;
  cu->write_sgpr(sbase + 0, static_cast<uint32_t>(kBase));
  cu->write_sgpr(sbase + 1, static_cast<uint32_t>(kBase >> 32));

  const uint32_t vbase = wf->vgpr_alloc().base;
  cu->write_vgpr(vbase, 0, 0x3000);
  cu->write_vgpr(vbase + 1, 0, 0x7777);
  wf->set_exec(0x1);

  gfx1250::VglobalMachineInst inst{};
  inst.saddr = 0;
  inst.vaddr = 0;
  inst.ioffset = 0x40;

  amdgpu::VectorMemState d(amdgpu::GLOBAL_MEM);
  d.elem_size = 16;
  d.num_elems = 1;
  gfx1250::flat_calculate_addresses(inst, *wf, d);

  EXPECT_EQ(d.per_lane_addr[0], kBase + 0x3040);
}

TEST(Gfx1250AddrCalcTest, SmemNullSoffsetIgnoresScalarOffsetSlot) {
  amdgpu::GpuMemory mem("gfx1250_smem_null_soffset_mem");
  amdgpu::L2Cache l2("gfx1250_smem_null_soffset_l2");
  auto cu = make_gfx1250_cu(mem, l2, "gfx1250_smem_null_soffset_cu");
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 128);
  ASSERT_NE(wf, nullptr);

  constexpr uint64_t kBase = 0x3'0000'2000ULL;
  const uint32_t sbase = wf->sgpr_alloc().base;
  cu->write_sgpr(sbase + 4, static_cast<uint32_t>(kBase));
  cu->write_sgpr(sbase + 5, static_cast<uint32_t>(kBase >> 32));
  cu->write_sgpr(sbase + gfx1250::OPR_SMEM_OFFSET_NULL, 0x1000);

  gfx1250::SmemMachineInst inst{};
  inst.sbase = 2;
  inst.soffset = gfx1250::OPR_SMEM_OFFSET_NULL;
  inst.ioffset = 0x14;

  EXPECT_EQ(gfx1250::smem_calculate_address(inst, *wf), kBase + 0x14);
}

TEST(Gfx1250AddrCalcTest, SmemKeepsByteGranularOffsets) {
  amdgpu::GpuMemory mem("gfx1250_smem_byte_offset_mem");
  amdgpu::L2Cache l2("gfx1250_smem_byte_offset_l2");
  auto cu = make_gfx1250_cu(mem, l2, "gfx1250_smem_byte_offset_cu");
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 128);
  ASSERT_NE(wf, nullptr);

  constexpr uint64_t kBase = 0x3'0000'2000ULL;
  const uint32_t sbase = wf->sgpr_alloc().base;
  cu->write_sgpr(sbase + 4, static_cast<uint32_t>(kBase));
  cu->write_sgpr(sbase + 5, static_cast<uint32_t>(kBase >> 32));

  gfx1250::SmemMachineInst inst{};
  inst.sbase = 2;
  inst.soffset = gfx1250::OPR_SMEM_OFFSET_NULL;
  inst.ioffset = 0x15;

  EXPECT_EQ(gfx1250::smem_calculate_address(inst, *wf), kBase + 0x15);
}

TEST(Gfx1250AddrCalcTest, ScratchOffVaddrIgnoresVaddrField) {
  amdgpu::GpuMemory mem("gfx1250_scratch_off_vaddr_mem");
  amdgpu::L2Cache l2("gfx1250_scratch_off_vaddr_l2");
  auto cu = make_gfx1250_cu(mem, l2, "gfx1250_scratch_off_vaddr_cu");
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 128);
  ASSERT_NE(wf, nullptr);

  constexpr uint64_t kScratchBase = 0x4'0000'0000ULL;
  constexpr uint32_t kLaneScratchSize = 0x100;
  wf->set_scratch_base(kScratchBase);
  wf->set_scratch_lane_size(kLaneScratchSize);
  wf->set_exec(0x3);
  cu->write_vgpr(wf->vgpr_alloc().base, 0, 0x4000);
  cu->write_vgpr(wf->vgpr_alloc().base, 1, 0x8000);

  gfx1250::VscratchMachineInst inst{};
  inst.saddr = gfx1250::OPR_SREG_NULL;
  inst.vaddr = 0;
  inst.sve = 0;
  inst.ioffset = 0x10;

  amdgpu::VectorMemState d(amdgpu::GLOBAL_MEM);
  d.elem_size = 16;
  d.num_elems = 1;
  gfx1250::flat_calculate_addresses(inst, *wf, d);

  EXPECT_EQ(d.per_lane_addr[0], kScratchBase + 0x10);
  EXPECT_EQ(d.per_lane_addr[1], kScratchBase + kLaneScratchSize + 0x10);
}

TEST(Gfx1250AddrCalcTest, ScratchScalarOffsetWorksWithoutVaddr) {
  amdgpu::GpuMemory mem("gfx1250_scratch_scalar_offset_mem");
  amdgpu::L2Cache l2("gfx1250_scratch_scalar_offset_l2");
  auto cu = make_gfx1250_cu(mem, l2, "gfx1250_scratch_scalar_offset_cu");
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 128);
  ASSERT_NE(wf, nullptr);

  constexpr uint64_t kScratchBase = 0x5'0000'0000ULL;
  constexpr uint32_t kSaddr = 66;
  constexpr uint32_t kScalarOffset = 0x180;
  constexpr uint32_t kLaneScratchSize = 0x200;
  wf->set_scratch_base(kScratchBase);
  wf->set_scratch_lane_size(kLaneScratchSize);
  wf->set_exec(0x3);
  cu->write_vgpr(wf->vgpr_alloc().base, 0, 0x4000);
  cu->write_vgpr(wf->vgpr_alloc().base, 1, 0x8000);
  cu->write_sgpr(wf->sgpr_alloc().base + kSaddr, kScalarOffset);

  gfx1250::VscratchMachineInst inst{};
  inst.saddr = kSaddr;
  inst.vaddr = 0;
  inst.sve = 0;
  inst.ioffset = 0xFFFFC0;

  amdgpu::VectorMemState d(amdgpu::GLOBAL_MEM);
  d.elem_size = 16;
  d.num_elems = 1;
  gfx1250::flat_calculate_addresses(inst, *wf, d);

  EXPECT_EQ(d.per_lane_addr[0], kScratchBase + kScalarOffset - 64);
  EXPECT_EQ(d.per_lane_addr[1], kScratchBase + kLaneScratchSize + kScalarOffset - 64);
}

TEST(Gfx1250AddrCalcTest, MubufPointerDescriptorAllowsZeroUpperDwords) {
  amdgpu::GpuMemory mem("gfx1250_mubuf_pointer_descriptor_mem");
  amdgpu::L2Cache l2("gfx1250_mubuf_pointer_descriptor_l2");
  auto cu = make_gfx1250_cu(mem, l2, "gfx1250_mubuf_pointer_descriptor_cu");
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 128);
  ASSERT_NE(wf, nullptr);

  constexpr uint64_t kBase = 0x6'0000'0000ULL;
  constexpr uint32_t kRsrc = 4;
  constexpr uint32_t kVaddr = 8;
  const uint32_t sbase = wf->sgpr_alloc().base;
  cu->write_sgpr(sbase + kRsrc, static_cast<uint32_t>(kBase));
  cu->write_sgpr(sbase + kRsrc + 1, static_cast<uint32_t>(kBase >> 32));
  cu->write_sgpr(sbase + kRsrc + 2, 0);
  cu->write_sgpr(sbase + kRsrc + 3, 0);

  const uint32_t vbase = wf->vgpr_alloc().base + kVaddr;
  cu->write_vgpr(vbase, 0, 0);
  cu->write_vgpr(vbase, 1, 4);
  cu->write_vgpr(vbase, 2, 8);
  cu->write_vgpr(vbase, 3, 12);
  wf->set_exec(0xF);

  gfx1250::VbufferMachineInst inst{};
  inst.rsrc = kRsrc;
  inst.soffset = gfx1250::OPR_SMEM_OFFSET_NULL;
  inst.offen = 1;
  inst.vaddr = kVaddr;

  amdgpu::VectorMemState d(amdgpu::GLOBAL_MEM);
  gfx1250::mubuf_calculate_addresses(inst, *wf, d);

  EXPECT_EQ(d.exec_mask, 0xFULL);
  EXPECT_EQ(d.lane_mask, 0xFULL);
  EXPECT_EQ(d.per_lane_addr[0], kBase);
  EXPECT_EQ(d.per_lane_addr[1], kBase + 4);
  EXPECT_EQ(d.per_lane_addr[2], kBase + 8);
  EXPECT_EQ(d.per_lane_addr[3], kBase + 12);
}

TEST(Gfx1250AddrCalcTest, MubufIdxenOffenUsesResolvedVaddrAndSoffset) {
  amdgpu::GpuMemory mem("gfx1250_mubuf_idxen_offen_mem");
  amdgpu::L2Cache l2("gfx1250_mubuf_idxen_offen_l2");
  auto cu = make_gfx1250_cu(mem, l2, "gfx1250_mubuf_idxen_offen_cu", /*vgprs_per_wf=*/512);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 512);
  ASSERT_NE(wf, nullptr);

  constexpr uint64_t kBase = 0x7'0000'0000ULL;
  constexpr uint32_t kRsrc = 8;
  constexpr uint32_t kSoffset = 20;
  constexpr uint32_t kVaddr = 12;
  constexpr uint32_t kVaddrBank = 1;
  constexpr uint32_t kStride = 16;
  const uint32_t sbase = wf->sgpr_alloc().base;
  cu->write_sgpr(sbase + kRsrc, static_cast<uint32_t>(kBase));
  cu->write_sgpr(sbase + kRsrc + 1, static_cast<uint32_t>(kBase >> 32) | (kStride << 16));
  cu->write_sgpr(sbase + kRsrc + 2, 0);
  cu->write_sgpr(sbase + kRsrc + 3, 0);
  cu->write_sgpr(sbase + kSoffset, 0x80);

  const uint32_t vbase = wf->vgpr_alloc().base + (kVaddrBank << 8) + kVaddr;
  cu->write_vgpr(vbase, 0, 1);
  cu->write_vgpr(vbase + 1, 0, 4);
  cu->write_vgpr(vbase, 1, 2);
  cu->write_vgpr(vbase + 1, 1, 4);
  wf->set_exec(0x3);
  wf->set_vgpr_msb_mode(kVaddrBank << 0);

  gfx1250::VbufferMachineInst inst{};
  inst.rsrc = kRsrc;
  inst.soffset = kSoffset;
  inst.idxen = 1;
  inst.offen = 1;
  inst.vaddr = kVaddr;

  amdgpu::VectorMemState d(amdgpu::GLOBAL_MEM);
  gfx1250::mubuf_calculate_addresses(inst, *wf, d);

  EXPECT_EQ(d.exec_mask, 0x3ULL);
  EXPECT_EQ(d.lane_mask, 0x3ULL);
  EXPECT_EQ(d.per_lane_addr[0], kBase + kStride + 4 + 0x80);
  EXPECT_EQ(d.per_lane_addr[1], kBase + 2 * kStride + 4 + 0x80);
}

TEST(Gfx1250MemoryExecutionTest, GlobalLoadRequestUsesResolvedAddressAndDestinationVgprs) {
  amdgpu::GpuMemory mem("gfx1250_global_high_bank_mem");
  amdgpu::L2Cache l2("gfx1250_global_high_bank_l2");
  auto cu = make_gfx1250_cu(mem, l2, "gfx1250_global_high_bank_cu", /*vgprs_per_wf=*/1024);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 1024);
  ASSERT_NE(wf, nullptr);

  constexpr uint32_t kAddrBank = 1;
  constexpr uint32_t kDstBank = 2;
  constexpr uint32_t kVaddr = 4;
  constexpr uint32_t kVdst = 16;
  constexpr uint64_t kAddr = 0x100000;

  const uint32_t vb = wf->vgpr_alloc().base;
  cu->write_vgpr(vb + (kAddrBank << 8) + kVaddr, 0, static_cast<uint32_t>(kAddr));
  cu->write_vgpr(vb + (kAddrBank << 8) + kVaddr + 1, 0, static_cast<uint32_t>(kAddr >> 32));
  wf->set_exec(1ULL);
  wf->set_vgpr_msb_mode((kAddrBank << 0) | (kDstBank << 6));

  gfx1250::VglobalMachineInst raw{};
  raw.saddr = gfx1250::OPR_SREG_NULL;
  raw.vaddr = kVaddr;
  raw.vdst = kVdst;
  auto inst = std::make_unique<gfx1250::GlobalLoadB32Vglobal>(
      reinterpret_cast<const gfx1250::MachineInst *>(&raw));
  inst->execute_impl(*wf);

  const auto *d = inst->data_as<amdgpu::VectorMemState>();
  ASSERT_NE(d, nullptr);
  EXPECT_TRUE(d->is_load);
  EXPECT_EQ(d->dst_reg_base, vb + (kDstBank << 8) + kVdst);
  EXPECT_EQ(d->per_lane_addr[0], kAddr);
}

TEST(Gfx1250MemoryExecutionTest, MonitorLoadsUseOrdinaryFlatLoadRequests) {
  expect_gfx1250_monitor_load_request<gfx1250::FlatLoadMonitorB32Vflat, gfx1250::VflatMachineInst>(
      1);
  expect_gfx1250_monitor_load_request<gfx1250::FlatLoadMonitorB64Vflat, gfx1250::VflatMachineInst>(
      2);
  expect_gfx1250_monitor_load_request<gfx1250::FlatLoadMonitorB128Vflat, gfx1250::VflatMachineInst>(
      4);
  expect_gfx1250_monitor_load_request<gfx1250::GlobalLoadMonitorB32Vglobal,
                                      gfx1250::VglobalMachineInst>(1);
  expect_gfx1250_monitor_load_request<gfx1250::GlobalLoadMonitorB64Vglobal,
                                      gfx1250::VglobalMachineInst>(2);
  expect_gfx1250_monitor_load_request<gfx1250::GlobalLoadMonitorB128Vglobal,
                                      gfx1250::VglobalMachineInst>(4);
}

TEST(Gfx1250MemoryExecutionTest, GlobalLoadMonitorB128CompletesThroughPipeline) {
  amdgpu::GpuMemory mem("gfx1250_global_monitor_load_mem");
  amdgpu::L2Cache l2("gfx1250_global_monitor_load_l2");
  l2.set_backing_memory(&mem);
  auto cu = make_gfx1250_cu(mem, l2, "gfx1250_global_monitor_load_cu");
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 128);
  ASSERT_NE(wf, nullptr);

  constexpr uint32_t kVaddr = 4;
  constexpr uint32_t kVdst = 8;
  constexpr uint64_t kAddr = 0x100000;
  constexpr std::array<uint32_t, 4> kValues{0x12345678u, 0x90abcdefu, 0x0badc0deu, 0xf00d1250u};

  for (uint32_t i = 0; i < kValues.size(); ++i)
    write_global_u32(mem, kAddr + i * sizeof(uint32_t), kValues[i]);

  const uint32_t vb = wf->vgpr_alloc().base;
  cu->write_vgpr(vb + kVaddr, 0, static_cast<uint32_t>(kAddr));
  cu->write_vgpr(vb + kVaddr + 1, 0, static_cast<uint32_t>(kAddr >> 32));
  wf->set_exec(1ULL);

  gfx1250::VglobalMachineInst raw{};
  raw.saddr = gfx1250::OPR_SREG_NULL;
  raw.vaddr = kVaddr;
  raw.vdst = kVdst;
  auto inst = std::make_unique<gfx1250::GlobalLoadMonitorB128Vglobal>(
      reinterpret_cast<const gfx1250::MachineInst *>(&raw));
  inst->execute_impl(*wf);

  amdgpu::GlobalMemPipeline pipeline(&cu->l1_vector(), &l2);
  pipeline.issue(inst.release(), *wf);

  for (uint32_t i = 0; i < kValues.size(); ++i)
    EXPECT_EQ(cu->read_vgpr(vb + kVdst + i, 0), kValues[i]);
  EXPECT_TRUE(wf->wait_counters().empty());
}

TEST(Gfx1250MemoryExecutionTest, FlatLoadMonitorB64CompletesThroughPipeline) {
  amdgpu::GpuMemory mem("gfx1250_flat_monitor_load_mem");
  amdgpu::L2Cache l2("gfx1250_flat_monitor_load_l2");
  l2.set_backing_memory(&mem);
  auto cu = make_gfx1250_cu(mem, l2, "gfx1250_flat_monitor_load_cu");
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 128);
  ASSERT_NE(wf, nullptr);

  constexpr uint32_t kVaddr = 4;
  constexpr uint32_t kVdst = 8;
  constexpr uint64_t kAddr = 0x110000;
  constexpr std::array<uint32_t, 2> kValues{0x55aa1250u, 0xcc33aa77u};

  for (uint32_t i = 0; i < kValues.size(); ++i)
    write_global_u32(mem, kAddr + i * sizeof(uint32_t), kValues[i]);

  const uint32_t vb = wf->vgpr_alloc().base;
  cu->write_vgpr(vb + kVaddr, 0, static_cast<uint32_t>(kAddr));
  cu->write_vgpr(vb + kVaddr + 1, 0, static_cast<uint32_t>(kAddr >> 32));
  wf->set_exec(1ULL);

  gfx1250::VflatMachineInst raw{};
  raw.saddr = gfx1250::OPR_SREG_NULL;
  raw.vaddr = kVaddr;
  raw.vdst = kVdst;
  auto inst = std::make_unique<gfx1250::FlatLoadMonitorB64Vflat>(
      reinterpret_cast<const gfx1250::MachineInst *>(&raw));
  inst->execute_impl(*wf);

  amdgpu::GlobalMemPipeline pipeline(&cu->l1_vector(), &l2);
  pipeline.issue(inst.release(), *wf);

  for (uint32_t i = 0; i < kValues.size(); ++i)
    EXPECT_EQ(cu->read_vgpr(vb + kVdst + i, 0), kValues[i]);
  EXPECT_TRUE(wf->wait_counters().empty());
}

TEST(Gfx1250MemoryExecutionTest, GlobalAddtidB32UsesLaneIdOffsetForLoadAndStore) {
  amdgpu::GpuMemory mem("gfx1250_global_addtid_mem");
  amdgpu::L2Cache l2("gfx1250_global_addtid_l2");
  l2.set_backing_memory(&mem);
  auto cu = make_gfx1250_cu(mem, l2, "gfx1250_global_addtid_cu", /*vgprs_per_wf=*/512);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 512);
  ASSERT_NE(wf, nullptr);

  constexpr uint32_t kSaddr = 4;
  constexpr uint32_t kLoadDstBank = 1;
  constexpr uint32_t kStoreSrcBank = 2;
  constexpr uint32_t kLoadDst = 8;
  constexpr uint32_t kStoreSrc = 12;
  constexpr uint64_t kBase = 0x120000;
  constexpr uint32_t kLoadOffset = 16;
  constexpr uint32_t kStoreOffset = 64;
  constexpr std::array<uint32_t, 3> kLoadValues{0x11112222u, 0x33334444u, 0x55556666u};
  constexpr std::array<uint32_t, 3> kStoreValues{0x77778888u, 0x9999aaaau, 0xbbbbccccu};

  const uint32_t sb = wf->sgpr_alloc().base;
  cu->write_sgpr(sb + kSaddr, static_cast<uint32_t>(kBase));
  cu->write_sgpr(sb + kSaddr + 1, static_cast<uint32_t>(kBase >> 32));

  for (uint32_t lane = 0; lane < kLoadValues.size(); ++lane)
    write_global_u32(mem, kBase + kLoadOffset + lane * sizeof(uint32_t), kLoadValues[lane]);

  const uint32_t vb = wf->vgpr_alloc().base;
  wf->set_exec(0x7ULL);
  wf->set_vgpr_msb_mode(kLoadDstBank << 6);

  gfx1250::VglobalMachineInst load_raw{};
  load_raw.saddr = kSaddr;
  load_raw.vdst = kLoadDst;
  load_raw.ioffset = kLoadOffset;
  auto load = std::make_unique<gfx1250::GlobalLoadAddtidB32Vglobal>(
      reinterpret_cast<const gfx1250::MachineInst *>(&load_raw));
  load->execute_impl(*wf);

  const auto *load_d = load->data_as<amdgpu::VectorMemState>();
  ASSERT_NE(load_d, nullptr);
  EXPECT_TRUE(load_d->is_load);
  EXPECT_EQ(load_d->wait_counter_type, amdgpu::WaitCounterType::LOADCNT);
  EXPECT_EQ(load_d->dst_reg_base, vb + (kLoadDstBank << 8) + kLoadDst);
  for (uint32_t lane = 0; lane < kLoadValues.size(); ++lane)
    EXPECT_EQ(load_d->per_lane_addr[lane], kBase + kLoadOffset + lane * sizeof(uint32_t));

  amdgpu::GlobalMemPipeline pipeline(&cu->l1_vector(), &l2);
  pipeline.issue(load.release(), *wf);
  for (uint32_t lane = 0; lane < kLoadValues.size(); ++lane)
    EXPECT_EQ(cu->read_vgpr(vb + (kLoadDstBank << 8) + kLoadDst, lane), kLoadValues[lane]);

  wf->set_vgpr_msb_mode(kStoreSrcBank << 0);
  for (uint32_t lane = 0; lane < kStoreValues.size(); ++lane)
    cu->write_vgpr(vb + (kStoreSrcBank << 8) + kStoreSrc, lane, kStoreValues[lane]);

  gfx1250::VglobalMachineInst store_raw{};
  store_raw.saddr = kSaddr;
  store_raw.vsrc = kStoreSrc;
  store_raw.ioffset = kStoreOffset;
  auto store = std::make_unique<gfx1250::GlobalStoreAddtidB32Vglobal>(
      reinterpret_cast<const gfx1250::MachineInst *>(&store_raw));
  store->execute_impl(*wf);

  const auto *store_d = store->data_as<amdgpu::VectorMemState>();
  ASSERT_NE(store_d, nullptr);
  EXPECT_FALSE(store_d->is_load);
  EXPECT_EQ(store_d->wait_counter_type, amdgpu::WaitCounterType::STORECNT);
  for (uint32_t lane = 0; lane < kStoreValues.size(); ++lane) {
    EXPECT_EQ(store_d->per_lane_addr[lane], kBase + kStoreOffset + lane * sizeof(uint32_t));
    uint32_t packed = 0;
    std::memcpy(&packed, &store_d->store_data[lane * sizeof(uint32_t)], sizeof(uint32_t));
    EXPECT_EQ(packed, kStoreValues[lane]);
  }

  pipeline.issue(store.release(), *wf);
  for (uint32_t lane = 0; lane < kStoreValues.size(); ++lane)
    EXPECT_EQ(read_global_u32(mem, kBase + kStoreOffset + lane * sizeof(uint32_t)),
              kStoreValues[lane]);
  EXPECT_TRUE(wf->wait_counters().empty());
}

TEST(Gfx1250MemoryExecutionTest, BlockGlobalAndScratchMoveThirtyTwoDwordsPerLane) {
  amdgpu::GpuMemory mem("gfx1250_block_memory_mem");
  amdgpu::L2Cache l2("gfx1250_block_memory_l2");
  l2.set_backing_memory(&mem);
  auto cu = make_gfx1250_cu(mem, l2, "gfx1250_block_memory_cu", /*vgprs_per_wf=*/192);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 192);
  ASSERT_NE(wf, nullptr);

  constexpr uint32_t kBlockDwords = 32;
  constexpr uint32_t kBlockBytes = kBlockDwords * sizeof(uint32_t);
  constexpr uint32_t kActiveLanes = 2;
  constexpr uint32_t kGlobalSaddr = 4;
  constexpr uint32_t kGlobalVaddr = 0;
  constexpr uint32_t kGlobalLoadDst = 8;
  constexpr uint32_t kGlobalStoreSrc = 48;
  constexpr uint64_t kGlobalBase = 0x140000;
  constexpr uint32_t kGlobalLaneStride = 0x200;
  constexpr uint32_t kGlobalLoadOffset = 0x40;
  constexpr uint32_t kGlobalStoreOffset = 0x140;

  const uint32_t sb = wf->sgpr_alloc().base;
  const uint32_t vb = wf->vgpr_alloc().base;
  cu->write_sgpr(sb + kGlobalSaddr, static_cast<uint32_t>(kGlobalBase));
  cu->write_sgpr(sb + kGlobalSaddr + 1, static_cast<uint32_t>(kGlobalBase >> 32));
  wf->set_exec((1ULL << kActiveLanes) - 1);

  auto value = [](uint32_t tag, uint32_t lane, uint32_t word) { return tag | (lane << 8) | word; };

  for (uint32_t lane = 0; lane < kActiveLanes; ++lane) {
    cu->write_vgpr(vb + kGlobalVaddr, lane, lane * kGlobalLaneStride);
    for (uint32_t word = 0; word < kBlockDwords; ++word) {
      write_global_u32(
          mem, kGlobalBase + kGlobalLoadOffset + lane * kGlobalLaneStride + word * sizeof(uint32_t),
          value(0xa0000000u, lane, word));
      cu->write_vgpr(vb + kGlobalStoreSrc + word, lane, value(0xb0000000u, lane, word));
    }
  }

  amdgpu::GlobalMemPipeline pipeline(&cu->l1_vector(), &l2);

  gfx1250::VglobalMachineInst global_load_raw{};
  global_load_raw.saddr = kGlobalSaddr;
  global_load_raw.vaddr = kGlobalVaddr;
  global_load_raw.vdst = kGlobalLoadDst;
  global_load_raw.ioffset = kGlobalLoadOffset;
  auto global_load = std::make_unique<gfx1250::GlobalLoadBlockVglobal>(
      reinterpret_cast<const gfx1250::MachineInst *>(&global_load_raw));
  global_load->execute_impl(*wf);

  const auto *global_load_d = global_load->data_as<amdgpu::VectorMemState>();
  ASSERT_NE(global_load_d, nullptr);
  EXPECT_TRUE(global_load_d->is_load);
  EXPECT_EQ(global_load_d->elem_size, 4u);
  EXPECT_EQ(global_load_d->num_elems, kBlockDwords);
  EXPECT_EQ(global_load_d->wait_counter_type, amdgpu::WaitCounterType::LOADCNT);
  EXPECT_EQ(global_load_d->dst_reg_base, vb + kGlobalLoadDst);
  for (uint32_t lane = 0; lane < kActiveLanes; ++lane)
    EXPECT_EQ(global_load_d->per_lane_addr[lane],
              kGlobalBase + kGlobalLoadOffset + lane * kGlobalLaneStride);

  pipeline.issue(global_load.release(), *wf);
  for (uint32_t lane = 0; lane < kActiveLanes; ++lane) {
    for (uint32_t word = 0; word < kBlockDwords; ++word)
      EXPECT_EQ(cu->read_vgpr(vb + kGlobalLoadDst + word, lane), value(0xa0000000u, lane, word));
  }

  gfx1250::VglobalMachineInst global_store_raw{};
  global_store_raw.saddr = kGlobalSaddr;
  global_store_raw.vaddr = kGlobalVaddr;
  global_store_raw.vsrc = kGlobalStoreSrc;
  global_store_raw.ioffset = kGlobalStoreOffset;
  auto global_store = std::make_unique<gfx1250::GlobalStoreBlockVglobal>(
      reinterpret_cast<const gfx1250::MachineInst *>(&global_store_raw));
  global_store->execute_impl(*wf);

  const auto *global_store_d = global_store->data_as<amdgpu::VectorMemState>();
  ASSERT_NE(global_store_d, nullptr);
  EXPECT_FALSE(global_store_d->is_load);
  EXPECT_EQ(global_store_d->elem_size, 4u);
  EXPECT_EQ(global_store_d->num_elems, kBlockDwords);
  EXPECT_EQ(global_store_d->wait_counter_type, amdgpu::WaitCounterType::STORECNT);
  for (uint32_t lane = 0; lane < kActiveLanes; ++lane) {
    EXPECT_EQ(global_store_d->per_lane_addr[lane],
              kGlobalBase + kGlobalStoreOffset + lane * kGlobalLaneStride);
    for (uint32_t word = 0; word < kBlockDwords; ++word) {
      uint32_t packed = 0;
      std::memcpy(&packed,
                  &global_store_d->store_data[lane * kBlockBytes + word * sizeof(uint32_t)],
                  sizeof(uint32_t));
      EXPECT_EQ(packed, value(0xb0000000u, lane, word));
    }
  }

  pipeline.issue(global_store.release(), *wf);
  for (uint32_t lane = 0; lane < kActiveLanes; ++lane) {
    for (uint32_t word = 0; word < kBlockDwords; ++word)
      EXPECT_EQ(read_global_u32(mem, kGlobalBase + kGlobalStoreOffset + lane * kGlobalLaneStride +
                                         word * sizeof(uint32_t)),
                value(0xb0000000u, lane, word));
  }

  constexpr uint64_t kScratchBase = 0x240000;
  constexpr uint32_t kScratchLaneSize = 0x400;
  constexpr uint32_t kScratchSaddr = 10;
  constexpr uint32_t kScratchScalarOffset = 0x20;
  constexpr uint32_t kScratchVaddr = 2;
  constexpr uint32_t kScratchLoadDst = 80;
  constexpr uint32_t kScratchStoreSrc = 144;
  constexpr uint32_t kScratchLoadOffset = 0x10;
  constexpr uint32_t kScratchStoreOffset = 0x90;
  wf->set_scratch_base(kScratchBase);
  wf->set_scratch_lane_size(kScratchLaneSize);
  cu->write_sgpr(sb + kScratchSaddr, kScratchScalarOffset);

  for (uint32_t lane = 0; lane < kActiveLanes; ++lane) {
    const uint32_t lane_vaddr = lane * 0x20;
    cu->write_vgpr(vb + kScratchVaddr, lane, lane_vaddr);
    for (uint32_t word = 0; word < kBlockDwords; ++word) {
      write_global_u32(mem,
                       kScratchBase + lane * kScratchLaneSize + lane_vaddr + kScratchScalarOffset +
                           kScratchLoadOffset + word * sizeof(uint32_t),
                       value(0xc0000000u, lane, word));
      cu->write_vgpr(vb + kScratchStoreSrc + word, lane, value(0xd0000000u, lane, word));
    }
  }

  gfx1250::VscratchMachineInst scratch_load_raw{};
  scratch_load_raw.saddr = kScratchSaddr;
  scratch_load_raw.sve = 1;
  scratch_load_raw.vaddr = kScratchVaddr;
  scratch_load_raw.vdst = kScratchLoadDst;
  scratch_load_raw.ioffset = kScratchLoadOffset;
  auto scratch_load = std::make_unique<gfx1250::ScratchLoadBlockVscratch>(
      reinterpret_cast<const gfx1250::MachineInst *>(&scratch_load_raw));
  scratch_load->execute_impl(*wf);

  const auto *scratch_load_d = scratch_load->data_as<amdgpu::VectorMemState>();
  ASSERT_NE(scratch_load_d, nullptr);
  EXPECT_TRUE(scratch_load_d->is_load);
  EXPECT_EQ(scratch_load_d->elem_size, 4u);
  EXPECT_EQ(scratch_load_d->num_elems, kBlockDwords);
  EXPECT_EQ(scratch_load_d->wait_counter_type, amdgpu::WaitCounterType::LOADCNT);
  EXPECT_EQ(scratch_load_d->dst_reg_base, vb + kScratchLoadDst);
  for (uint32_t lane = 0; lane < kActiveLanes; ++lane)
    EXPECT_EQ(scratch_load_d->per_lane_addr[lane], kScratchBase + lane * kScratchLaneSize +
                                                       lane * 0x20 + kScratchScalarOffset +
                                                       kScratchLoadOffset);

  pipeline.issue(scratch_load.release(), *wf);
  for (uint32_t lane = 0; lane < kActiveLanes; ++lane) {
    for (uint32_t word = 0; word < kBlockDwords; ++word)
      EXPECT_EQ(cu->read_vgpr(vb + kScratchLoadDst + word, lane), value(0xc0000000u, lane, word));
  }

  gfx1250::VscratchMachineInst scratch_store_raw{};
  scratch_store_raw.saddr = kScratchSaddr;
  scratch_store_raw.sve = 1;
  scratch_store_raw.vaddr = kScratchVaddr;
  scratch_store_raw.vsrc = kScratchStoreSrc;
  scratch_store_raw.ioffset = kScratchStoreOffset;
  auto scratch_store = std::make_unique<gfx1250::ScratchStoreBlockVscratch>(
      reinterpret_cast<const gfx1250::MachineInst *>(&scratch_store_raw));
  scratch_store->execute_impl(*wf);

  const auto *scratch_store_d = scratch_store->data_as<amdgpu::VectorMemState>();
  ASSERT_NE(scratch_store_d, nullptr);
  EXPECT_FALSE(scratch_store_d->is_load);
  EXPECT_EQ(scratch_store_d->elem_size, 4u);
  EXPECT_EQ(scratch_store_d->num_elems, kBlockDwords);
  EXPECT_EQ(scratch_store_d->wait_counter_type, amdgpu::WaitCounterType::STORECNT);
  for (uint32_t lane = 0; lane < kActiveLanes; ++lane) {
    EXPECT_EQ(scratch_store_d->per_lane_addr[lane], kScratchBase + lane * kScratchLaneSize +
                                                        lane * 0x20 + kScratchScalarOffset +
                                                        kScratchStoreOffset);
    for (uint32_t word = 0; word < kBlockDwords; ++word) {
      uint32_t packed = 0;
      std::memcpy(&packed,
                  &scratch_store_d->store_data[lane * kBlockBytes + word * sizeof(uint32_t)],
                  sizeof(uint32_t));
      EXPECT_EQ(packed, value(0xd0000000u, lane, word));
    }
  }

  pipeline.issue(scratch_store.release(), *wf);
  for (uint32_t lane = 0; lane < kActiveLanes; ++lane) {
    for (uint32_t word = 0; word < kBlockDwords; ++word)
      EXPECT_EQ(read_global_u32(mem, kScratchBase + lane * kScratchLaneSize + lane * 0x20 +
                                         kScratchScalarOffset + kScratchStoreOffset +
                                         word * sizeof(uint32_t)),
                value(0xd0000000u, lane, word));
  }
  EXPECT_TRUE(wf->wait_counters().empty());
}

TEST(Gfx1250MemoryExecutionTest, ClusterLoadB32CompletesThroughGlobalMemoryPipeline) {
  amdgpu::GpuMemory mem("gfx1250_cluster_load_mem");
  amdgpu::L2Cache l2("gfx1250_cluster_load_l2");
  l2.set_backing_memory(&mem);
  auto cu = make_gfx1250_cu(mem, l2, "gfx1250_cluster_load_cu");
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 128);
  ASSERT_NE(wf, nullptr);

  constexpr uint32_t kVaddr = 4;
  constexpr uint32_t kVdst = 8;
  constexpr uint64_t kAddr0 = 0x100000;
  constexpr uint64_t kAddr1 = 0x100100;
  constexpr uint32_t kValue0 = 0x12345678;
  constexpr uint32_t kValue1 = 0x90abcdef;

  write_global_u32(mem, kAddr0, kValue0);
  write_global_u32(mem, kAddr1, kValue1);

  const uint32_t vb = wf->vgpr_alloc().base;
  cu->write_vgpr(vb + kVaddr, 0, static_cast<uint32_t>(kAddr0));
  cu->write_vgpr(vb + kVaddr + 1, 0, static_cast<uint32_t>(kAddr0 >> 32));
  cu->write_vgpr(vb + kVaddr, 1, static_cast<uint32_t>(kAddr1));
  cu->write_vgpr(vb + kVaddr + 1, 1, static_cast<uint32_t>(kAddr1 >> 32));
  wf->set_exec(0x3ULL);

  gfx1250::VglobalMachineInst raw{};
  raw.saddr = gfx1250::OPR_SREG_NULL;
  raw.vaddr = kVaddr;
  raw.vdst = kVdst;
  auto inst = std::make_unique<gfx1250::ClusterLoadB32Vglobal>(
      reinterpret_cast<const gfx1250::MachineInst *>(&raw));
  inst->execute_impl(*wf);

  const auto *d = inst->data_as<amdgpu::VectorMemState>();
  ASSERT_NE(d, nullptr);
  EXPECT_TRUE(d->is_load);
  EXPECT_EQ(d->wait_counter_type, amdgpu::WaitCounterType::LOADCNT);
  EXPECT_EQ(d->dst_reg_base, vb + kVdst);
  EXPECT_EQ(d->per_lane_addr[0], kAddr0);
  EXPECT_EQ(d->per_lane_addr[1], kAddr1);

  amdgpu::GlobalMemPipeline pipeline(&cu->l1_vector(), &l2);
  pipeline.issue(inst.release(), *wf);

  EXPECT_EQ(cu->read_vgpr(vb + kVdst, 0), kValue0);
  EXPECT_EQ(cu->read_vgpr(vb + kVdst, 1), kValue1);
  EXPECT_TRUE(wf->wait_counters().empty());
}

template <typename InstT> void run_gfx1250_async_to_lds_b32_test(const char *cu_name) {
  amdgpu::GpuMemory mem(cu_name);
  amdgpu::L2Cache l2(std::string(cu_name) + "_l2");
  l2.set_backing_memory(&mem);
  auto cu = make_gfx1250_cu(mem, l2, cu_name);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 128);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(0x100);

  constexpr uint32_t kLdsOffsetReg = 6;
  constexpr uint32_t kVaddr = 10;
  constexpr uint64_t kAddr0 = 0x110000;
  constexpr uint64_t kAddr1 = 0x110100;
  constexpr uint32_t kLdsOff0 = 0x20;
  constexpr uint32_t kLdsOff1 = 0x44;
  constexpr uint32_t kValue0 = 0x0badc0de;
  constexpr uint32_t kValue1 = 0xfeedface;

  write_global_u32(mem, kAddr0, kValue0);
  write_global_u32(mem, kAddr1, kValue1);

  const uint32_t vb = wf->vgpr_alloc().base;
  cu->write_vgpr(vb + kLdsOffsetReg, 0, kLdsOff0);
  cu->write_vgpr(vb + kLdsOffsetReg, 1, kLdsOff1);
  cu->write_vgpr(vb + kVaddr, 0, static_cast<uint32_t>(kAddr0));
  cu->write_vgpr(vb + kVaddr + 1, 0, static_cast<uint32_t>(kAddr0 >> 32));
  cu->write_vgpr(vb + kVaddr, 1, static_cast<uint32_t>(kAddr1));
  cu->write_vgpr(vb + kVaddr + 1, 1, static_cast<uint32_t>(kAddr1 >> 32));
  wf->set_exec(0x3ULL);

  gfx1250::VglobalMachineInst raw{};
  raw.saddr = gfx1250::OPR_SREG_NULL;
  raw.vdst = kLdsOffsetReg;
  raw.vaddr = kVaddr;
  auto inst = std::make_unique<InstT>(reinterpret_cast<const gfx1250::MachineInst *>(&raw));
  EXPECT_TRUE(inst->is_memory_op());
  inst->execute_impl(*wf);

  const auto *d = inst->template data_as<amdgpu::VectorMemState>();
  ASSERT_NE(d, nullptr);
  EXPECT_TRUE(d->is_load);
  EXPECT_TRUE(d->lds_dst);
  EXPECT_TRUE(d->lds_per_lane_addr);
  EXPECT_EQ(d->wait_counter_type, amdgpu::WaitCounterType::ASYNCCNT);
  EXPECT_EQ(d->per_lane_lds_addr[0], wf->lds_base() + kLdsOff0);
  EXPECT_EQ(d->per_lane_lds_addr[1], wf->lds_base() + kLdsOff1);

  amdgpu::GlobalMemPipeline pipeline(&cu->l1_vector(), &l2);
  pipeline.issue(inst.release(), *wf);

  EXPECT_EQ(cu->lds().read32(wf->lds_base() + kLdsOff0), kValue0);
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + kLdsOff1), kValue1);
  EXPECT_TRUE(wf->wait_counters().empty());
}

TEST(Gfx1250MemoryExecutionTest, GlobalLoadAsyncToLdsUsesPerLaneLdsOffsets) {
  run_gfx1250_async_to_lds_b32_test<gfx1250::GlobalLoadAsyncToLdsB32Vglobal>(
      "gfx1250_global_async_to_lds_cu");
}

TEST(Gfx1250MemoryExecutionTest, ClusterLoadAsyncToLdsUsesPerLaneLdsOffsets) {
  run_gfx1250_async_to_lds_b32_test<gfx1250::ClusterLoadAsyncToLdsB32Vglobal>(
      "gfx1250_cluster_async_to_lds_cu");
}

TEST(Gfx1250MemoryExecutionTest, GlobalStoreAsyncFromLdsUsesPerLaneLdsOffsets) {
  amdgpu::GpuMemory mem("gfx1250_async_from_lds_mem");
  amdgpu::L2Cache l2("gfx1250_async_from_lds_l2");
  l2.set_backing_memory(&mem);
  auto cu = make_gfx1250_cu(mem, l2, "gfx1250_async_from_lds_cu");
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 128);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(0x200);

  constexpr uint32_t kVaddr = 4;
  constexpr uint32_t kLdsOffsetReg = 12;
  constexpr uint64_t kAddr0 = 0x120000;
  constexpr uint64_t kAddr1 = 0x120100;
  constexpr uint32_t kLdsOff0 = 0x30;
  constexpr uint32_t kLdsOff1 = 0x58;
  constexpr uint32_t kValue0 = 0xa5a50001;
  constexpr uint32_t kValue1 = 0x5a5a0002;

  const uint32_t vb = wf->vgpr_alloc().base;
  cu->write_vgpr(vb + kVaddr, 0, static_cast<uint32_t>(kAddr0));
  cu->write_vgpr(vb + kVaddr + 1, 0, static_cast<uint32_t>(kAddr0 >> 32));
  cu->write_vgpr(vb + kVaddr, 1, static_cast<uint32_t>(kAddr1));
  cu->write_vgpr(vb + kVaddr + 1, 1, static_cast<uint32_t>(kAddr1 >> 32));
  cu->write_vgpr(vb + kLdsOffsetReg, 0, kLdsOff0);
  cu->write_vgpr(vb + kLdsOffsetReg, 1, kLdsOff1);
  cu->lds().write32(wf->lds_base() + kLdsOff0, kValue0);
  cu->lds().write32(wf->lds_base() + kLdsOff1, kValue1);
  wf->set_exec(0x3ULL);

  gfx1250::VglobalMachineInst raw{};
  raw.saddr = gfx1250::OPR_SREG_NULL;
  raw.vaddr = kVaddr;
  raw.vsrc = kLdsOffsetReg;
  auto inst = std::make_unique<gfx1250::GlobalStoreAsyncFromLdsB32Vglobal>(
      reinterpret_cast<const gfx1250::MachineInst *>(&raw));
  EXPECT_TRUE(inst->is_memory_op());
  inst->execute_impl(*wf);

  const auto *d = inst->data_as<amdgpu::VectorMemState>();
  ASSERT_NE(d, nullptr);
  EXPECT_FALSE(d->is_load);
  EXPECT_EQ(d->wait_counter_type, amdgpu::WaitCounterType::ASYNCCNT);
  ASSERT_GE(d->store_data.size(), 2u * sizeof(uint32_t));
  uint32_t stored0 = 0;
  uint32_t stored1 = 0;
  std::memcpy(&stored0, &d->store_data[0], sizeof(stored0));
  std::memcpy(&stored1, &d->store_data[sizeof(uint32_t)], sizeof(stored1));
  EXPECT_EQ(stored0, kValue0);
  EXPECT_EQ(stored1, kValue1);

  amdgpu::GlobalMemPipeline pipeline(&cu->l1_vector(), &l2);
  pipeline.issue(inst.release(), *wf);

  EXPECT_EQ(read_global_u32(mem, kAddr0), kValue0);
  EXPECT_EQ(read_global_u32(mem, kAddr1), kValue1);
  EXPECT_TRUE(wf->wait_counters().empty());
}

TEST(Gfx1250MemoryExecutionTest, NarrowSmemLoadsSignAndZeroExtend) {
  amdgpu::GpuMemory mem("gfx1250_narrow_smem_mem");
  amdgpu::L2Cache l2("gfx1250_narrow_smem_l2");
  l2.set_backing_memory(&mem);
  auto cu = make_gfx1250_cu(mem, l2, "gfx1250_narrow_smem_cu");
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 128);
  ASSERT_NE(wf, nullptr);

  constexpr uint64_t kBase = 0x100000;
  mem.write8(kBase + 1, 0x80);
  mem.write8(kBase + 2, 0x7f);
  mem.write8(kBase + 4, 0x00);
  mem.write8(kBase + 5, 0x80);
  mem.write8(kBase + 6, 0xff);
  mem.write8(kBase + 7, 0x7f);

  const uint32_t sbase = wf->sgpr_alloc().base;
  cu->write_sgpr(sbase + 4, static_cast<uint32_t>(kBase));
  cu->write_sgpr(sbase + 5, static_cast<uint32_t>(kBase >> 32));

  run_gfx1250_smem_load<gfx1250::SLoadI8Smem>(*cu, *wf, 10, 1);
  run_gfx1250_smem_load<gfx1250::SLoadU8Smem>(*cu, *wf, 11, 2);
  run_gfx1250_smem_load<gfx1250::SLoadI16Smem>(*cu, *wf, 12, 4);
  run_gfx1250_smem_load<gfx1250::SLoadU16Smem>(*cu, *wf, 13, 6);

  EXPECT_EQ(cu->read_sgpr(sbase + 10), 0xffff'ff80u);
  EXPECT_EQ(cu->read_sgpr(sbase + 11), 0x0000'007fu);
  EXPECT_EQ(cu->read_sgpr(sbase + 12), 0xffff'8000u);
  EXPECT_EQ(cu->read_sgpr(sbase + 13), 0x0000'7fffu);
}

TEST(Gfx1250MemoryExecutionTest, DsRequestsUseResolvedVgprs) {
  amdgpu::GpuMemory mem("gfx1250_ds_high_bank_mem");
  amdgpu::L2Cache l2("gfx1250_ds_high_bank_l2");
  auto cu = make_gfx1250_cu(mem, l2, "gfx1250_ds_high_bank_cu", /*vgprs_per_wf=*/1024);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 1024);
  ASSERT_NE(wf, nullptr);

  constexpr uint32_t kAddrBank = 1;
  constexpr uint32_t kDataBank = 2;
  constexpr uint32_t kDstBank = 3;
  constexpr uint32_t kAddrReg = 4;
  constexpr uint32_t kDataReg = 8;
  constexpr uint32_t kDstReg = 12;
  constexpr uint32_t kLdsAddr = 0x140;
  constexpr uint32_t kValue = 0x89ABCDEF;

  const uint32_t vb = wf->vgpr_alloc().base;
  cu->write_vgpr(vb + (kAddrBank << 8) + kAddrReg, 0, kLdsAddr);
  cu->write_vgpr(vb + (kDataBank << 8) + kDataReg, 0, kValue);
  wf->set_exec(1ULL);

  gfx1250::VdsMachineInst store_raw{};
  store_raw.addr = kAddrReg;
  store_raw.data0 = kDataReg;
  wf->set_vgpr_msb_mode((kAddrBank << 0) | (kDataBank << 2));
  auto store = std::make_unique<gfx1250::DsStoreB32Vds>(
      reinterpret_cast<const gfx1250::MachineInst *>(&store_raw));
  store->execute_impl(*wf);
  const auto *store_d = store->data_as<amdgpu::VectorMemState>();
  ASSERT_NE(store_d, nullptr);
  EXPECT_FALSE(store_d->is_load);
  EXPECT_EQ(store_d->per_lane_addr[0], kLdsAddr);
  ASSERT_GE(store_d->store_data.size(), sizeof(kValue));
  uint32_t stored = 0;
  std::memcpy(&stored, store_d->store_data.data(), sizeof(stored));
  EXPECT_EQ(stored, kValue);

  gfx1250::VdsMachineInst load_raw{};
  load_raw.addr = kAddrReg;
  load_raw.vdst = kDstReg;
  wf->set_vgpr_msb_mode((kAddrBank << 0) | (kDstBank << 6));
  auto load = std::make_unique<gfx1250::DsLoadB32Vds>(
      reinterpret_cast<const gfx1250::MachineInst *>(&load_raw));
  load->execute_impl(*wf);

  const auto *load_d = load->data_as<amdgpu::VectorMemState>();
  ASSERT_NE(load_d, nullptr);
  EXPECT_TRUE(load_d->is_load);
  EXPECT_EQ(load_d->dst_reg_base, vb + (kDstBank << 8) + kDstReg);
  EXPECT_EQ(load_d->per_lane_addr[0], kLdsAddr);
}

TEST(Gfx1250MemoryExecutionTest, DsMskorB32UpdatesLdsWithoutReturn) {
  amdgpu::GpuMemory mem("gfx1250_ds_mskor_b32_mem");
  amdgpu::L2Cache l2("gfx1250_ds_mskor_b32_l2");
  auto cu = make_gfx1250_cu(mem, l2, "gfx1250_ds_mskor_b32_cu");
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 128);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(0x100);

  constexpr uint32_t kAddrReg = 4;
  constexpr uint32_t kMaskReg = 8;
  constexpr uint32_t kSrcReg = 12;
  constexpr uint32_t kLdsOff = 0x40;
  constexpr uint32_t kOld = 0xaaaa'5555u;
  constexpr uint32_t kMask = 0x00ff'00ffu;
  constexpr uint32_t kSrc = 0x0012'0034u;
  constexpr uint32_t kExpected = (kOld & ~kMask) | kSrc;

  const uint32_t vb = wf->vgpr_alloc().base;
  cu->write_vgpr(vb + kAddrReg, 0, kLdsOff);
  cu->write_vgpr(vb + kMaskReg, 0, kMask);
  cu->write_vgpr(vb + kSrcReg, 0, kSrc);
  cu->lds().write32(wf->lds_base() + kLdsOff, kOld);
  wf->set_exec(1ULL);

  gfx1250::VdsMachineInst raw{};
  raw.addr = kAddrReg;
  raw.data0 = kMaskReg;
  raw.data1 = kSrcReg;
  auto inst = std::make_unique<gfx1250::DsMskorB32Vds>(
      reinterpret_cast<const gfx1250::MachineInst *>(&raw));
  EXPECT_TRUE(inst->is_memory_op());
  inst->execute_impl(*wf);

  const auto *d = inst->data_as<amdgpu::VectorMemState>();
  ASSERT_NE(d, nullptr);
  EXPECT_FALSE(d->is_load);
  EXPECT_EQ(d->atomic_op, amdgpu::AtomicOp::MSKOR);
  EXPECT_EQ(d->wait_counter_type, amdgpu::WaitCounterType::DSCNT);
  EXPECT_EQ(d->per_lane_addr[0], wf->lds_base() + kLdsOff);
  ASSERT_GE(d->store_data.size(), 2u * sizeof(uint32_t));
  uint32_t recorded_mask = 0;
  uint32_t recorded_src = 0;
  std::memcpy(&recorded_mask, d->store_data.data(), sizeof(recorded_mask));
  std::memcpy(&recorded_src, d->store_data.data() + sizeof(uint32_t), sizeof(recorded_src));
  EXPECT_EQ(recorded_mask, kMask);
  EXPECT_EQ(recorded_src, kSrc);

  amdgpu::LocalMemPipeline pipeline(&cu->lds());
  pipeline.issue(inst.release(), *wf);

  EXPECT_EQ(cu->lds().read32(wf->lds_base() + kLdsOff), kExpected);
  EXPECT_TRUE(wf->wait_counters().empty());
}

TEST(Gfx1250MemoryExecutionTest, DsMskorRtnB64ReturnsOldAndUpdatesLds) {
  amdgpu::GpuMemory mem("gfx1250_ds_mskor_rtn_b64_mem");
  amdgpu::L2Cache l2("gfx1250_ds_mskor_rtn_b64_l2");
  auto cu = make_gfx1250_cu(mem, l2, "gfx1250_ds_mskor_rtn_b64_cu");
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 128);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(0x200);

  constexpr uint32_t kAddrReg = 4;
  constexpr uint32_t kMaskReg = 8;
  constexpr uint32_t kSrcReg = 12;
  constexpr uint32_t kDstReg = 16;
  constexpr uint32_t kLdsOff = 0x60;
  constexpr uint64_t kOld = 0x1122'3344'5566'7788ULL;
  constexpr uint64_t kMask = 0x00ff'00ff'00ff'00ffULL;
  constexpr uint64_t kSrc = 0xaa00'bb00'cc00'dd00ULL;
  constexpr uint64_t kExpected = (kOld & ~kMask) | kSrc;

  const uint32_t vb = wf->vgpr_alloc().base;
  cu->write_vgpr(vb + kAddrReg, 0, kLdsOff);
  cu->write_vgpr(vb + kMaskReg, 0, static_cast<uint32_t>(kMask));
  cu->write_vgpr(vb + kMaskReg + 1, 0, static_cast<uint32_t>(kMask >> 32));
  cu->write_vgpr(vb + kSrcReg, 0, static_cast<uint32_t>(kSrc));
  cu->write_vgpr(vb + kSrcReg + 1, 0, static_cast<uint32_t>(kSrc >> 32));
  cu->lds().write64(wf->lds_base() + kLdsOff, kOld);
  wf->set_exec(1ULL);

  gfx1250::VdsMachineInst raw{};
  raw.addr = kAddrReg;
  raw.data0 = kMaskReg;
  raw.data1 = kSrcReg;
  raw.vdst = kDstReg;
  auto inst = std::make_unique<gfx1250::DsMskorRtnB64Vds>(
      reinterpret_cast<const gfx1250::MachineInst *>(&raw));
  EXPECT_TRUE(inst->is_memory_op());
  inst->execute_impl(*wf);

  const auto *d = inst->data_as<amdgpu::VectorMemState>();
  ASSERT_NE(d, nullptr);
  EXPECT_TRUE(d->is_load);
  EXPECT_EQ(d->atomic_op, amdgpu::AtomicOp::MSKOR);
  EXPECT_EQ(d->dst_reg_base, vb + kDstReg);
  ASSERT_GE(d->store_data.size(), 4u * sizeof(uint32_t));
  uint64_t recorded_mask = 0;
  uint64_t recorded_src = 0;
  std::memcpy(&recorded_mask, d->store_data.data(), sizeof(recorded_mask));
  std::memcpy(&recorded_src, d->store_data.data() + sizeof(recorded_mask), sizeof(recorded_src));
  EXPECT_EQ(recorded_mask, kMask);
  EXPECT_EQ(recorded_src, kSrc);

  amdgpu::LocalMemPipeline pipeline(&cu->lds());
  pipeline.issue(inst.release(), *wf);

  EXPECT_EQ(cu->lds().read64(wf->lds_base() + kLdsOff), kExpected);
  EXPECT_EQ(cu->read_vgpr(vb + kDstReg, 0), static_cast<uint32_t>(kOld));
  EXPECT_EQ(cu->read_vgpr(vb + kDstReg + 1, 0), static_cast<uint32_t>(kOld >> 32));
  EXPECT_TRUE(wf->wait_counters().empty());
}

TEST(Gfx1250MemoryExecutionTest, DsAppendConsumeUseM0CounterAndActiveLaneRanks) {
  amdgpu::GpuMemory mem("gfx1250_ds_append_consume_mem");
  amdgpu::L2Cache l2("gfx1250_ds_append_consume_l2");
  auto cu = make_gfx1250_cu(mem, l2, "gfx1250_ds_append_consume_cu");
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 128);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(0x300);
  wf->set_m0(0x80);
  wf->set_exec(0b1101ULL);

  constexpr uint32_t kOffset = 0x24;
  constexpr uint32_t kAppendDst = 20;
  constexpr uint32_t kConsumeDst = 24;
  constexpr uint32_t kInactiveSentinel = 0xfeed'faceu;
  const uint32_t counter_addr = wf->lds_base() + wf->m0() + kOffset;
  const uint32_t vb = wf->vgpr_alloc().base;
  cu->write_vgpr(vb + kAppendDst, 1, kInactiveSentinel);

  gfx1250::VdsMachineInst append_raw{};
  append_raw.offset0 = kOffset & 0xff;
  append_raw.offset1 = kOffset >> 8;
  append_raw.vdst = kAppendDst;
  auto append = std::make_unique<gfx1250::DsAppendVds>(
      reinterpret_cast<const gfx1250::MachineInst *>(&append_raw));
  cu->lds().write32(counter_addr, 10);
  append->execute_impl(*wf);
  const auto *append_d = append->data_as<amdgpu::VectorMemState>();
  ASSERT_NE(append_d, nullptr);
  EXPECT_EQ(append_d->atomic_op, amdgpu::AtomicOp::APPEND);
  EXPECT_EQ(append_d->per_lane_addr[0], counter_addr);
  EXPECT_EQ(append_d->per_lane_addr[2], counter_addr);
  EXPECT_EQ(append_d->per_lane_addr[3], counter_addr);

  amdgpu::LocalMemPipeline pipeline(&cu->lds());
  pipeline.issue(append.release(), *wf);

  EXPECT_EQ(cu->lds().read32(counter_addr), 13u);
  EXPECT_EQ(cu->read_vgpr(vb + kAppendDst, 0), 10u);
  EXPECT_EQ(cu->read_vgpr(vb + kAppendDst, 1), kInactiveSentinel);
  EXPECT_EQ(cu->read_vgpr(vb + kAppendDst, 2), 11u);
  EXPECT_EQ(cu->read_vgpr(vb + kAppendDst, 3), 12u);

  gfx1250::VdsMachineInst consume_raw{};
  consume_raw.offset0 = kOffset & 0xff;
  consume_raw.offset1 = kOffset >> 8;
  consume_raw.vdst = kConsumeDst;
  auto consume = std::make_unique<gfx1250::DsConsumeVds>(
      reinterpret_cast<const gfx1250::MachineInst *>(&consume_raw));
  cu->lds().write32(counter_addr, 20);
  cu->write_vgpr(vb + kConsumeDst, 1, kInactiveSentinel);
  consume->execute_impl(*wf);
  const auto *consume_d = consume->data_as<amdgpu::VectorMemState>();
  ASSERT_NE(consume_d, nullptr);
  EXPECT_EQ(consume_d->atomic_op, amdgpu::AtomicOp::CONSUME);
  EXPECT_EQ(consume_d->per_lane_addr[0], counter_addr);
  EXPECT_EQ(consume_d->per_lane_addr[2], counter_addr);
  EXPECT_EQ(consume_d->per_lane_addr[3], counter_addr);

  pipeline.issue(consume.release(), *wf);

  EXPECT_EQ(cu->lds().read32(counter_addr), 17u);
  EXPECT_EQ(cu->read_vgpr(vb + kConsumeDst, 0), 19u);
  EXPECT_EQ(cu->read_vgpr(vb + kConsumeDst, 1), kInactiveSentinel);
  EXPECT_EQ(cu->read_vgpr(vb + kConsumeDst, 2), 18u);
  EXPECT_EQ(cu->read_vgpr(vb + kConsumeDst, 3), 17u);
  EXPECT_TRUE(wf->wait_counters().empty());
}

TEST(Gfx1250MemoryExecutionTest, DsBarrierArriveUpdatesAsyncAndReturningForms) {
  amdgpu::GpuMemory mem("gfx1250_ds_barrier_arrive_mem");
  amdgpu::L2Cache l2("gfx1250_ds_barrier_arrive_l2");
  auto cu = make_gfx1250_cu(mem, l2, "gfx1250_ds_barrier_arrive_cu");
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 128);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(0x400);
  wf->set_exec(1ULL);

  constexpr uint32_t kAddrReg = 4;
  constexpr uint32_t kDataReg = 8;
  constexpr uint32_t kDstReg = 12;
  constexpr uint32_t kAsyncOff = 0x80;
  constexpr uint32_t kRtnOff = 0xa0;
  const uint32_t vb = wf->vgpr_alloc().base;

  cu->write_vgpr(vb + kAddrReg, 0, kAsyncOff);
  cu->lds().write64(wf->lds_base() + kAsyncOff, 0);

  gfx1250::VdsMachineInst async_raw{};
  async_raw.addr = kAddrReg;
  auto async = std::make_unique<gfx1250::DsAtomicAsyncBarrierArriveB64Vds>(
      reinterpret_cast<const gfx1250::MachineInst *>(&async_raw));
  async->execute_impl(*wf);
  const auto *async_d = async->data_as<amdgpu::VectorMemState>();
  ASSERT_NE(async_d, nullptr);
  EXPECT_FALSE(async_d->is_load);
  EXPECT_EQ(async_d->atomic_op, amdgpu::AtomicOp::BARRIER_ARRIVE);
  EXPECT_EQ(async_d->wait_counter_type, amdgpu::WaitCounterType::ASYNCCNT);

  amdgpu::LocalMemPipeline pipeline(&cu->lds());
  pipeline.issue(async.release(), *wf);

  EXPECT_EQ(cu->lds().read64(wf->lds_base() + kAsyncOff), 7ULL << 29);
  EXPECT_TRUE(wf->wait_counters().empty());

  constexpr uint64_t kOldState = (4ULL << 32) | 4ULL;
  constexpr uint64_t kDecrement = 2;
  constexpr uint64_t kExpectedState = (4ULL << 32) | 2ULL;
  cu->write_vgpr(vb + kAddrReg, 0, kRtnOff);
  cu->write_vgpr(vb + kDataReg, 0, static_cast<uint32_t>(kDecrement));
  cu->write_vgpr(vb + kDataReg + 1, 0, static_cast<uint32_t>(kDecrement >> 32));
  cu->lds().write64(wf->lds_base() + kRtnOff, kOldState);

  gfx1250::VdsMachineInst rtn_raw{};
  rtn_raw.addr = kAddrReg;
  rtn_raw.data0 = kDataReg;
  rtn_raw.vdst = kDstReg;
  auto rtn = std::make_unique<gfx1250::DsAtomicBarrierArriveRtnB64Vds>(
      reinterpret_cast<const gfx1250::MachineInst *>(&rtn_raw));
  rtn->execute_impl(*wf);
  const auto *rtn_d = rtn->data_as<amdgpu::VectorMemState>();
  ASSERT_NE(rtn_d, nullptr);
  EXPECT_TRUE(rtn_d->is_load);
  EXPECT_EQ(rtn_d->atomic_op, amdgpu::AtomicOp::BARRIER_ARRIVE);
  EXPECT_EQ(rtn_d->wait_counter_type, amdgpu::WaitCounterType::DSCNT);
  EXPECT_EQ(rtn_d->dst_reg_base, vb + kDstReg);

  pipeline.issue(rtn.release(), *wf);

  EXPECT_EQ(cu->lds().read64(wf->lds_base() + kRtnOff), kExpectedState);
  EXPECT_EQ(cu->read_vgpr(vb + kDstReg, 0), static_cast<uint32_t>(kOldState));
  EXPECT_EQ(cu->read_vgpr(vb + kDstReg + 1, 0), static_cast<uint32_t>(kOldState >> 32));
  EXPECT_TRUE(wf->wait_counters().empty());
}

TEST(Gfx1250MemoryExecutionTest, DsBpermuteFiFetchesInactiveSourceLane) {
  amdgpu::GpuMemory mem("gfx1250_ds_bpermute_fi_mem");
  amdgpu::L2Cache l2("gfx1250_ds_bpermute_fi_l2");
  auto cu = make_gfx1250_cu(mem, l2, "gfx1250_ds_bpermute_fi_cu");
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 128);
  ASSERT_NE(wf, nullptr);

  constexpr uint32_t kAddrReg = 4;
  constexpr uint32_t kDataReg = 8;
  constexpr uint32_t kDstReg = 12;
  constexpr uint32_t kLane1Value = 0x1357'9bdfu;
  constexpr uint32_t kInactiveDstSentinel = 0x2468'ace0u;
  const uint32_t vb = wf->vgpr_alloc().base;
  cu->write_vgpr(vb + kAddrReg, 0, 4);
  cu->write_vgpr(vb + kDataReg, 0, 0);
  cu->write_vgpr(vb + kDataReg, 1, kLane1Value);
  cu->write_vgpr(vb + kDstReg, 1, kInactiveDstSentinel);
  wf->set_exec(1ULL);

  gfx1250::VdsMachineInst raw{};
  raw.addr = kAddrReg;
  raw.data0 = kDataReg;
  raw.vdst = kDstReg;
  gfx1250::DsBpermuteFiB32Vds inst(reinterpret_cast<const gfx1250::MachineInst *>(&raw));
  inst.execute_impl(*wf);

  EXPECT_EQ(cu->read_vgpr(vb + kDstReg, 0), kLane1Value);
  EXPECT_EQ(cu->read_vgpr(vb + kDstReg, 1), kInactiveDstSentinel);
}

TEST(Gfx1250WavefrontTest, MasksExecAndVccToWave32) {
  amdgpu::GpuMemory mem("gfx1250_wave_mask_mem");
  amdgpu::L2Cache l2("gfx1250_wave_mask_l2");
  auto cu = make_gfx1250_cu(mem, l2, "gfx1250_wave_mask_cu");
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 128);
  ASSERT_NE(wf, nullptr);

  wf->set_exec(~0ULL);
  EXPECT_EQ(wf->exec(), 0xFFFF'FFFFULL);
  wf->set_exec(0xFFFF'FFFF'0000'0000ULL);
  EXPECT_EQ(wf->exec(), 0ULL);

  wf->set_vcc(~0ULL);
  EXPECT_EQ(wf->vcc(), 0xFFFF'FFFFULL);
  wf->set_vcc(0xFFFF'FFFF'0000'0000ULL);
  EXPECT_EQ(wf->vcc(), 0ULL);
}

TEST(Gfx1250ScalarExecutionTest, PackHlUsesHighLowHalves) {
  const uint32_t result = run_gfx1250_sop2<gfx1250::SPackHlB32B16Sop2>(0xAABB'CCDDu, 0x1122'3344u);
  EXPECT_EQ(result, 0x3344'AABBu);
}

TEST(Gfx1250ScalarExecutionTest, MinNumMaxNumIgnoreSingleNan) {
  const uint32_t qnan = 0x7FC0'0000u;
  const uint32_t one = std::bit_cast<uint32_t>(1.0f);
  const uint32_t two = std::bit_cast<uint32_t>(2.0f);

  EXPECT_EQ(run_gfx1250_sop2<gfx1250::SMinNumF32Sop2>(qnan, one), one);
  EXPECT_EQ(run_gfx1250_sop2<gfx1250::SMaxNumF32Sop2>(two, qnan), two);
}

TEST(Gfx1250ScalarExecutionTest, MinimumMaximumPropagateNanAndHandleSignedZero) {
  const uint32_t qnan = 0x7FC0'0000u;
  const uint32_t one = std::bit_cast<uint32_t>(1.0f);
  EXPECT_TRUE(
      std::isnan(std::bit_cast<float>(run_gfx1250_sop2<gfx1250::SMinimumF32Sop2>(qnan, one))));
  EXPECT_TRUE(
      std::isnan(std::bit_cast<float>(run_gfx1250_sop2<gfx1250::SMaximumF32Sop2>(one, qnan))));

  const uint32_t pos_zero = std::bit_cast<uint32_t>(0.0f);
  const uint32_t neg_zero = std::bit_cast<uint32_t>(-0.0f);
  EXPECT_EQ(run_gfx1250_sop2<gfx1250::SMinimumF32Sop2>(pos_zero, neg_zero), neg_zero);
  EXPECT_EQ(run_gfx1250_sop2<gfx1250::SMaximumF32Sop2>(neg_zero, pos_zero), pos_zero);
}

TEST(Gfx1250ScalarExecutionTest, F16MinMaxAndCvtPkRtz) {
  const uint32_t three_h = util::f32_to_f16(3.0f);
  const uint32_t neg_two_h = util::f32_to_f16(-2.0f);
  EXPECT_EQ(run_gfx1250_sop2<gfx1250::SMinNumF16Sop2>(three_h, neg_two_h) & 0xFFFFu, neg_two_h);
  EXPECT_EQ(run_gfx1250_sop2<gfx1250::SMaximumF16Sop2>(three_h, neg_two_h) & 0xFFFFu, three_h);

  const uint32_t result = run_gfx1250_sop2<gfx1250::SCvtPkRtzF16F32Sop2>(
      std::bit_cast<uint32_t>(1.0008f), std::bit_cast<uint32_t>(-1.0008f));
  EXPECT_EQ(result, 0xBC00'3C00u);
}

TEST(Gfx1250ScalarExecutionTest, SCallI64WritesReturnPcAndBranches) {
  amdgpu::GpuMemory mem("gfx1250_s_call_mem");
  amdgpu::L2Cache l2("gfx1250_s_call_l2");
  auto cu = make_gfx1250_cu(mem, l2, "gfx1250_s_call_cu");
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 128);
  ASSERT_NE(wf, nullptr);

  constexpr uint32_t kDst = 8;
  const uint32_t sb = wf->sgpr_alloc().base;
  auto read_dst = [&]() {
    return static_cast<uint64_t>(cu->read_sgpr(sb + kDst + 1)) << 32 | cu->read_sgpr(sb + kDst);
  };

  gfx1250::SopkMachineInst raw{};
  raw.sdst = kDst;
  raw.simm16 = 3;
  gfx1250::SCallI64Sopk forward(reinterpret_cast<const gfx1250::MachineInst *>(&raw));

  wf->pc = 0x1000;
  forward.execute_impl(*wf);
  EXPECT_EQ(read_dst(), 0x1004ull);
  EXPECT_EQ(wf->pc, 0x1000ull + 4ull + 3ull * 4ull - 4ull);

  raw.simm16 = 0xfffeu;
  gfx1250::SCallI64Sopk backward(reinterpret_cast<const gfx1250::MachineInst *>(&raw));
  wf->pc = 0x2000;
  backward.execute_impl(*wf);
  EXPECT_EQ(read_dst(), 0x2004ull);
  EXPECT_EQ(wf->pc, 0x2000ull + 4ull - 2ull * 4ull - 4ull);
}

TEST(Gfx1250ScalarExecutionTest, SSendmsgRtnWritesDeterministicPlaceholders) {
  amdgpu::GpuMemory mem("gfx1250_sendmsg_rtn_mem");
  amdgpu::L2Cache l2("gfx1250_sendmsg_rtn_l2");
  auto cu = make_gfx1250_cu(mem, l2, "gfx1250_sendmsg_rtn_cu");
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 128);
  ASSERT_NE(wf, nullptr);

  const uint32_t sb = wf->sgpr_alloc().base;
  gfx1250::Sop1MachineInst raw{};
  raw.sdst = 8;
  raw.ssrc0 = 0x80; // MSG_RTN_GET_DOORBELL.
  gfx1250::SSendmsgRtnB32Sop1 doorbell(reinterpret_cast<const gfx1250::MachineInst *>(&raw));
  doorbell.execute_impl(*wf);
  EXPECT_EQ(cu->read_sgpr(sb + 8), 0u);

  cu->write_sgpr(sb + 10, 0xffff'ffffu);
  cu->write_sgpr(sb + 11, 0xffff'ffffu);
  raw.sdst = 10;
  raw.ssrc0 = 0x82; // MSG_RTN_GET_TMA.
  gfx1250::SSendmsgRtnB64Sop1 tma(reinterpret_cast<const gfx1250::MachineInst *>(&raw));
  tma.execute_impl(*wf);
  EXPECT_EQ(cu->read_sgpr(sb + 10), 0u);
  EXPECT_EQ(cu->read_sgpr(sb + 11), 0u);
}

TEST(Gfx1250ScalarExecutionTest, SVersionIsNoop) {
  amdgpu::GpuMemory mem("gfx1250_version_mem");
  amdgpu::L2Cache l2("gfx1250_version_l2");
  auto cu = make_gfx1250_cu(mem, l2, "gfx1250_version_cu");
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 128);
  ASSERT_NE(wf, nullptr);

  gfx1250::SopkMachineInst raw{};
  raw.simm16 = 1;
  gfx1250::SVersionSopk inst(reinterpret_cast<const gfx1250::MachineInst *>(&raw));
  EXPECT_NO_THROW(inst.execute_impl(*wf));
}

TEST(Gfx1250ScalarCoTest, SignedAddCoUsesUnsignedCarryOut) {
  amdgpu::GpuMemory mem("gfx1250_add_co_mem");
  amdgpu::L2Cache l2("gfx1250_add_co_l2");
  auto cu = make_gfx1250_cu(mem, l2, "gfx1250_add_co_cu");
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 128);
  ASSERT_NE(wf, nullptr);

  uint32_t word = (0x2u << 30) | (2u << 23) | (2u << 16) | (1u << 8) | 0u;
  gfx1250::SAddCoI32Sop2 inst(&word);
  const uint32_t sb = wf->sgpr_alloc().base;

  cu->write_sgpr(sb + 0, 0x7FFF'FFFFu);
  cu->write_sgpr(sb + 1, 1u);
  inst.execute_impl(*wf);
  EXPECT_EQ(cu->read_sgpr(sb + 2), 0x8000'0000u);
  EXPECT_FALSE(wf->read_scc());

  cu->write_sgpr(sb + 0, 0xFFFF'FFFFu);
  cu->write_sgpr(sb + 1, 1u);
  inst.execute_impl(*wf);
  EXPECT_EQ(cu->read_sgpr(sb + 2), 0u);
  EXPECT_TRUE(wf->read_scc());
}

TEST(Gfx1250ScalarCoTest, SignedSubCoUsesUnsignedBorrowOut) {
  amdgpu::GpuMemory mem("gfx1250_sub_co_mem");
  amdgpu::L2Cache l2("gfx1250_sub_co_l2");
  auto cu = make_gfx1250_cu(mem, l2, "gfx1250_sub_co_cu");
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 128, 128);
  ASSERT_NE(wf, nullptr);

  uint32_t word = (0x2u << 30) | (3u << 23) | (2u << 16) | (1u << 8) | 0u;
  gfx1250::SSubCoI32Sop2 inst(&word);
  const uint32_t sb = wf->sgpr_alloc().base;

  cu->write_sgpr(sb + 0, 0x8000'0000u);
  cu->write_sgpr(sb + 1, 1u);
  inst.execute_impl(*wf);
  EXPECT_EQ(cu->read_sgpr(sb + 2), 0x7FFF'FFFFu);
  EXPECT_FALSE(wf->read_scc());

  cu->write_sgpr(sb + 0, 0u);
  cu->write_sgpr(sb + 1, 1u);
  inst.execute_impl(*wf);
  EXPECT_EQ(cu->read_sgpr(sb + 2), 0xFFFF'FFFFu);
  EXPECT_TRUE(wf->read_scc());
}

// ---------------------------------------------------------------------------
// Scratch address calculation tests
// ---------------------------------------------------------------------------

TEST(ScratchAddrCalcTest, FlatScratchUsesWavefrontBase) {
  // Verify that FLAT with seg==1 (SCRATCH) computes:
  //   address = scratch_base + VGPR[lane] + offset
  amdgpu::GpuMemory mem("test_mem");
  amdgpu::L2Cache l2("test_l2");
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 104;
  cfg.vgprs_per_wf = 16;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("scratch_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 104, 16);
  ASSERT_NE(wf, nullptr);

  // Set scratch base to a known address.
  constexpr uint64_t SCRATCH_BASE = 0x1'0000'0000ULL;
  wf->set_scratch_base(SCRATCH_BASE);

  // Write a 32-bit offset into VGPR[0] lane 0.
  uint32_t vbase = wf->vgpr_alloc().base;
  cu->write_vgpr(vbase, 0, 0x100); // lane 0: offset 0x100

  // Set EXEC so only lane 0 is active.
  wf->set_exec(1ULL);

  // Build a FlatMachineInst with seg=1 (SCRATCH), saddr=0x7F (no SADDR),
  // offset=0x10.
  cdna4::FlatMachineInst inst{};
  inst.seg = 1;       // SCRATCH
  inst.saddr = 0x7F;  // No SADDR
  inst.addr = 0;      // VGPR index 0
  inst.offset = 0x10; // 12-bit immediate offset
  inst.pad_12 = 0;

  amdgpu::VectorMemState d(amdgpu::GLOBAL_MEM);
  amdgpu::addr_calc::flat_calculate_addresses(inst, *wf, d);

  EXPECT_EQ(d.lane_mask, 1ULL);
  // scratch_base (0x1_0000_0000) + VGPR (0x100) + offset (0x10) = 0x1_0000_0110
  EXPECT_EQ(d.per_lane_addr[0], SCRATCH_BASE + 0x100 + 0x10);
}

TEST(ScratchAddrCalcTest, FlatGlobalDoesNotUseScratchBase) {
  // Verify that FLAT with seg==2 (GLOBAL) does NOT add scratch_base.
  amdgpu::GpuMemory mem("test_mem");
  amdgpu::L2Cache l2("test_l2");
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 104;
  cfg.vgprs_per_wf = 16;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("global_cu", cfg, &mem, &l2);
  ASSERT_NE(cu, nullptr);

  auto *wf = cu->dispatch_wf(0, 0, 104, 16);
  ASSERT_NE(wf, nullptr);

  // Set scratch base — should be ignored for GLOBAL.
  wf->set_scratch_base(0xDEAD'0000ULL);

  // Write a 64-bit address into VGPR[0:1] lane 0.
  uint32_t vbase = wf->vgpr_alloc().base;
  cu->write_vgpr(vbase, 0, 0x2000);     // low 32
  cu->write_vgpr(vbase + 1, 0, 0x0001); // high 32 → addr = 0x1_0000_2000
  wf->set_exec(1ULL);

  cdna4::FlatMachineInst inst{};
  inst.seg = 2;      // GLOBAL
  inst.saddr = 0x7F; // No SADDR → use 64-bit VGPR pair
  inst.addr = 0;
  inst.offset = 0;
  inst.pad_12 = 0;

  amdgpu::VectorMemState d(amdgpu::GLOBAL_MEM);
  amdgpu::addr_calc::flat_calculate_addresses(inst, *wf, d);

  EXPECT_EQ(d.per_lane_addr[0], 0x1'0000'2000ULL); // No scratch_base added.
}

} // namespace
