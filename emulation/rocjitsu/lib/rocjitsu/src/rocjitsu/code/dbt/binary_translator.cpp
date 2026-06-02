// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/dbt/binary_translator.h"

#include "rocjitsu/analysis/def_use_chain.h"
#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/dbt/generated/encoding_cdna4_to_cdna3.h"
#include "rocjitsu/code/dbt/generated/encoding_cdna4_to_rdna3.h"
#include "rocjitsu/code/dbt/generated/encoding_cdna4_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/encoding_gfx1250_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna4_to_cdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna4_to_rdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna4_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_gfx1250_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_types.h"
#include "rocjitsu/code/dbt/kernel_descriptor_translator.h"
#include "rocjitsu/code/dbt/semantic_translator.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/vgpr_msb.h"

#include <algorithm>
#include <array>
#include <bit>
#include <bitset>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rocjitsu {

namespace {

constexpr uint32_t kConservativeLoweringMinimumVgprs = 128;
constexpr uint32_t kGfx1250Rdna4SemanticMinimumVgprs = 128;
// The f16 K32 WMMA split can choose an aligned 9-VGPR scratch window at
// v216..v224 in dense IREE kernels. RDNA4 wave32 descriptors granulate VGPRs
// by 8, so those scopes need a 232-VGPR floor to make v224 addressable.
constexpr uint32_t kGfx1250F16WmmaScratchMinimumVgprs = 232;
constexpr uint32_t kMaxInPlaceMetadataVgprs = 127;
constexpr uint32_t kSCodeEnd = 0xBF9F0000u;
constexpr uint32_t kSNop0 = 0xBF800000u;
constexpr uint16_t kGfx1250VopdEncodingId = 0x032u;
constexpr uint16_t kGfx1250Vopd3EncodingId = 0x0CFu;
constexpr uint16_t kGfx1250Vop3pEncodingId = 0x198u;
constexpr uint16_t kGfx1250Vop3p1EncodingId = 0x199u;
constexpr uint16_t kGfx1250Vop2AddNcU64EncodingId0 = 0x0A0u;
constexpr uint16_t kGfx1250Vop2AddNcU64EncodingId3 = 0x0A3u;
constexpr uint16_t kGfx1250Vop2SubNcU64EncodingId0 = 0x0A4u;
constexpr uint16_t kGfx1250Vop2SubNcU64EncodingId3 = 0x0A7u;
constexpr uint16_t kGfx1250Vop2MulU64EncodingId0 = 0x0A8u;
constexpr uint16_t kGfx1250Vop2MulU64EncodingId3 = 0x0ABu;
constexpr uint16_t kGfx1250VAddNcU64E32Opcode = 40u;
constexpr uint16_t kGfx1250VSubNcU64E32Opcode = 41u;
constexpr uint16_t kGfx1250VMulU64E32Opcode = 42u;
constexpr uint8_t kGfx1250VAddF16E32Opcode = 50u;
constexpr uint32_t kVop3Encoding = 0x35u;
constexpr uint16_t kGfx1250WmmaF32F16K32Opcode = 0x60u;
constexpr uint16_t kGfx1250SwmmacF32F16K64Opcode = 0x65u;
constexpr uint16_t kGfx1250WmmaI32Iu8K64Opcode = 0x72u;
constexpr uint16_t kGfx1250SwmmacI32Iu8K128Opcode = 0x7Bu;
constexpr uint16_t kGfx1250WmmaF32Fp8Fp8K128Opcode = 0x80u;
constexpr uint32_t kGfx1250HighBankBaseVgpr = 256u;
constexpr uint32_t kGfx1250VMulU64HighBankScratchCount = 2u;
constexpr uint32_t kGfx1250K128Fp8BorrowedVgprCount = 5u;
constexpr uint32_t kGfx1250K128Fp8PrivateScratchBytes =
    kGfx1250K128Fp8BorrowedVgprCount * sizeof(uint32_t);
constexpr uint32_t kGfx1250HighBankShadowLowSaveVgprCount = 8u;
constexpr uint32_t kGfx1250HighBankShadowLowSaveBytes =
    kGfx1250HighBankShadowLowSaveVgprCount * sizeof(uint32_t);
constexpr uint32_t kRdna4MaxVgprsPerWave = 512u;
constexpr uint32_t kRdna4MaxSgprsPerWave = 106u;
constexpr uint8_t kRdna4NullSgpr = 124u;
constexpr uint32_t kRdna4RawBufferConfigWord = 0x31027000u;
constexpr uint32_t kRdna4RawBufferUnboundedRange = 0xFFFFFFFFu;
constexpr uint8_t kGfx1250SOpAndB32 = 22u;
constexpr uint8_t kGfx1250SOpOrB32 = 24u;
constexpr uint8_t kGfx1250SOpOrB64 = 25u;
constexpr uint16_t kGfx1250ScalarOperandTtmp9 = 117u;
constexpr uint8_t kSoppWaitLoadcnt = 64u;
constexpr uint8_t kSoppWaitStorecnt = 65u;
constexpr uint8_t kSoppWaitDscnt = 70u;
constexpr uint8_t kSoppWaitKmcnt = 71u;
constexpr uint8_t kGfx1250SoppWaitXcnt = 69u;
constexpr uint64_t kSoppBranchMaxForwardBytes =
    static_cast<uint64_t>(std::numeric_limits<int16_t>::max()) * sizeof(uint32_t);
constexpr size_t kGfx1250DescriptorUseScanLimit = 512;

uint32_t metadata_vgpr_count_for_in_place_patch(std::span<const KdTranslation> translations) {
  uint32_t max_vgprs = 0;
  for (const KdTranslation &translation : translations)
    max_vgprs = std::max(max_vgprs, translation.target_vgpr_count);
  return std::min(max_vgprs, kMaxInPlaceMetadataVgprs);
}

uint32_t conservative_lowering_minimum_vgprs(rj_code_arch_t guest_arch, rj_code_arch_t host_arch) {
  if (guest_arch == ROCJITSU_CODE_ARCH_GFX1250 && host_arch == ROCJITSU_CODE_ARCH_RDNA4)
    return kGfx1250Rdna4SemanticMinimumVgprs;
  return kConservativeLoweringMinimumVgprs;
}

bool needs_metadata_private_segment_patch(std::span<const KdTranslation> translations) {
  return std::ranges::any_of(translations, [](const KdTranslation &translation) {
    return translation.private_spill_zone_bytes != 0;
  });
}

EncodingTranslateFn select_encoding_translator(rj_code_arch_t guest, rj_code_arch_t host) {
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_RDNA4)
    return cdna4_to_rdna4::translate_encoding_cdna4_to_rdna4;
  if (guest == ROCJITSU_CODE_ARCH_GFX1250 && host == ROCJITSU_CODE_ARCH_RDNA4)
    return gfx1250_to_rdna4::translate_encoding_gfx1250_to_rdna4;
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_CDNA3)
    return cdna4_to_cdna3::translate_encoding_cdna4_to_cdna3;
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_RDNA3)
    return cdna4_to_rdna3::translate_encoding_cdna4_to_rdna3;
  return nullptr;
}

LegalizationLookupFn select_legalization(rj_code_arch_t guest, rj_code_arch_t host) {
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_RDNA4) {
    return [](uint16_t enc_id, uint16_t opcode) -> const InstructionLegalization * {
      return lookup(kLegalization_cdna4_to_rdna4, enc_id, opcode);
    };
  }
  if (guest == ROCJITSU_CODE_ARCH_GFX1250 && host == ROCJITSU_CODE_ARCH_RDNA4) {
    return [](uint16_t enc_id, uint16_t opcode) -> const InstructionLegalization * {
      return lookup(kLegalization_gfx1250_to_rdna4, enc_id, opcode);
    };
  }
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_CDNA3) {
    return [](uint16_t enc_id, uint16_t opcode) -> const InstructionLegalization * {
      return lookup(kLegalization_cdna4_to_cdna3, enc_id, opcode);
    };
  }
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_RDNA3) {
    return [](uint16_t enc_id, uint16_t opcode) -> const InstructionLegalization * {
      return lookup(kLegalization_cdna4_to_rdna3, enc_id, opcode);
    };
  }
  return nullptr;
}

[[nodiscard]] bool is_gfx1250_k128_fp8_wmma(const Instruction &inst) {
  return inst.encoding_id() == kGfx1250Vop3p1EncodingId &&
         inst.opcode() == kGfx1250WmmaF32Fp8Fp8K128Opcode;
}

[[nodiscard]] bool is_gfx1250_k32_f16_wmma(const Instruction &inst) {
  return inst.encoding_id() == kGfx1250Vop3pEncodingId &&
         inst.opcode() == kGfx1250WmmaF32F16K32Opcode;
}

[[nodiscard]] BasicBlock *block_for_offset(const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                                           uint64_t offset) {
  for (const auto &block : blocks) {
    if (block && block->start_offset() <= offset && offset < block->end_offset())
      return block.get();
  }
  return nullptr;
}

[[nodiscard]] bool compute_sopp_branch_offset(uint64_t branch_pc, uint64_t target,
                                              int16_t &offset_dwords) {
  // SOPP branches encode a signed dword offset from the next instruction. Keep
  // the range check shared so both cave entry and return branches fail closed.
  constexpr int64_t kBranchPcBiasBytes = static_cast<int64_t>(sizeof(uint32_t));
  constexpr uint64_t kMaxSignedTarget = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
  constexpr uint64_t kMaxSignedBranchPc =
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max() - kBranchPcBiasBytes);
  // The PCs are unsigned until this check passes. Compare against the casted
  // signed int64_t limits so the later signed conversion, and branch_pc + 4,
  // cannot overflow.
  if (branch_pc > kMaxSignedBranchPc || target > kMaxSignedTarget)
    return false;

  const int64_t delta_bytes = static_cast<int64_t>(target) - (static_cast<int64_t>(branch_pc) + 4);
  if (delta_bytes % static_cast<int64_t>(sizeof(uint32_t)) != 0)
    return false;

  const int64_t delta_dwords = delta_bytes / static_cast<int64_t>(sizeof(uint32_t));
  if (delta_dwords < std::numeric_limits<int16_t>::min() ||
      delta_dwords > std::numeric_limits<int16_t>::max())
    return false;

  offset_dwords = static_cast<int16_t>(delta_dwords);
  return true;
}

[[nodiscard]] std::vector<uint32_t> raw_words_for_inst(const Instruction &inst) {
  const uint32_t *raw = inst.raw_encoding();
  if (!raw)
    return {};
  return {raw, raw + inst.size() / sizeof(uint32_t)};
}

[[nodiscard]] bool words_changed(std::span<const uint32_t> before,
                                 std::span<const uint32_t> after) {
  if (before.size() != after.size())
    return true;
  return !std::ranges::equal(before, after);
}

void append_diagnostic(std::vector<TranslationDiagnostic> &diagnostics, DiagnosticSeverity severity,
                       DiagnosticKind kind, std::string message,
                       std::optional<uint64_t> guest_offset = std::nullopt,
                       std::string mnemonic = {}, std::vector<std::string> required_work = {}) {
  diagnostics.push_back({.severity = severity,
                         .kind = kind,
                         .guest_offset = guest_offset,
                         .mnemonic = std::move(mnemonic),
                         .message = std::move(message),
                         .required_work = std::move(required_work)});
}

void append_error(std::vector<TranslationDiagnostic> &diagnostics, DiagnosticKind kind,
                  std::string message, std::optional<uint64_t> guest_offset = std::nullopt,
                  std::string mnemonic = {}, std::vector<std::string> required_work = {}) {
  append_diagnostic(diagnostics, DiagnosticSeverity::Error, kind, std::move(message), guest_offset,
                    std::move(mnemonic), std::move(required_work));
}

void append_diagnostics(std::vector<TranslationDiagnostic> &dst,
                        const std::vector<TranslationDiagnostic> &src) {
  dst.insert(dst.end(), src.begin(), src.end());
}

[[nodiscard]] uint64_t align_up_to_word(uint64_t offset) {
  constexpr uint64_t kWordMask = sizeof(uint32_t) - 1;
  return (offset + kWordMask) & ~kWordMask;
}

[[nodiscard]] uint32_t read_u32(std::span<const uint8_t> bytes, uint64_t offset) {
  uint32_t value = 0;
  std::memcpy(&value, bytes.data() + offset, sizeof(value));
  return value;
}

[[nodiscard]] std::optional<uint32_t> read_trailing_literal_u32(std::span<const uint8_t> text,
                                                                uint64_t offset,
                                                                uint32_t literal_byte_offset) {
  if (offset + literal_byte_offset + sizeof(uint32_t) > text.size())
    return std::nullopt;
  return read_u32(text, offset + literal_byte_offset);
}

[[nodiscard]] bool has_unimplemented_expand_gap(const std::vector<std::string> *warnings) {
  if (warnings == nullptr)
    return false;
  return std::ranges::any_of(*warnings, [](const std::string &warning) {
    return warning.rfind("EXPAND not yet implemented for ", 0) == 0;
  });
}

[[nodiscard]] bool has_hardware_pending_semantic_lowering(std::string_view mnemonic) {
  (void)mnemonic;
  return false;
}

[[nodiscard]] constexpr std::pair<uint32_t, uint32_t>
build_rdna4_vop3(uint16_t op, uint8_t vdst, uint16_t src0, uint16_t src1, uint16_t src2 = 0) {
  const uint32_t w0 = (vdst & 0xFFu) | ((op & 0x3FFu) << 16) | (kVop3Encoding << 26);
  const uint32_t w1 = (src0 & 0x1FFu) | ((src1 & 0x1FFu) << 9) | ((src2 & 0x1FFu) << 18);
  return {w0, w1};
}

[[nodiscard]] constexpr std::pair<uint32_t, uint32_t>
build_rdna4_vop3_sdst(uint16_t op, uint8_t vdst, uint8_t sdst, uint16_t src0, uint16_t src1,
                      uint16_t src2 = 0) {
  const uint32_t w0 =
      (vdst & 0xFFu) | ((sdst & 0x7Fu) << 8) | ((op & 0x3FFu) << 16) | (kVop3Encoding << 26);
  const uint32_t w1 = (src0 & 0x1FFu) | ((src1 & 0x1FFu) << 9) | ((src2 & 0x1FFu) << 18);
  return {w0, w1};
}

[[nodiscard]] constexpr uint32_t build_rdna4_vop1(uint8_t op, uint8_t vdst, uint16_t src0) {
  return (src0 & 0x1FFu) | ((op & 0x7Fu) << 9) | ((vdst & 0xFFu) << 17) | (0x3Fu << 25);
}

[[nodiscard]] constexpr uint32_t build_rdna4_vop2(uint8_t op, uint8_t vdst, uint16_t src0,
                                                  uint8_t vsrc1) {
  return (src0 & 0x1FFu) | ((vsrc1 & 0xFFu) << 9) | ((vdst & 0xFFu) << 17) | ((op & 0x3Fu) << 25);
}

[[nodiscard]] constexpr uint32_t build_rdna4_vopc(uint8_t op, uint16_t src0, uint8_t vsrc1) {
  return (src0 & 0x1FFu) | ((vsrc1 & 0xFFu) << 9) | ((op & 0xFFu) << 17) | (0x3Eu << 25);
}

struct Gfx1250Sop1MovB32 {
  uint8_t sdst = 0;
  uint16_t ssrc0 = 0;
  std::optional<uint32_t> literal32;
  std::optional<uint64_t> literal64;
};

struct Gfx1250Sop1MovB64 {
  uint8_t sdst = 0;
  uint16_t ssrc0 = 0;
  std::optional<uint32_t> literal32;
  std::optional<uint64_t> literal64;
};

struct Gfx1250SopkMovkI32 {
  uint8_t sdst = 0;
  uint16_t simm16 = 0;
};

struct Gfx1250Sop2Literal32 {
  uint8_t sdst = 0;
  uint16_t non_literal_src = 0;
  uint32_t literal = 0;
};

struct Gfx1250Sop2Literal64 {
  uint8_t sdst = 0;
  uint16_t non_literal_src = 0;
  uint64_t literal = 0;
};

[[nodiscard]] std::optional<Gfx1250Sop1MovB32> decode_gfx1250_s_mov_b32(
    const Instruction &inst) {
  if (inst.encoding_id() != kEnc_SOP1 || inst.opcode() != 0)
    return std::nullopt;
  const uint32_t *raw = inst.raw_encoding();
  if (!raw)
    return std::nullopt;

  const auto src = std::bit_cast<gfx1250::Sop1MachineInst>(raw[0]);
  if (src.op != 0)
    return std::nullopt;
  Gfx1250Sop1MovB32 decoded{};
  decoded.sdst = static_cast<uint8_t>(src.sdst);
  decoded.ssrc0 = src.ssrc0;
  if (src.ssrc0 == 255u) {
    if (inst.size() != static_cast<int>(2 * sizeof(uint32_t)))
      return std::nullopt;
    const Operand *operand = inst.src_operand(0);
    if (!operand)
      return std::nullopt;
    decoded.literal32 = static_cast<uint32_t>(operand->encoding_value());
  } else if (src.ssrc0 == 254u) {
    if (inst.size() != static_cast<int>(3 * sizeof(uint32_t)))
      return std::nullopt;
    const Operand *operand = inst.src_operand(0);
    if (!operand)
      return std::nullopt;
    decoded.literal64 = operand->literal64_value();
    if (!decoded.literal64)
      return std::nullopt;
  } else if (inst.size() != static_cast<int>(sizeof(uint32_t))) {
    return std::nullopt;
  }
  return decoded;
}

[[nodiscard]] std::optional<Gfx1250Sop1MovB64> decode_gfx1250_s_mov_b64(
    const Instruction &inst) {
  if (inst.encoding_id() != kEnc_SOP1 || inst.opcode() != 1)
    return std::nullopt;
  const uint32_t *raw = inst.raw_encoding();
  if (!raw)
    return std::nullopt;

  const auto src = std::bit_cast<gfx1250::Sop1MachineInst>(raw[0]);
  if (src.op != 1)
    return std::nullopt;
  Gfx1250Sop1MovB64 decoded{};
  decoded.sdst = static_cast<uint8_t>(src.sdst);
  decoded.ssrc0 = src.ssrc0;
  if (src.ssrc0 == 255u) {
    if (inst.size() != static_cast<int>(2 * sizeof(uint32_t)))
      return std::nullopt;
    const Operand *operand = inst.src_operand(0);
    if (!operand)
      return std::nullopt;
    decoded.literal32 = static_cast<uint32_t>(operand->encoding_value());
  } else if (src.ssrc0 == 254u) {
    if (inst.size() != static_cast<int>(3 * sizeof(uint32_t)))
      return std::nullopt;
    const Operand *operand = inst.src_operand(0);
    if (!operand)
      return std::nullopt;
    decoded.literal64 = operand->literal64_value();
    if (!decoded.literal64)
      return std::nullopt;
  } else if (inst.size() != static_cast<int>(sizeof(uint32_t))) {
    return std::nullopt;
  }
  return decoded;
}

[[nodiscard]] std::optional<Gfx1250SopkMovkI32> decode_gfx1250_s_movk_i32(
    const Instruction &inst) {
  if ((inst.encoding_id() & 0x1E0u) != kEnc_SOPK || inst.opcode() != 0 ||
      inst.size() != static_cast<int>(sizeof(uint32_t)))
    return std::nullopt;
  const uint32_t *raw = inst.raw_encoding();
  if (!raw)
    return std::nullopt;

  const auto src = std::bit_cast<gfx1250::SopkMachineInst>(raw[0]);
  if (src.op != 0)
    return std::nullopt;
  return Gfx1250SopkMovkI32{static_cast<uint8_t>(src.sdst),
                            static_cast<uint16_t>(src.simm16)};
}

[[nodiscard]] bool is_sop2_encoding(const Instruction &inst) {
  return inst.encoding_id() >= kEnc_SOP2 && inst.encoding_id() < kEnc_SOPK;
}

bool remap_gfx1250_ttmp9_reads_to_sgpr(const Instruction &inst, int16_t rdna4_grid_x_sgpr,
                                       std::vector<uint32_t> &words) {
  if (rdna4_grid_x_sgpr < 0 || words.empty())
    return false;
  const auto sgpr = static_cast<uint16_t>(rdna4_grid_x_sgpr);
  if (sgpr >= kGfx1250ScalarOperandTtmp9)
    return false;

  const uint32_t *raw = inst.raw_encoding();
  if (!raw)
    return false;

  bool changed = false;
  auto replace_bits = [&](uint32_t mask, uint32_t shift) {
    words[0] = (words[0] & ~mask) | ((static_cast<uint32_t>(sgpr) << shift) & mask);
    changed = true;
  };

  if (inst.encoding_id() == kEnc_SOP1) {
    if (((words[0] >> 23) & 0x1FFu) != kEnc_SOP1)
      return false;
    const auto src = std::bit_cast<gfx1250::Sop1MachineInst>(raw[0]);
    if (src.ssrc0 == kGfx1250ScalarOperandTtmp9)
      replace_bits(0xFFu, 0);
    return changed;
  }

  if (inst.encoding_id() == kEnc_SOPC) {
    if (((words[0] >> 23) & 0x1FFu) != kEnc_SOPC)
      return false;
    const auto src = std::bit_cast<gfx1250::SopcMachineInst>(raw[0]);
    if (src.ssrc0 == kGfx1250ScalarOperandTtmp9)
      replace_bits(0xFFu, 0);
    if (src.ssrc1 == kGfx1250ScalarOperandTtmp9)
      replace_bits(0xFF00u, 8);
    return changed;
  }

  if (is_sop2_encoding(inst)) {
    if (((words[0] >> 30) & 0x3u) != kSop2EncodingPrefix)
      return false;
    const auto src = std::bit_cast<gfx1250::Sop2MachineInst>(raw[0]);
    if (src.ssrc0 == kGfx1250ScalarOperandTtmp9)
      replace_bits(0xFFu, 0);
    if (src.ssrc1 == kGfx1250ScalarOperandTtmp9)
      replace_bits(0xFF00u, 8);
  }
  return changed;
}

[[nodiscard]] std::optional<Gfx1250Sop2Literal32> decode_gfx1250_sop2_literal32(
    const Instruction &inst, uint8_t op) {
  if (!is_sop2_encoding(inst) || inst.opcode() != op ||
      inst.size() != static_cast<int>(2 * sizeof(uint32_t)))
    return std::nullopt;
  const uint32_t *raw = inst.raw_encoding();
  if (!raw)
    return std::nullopt;

  const auto src = std::bit_cast<gfx1250::Sop2MachineInst>(raw[0]);
  uint16_t non_literal_src = 0;
  uint8_t literal_operand = 0;
  if (src.ssrc0 == 255u && src.ssrc1 != 255u) {
    non_literal_src = src.ssrc1;
    literal_operand = 0;
  } else if (src.ssrc1 == 255u && src.ssrc0 != 255u) {
    non_literal_src = src.ssrc0;
    literal_operand = 1;
  } else {
    return std::nullopt;
  }

  const Operand *operand = inst.src_operand(literal_operand);
  if (!operand)
    return std::nullopt;
  return Gfx1250Sop2Literal32{static_cast<uint8_t>(src.sdst), non_literal_src,
                              static_cast<uint32_t>(operand->encoding_value())};
}

[[nodiscard]] std::optional<Gfx1250Sop2Literal64> decode_gfx1250_sop2_literal64(
    const Instruction &inst, uint8_t op) {
  if (!is_sop2_encoding(inst) || inst.opcode() != op ||
      inst.size() != static_cast<int>(3 * sizeof(uint32_t)))
    return std::nullopt;
  const uint32_t *raw = inst.raw_encoding();
  if (!raw)
    return std::nullopt;

  const auto src = std::bit_cast<gfx1250::Sop2MachineInst>(raw[0]);
  uint16_t non_literal_src = 0;
  uint8_t literal_operand = 0;
  if (src.ssrc0 == 254u && src.ssrc1 != 254u) {
    non_literal_src = src.ssrc1;
    literal_operand = 0;
  } else if (src.ssrc1 == 254u && src.ssrc0 != 254u) {
    non_literal_src = src.ssrc0;
    literal_operand = 1;
  } else {
    return std::nullopt;
  }

  const Operand *operand = inst.src_operand(literal_operand);
  if (!operand)
    return std::nullopt;
  const auto literal = operand->literal64_value();
  if (!literal)
    return std::nullopt;
  return Gfx1250Sop2Literal64{static_cast<uint8_t>(src.sdst), non_literal_src, *literal};
}

[[nodiscard]] std::optional<uint8_t> raw_buffer_resource_base_for_descriptor_word2(
    uint8_t sgpr) {
  if (sgpr >= 2 && (sgpr % 4u) == 2u)
    return static_cast<uint8_t>(sgpr - 2u);
  return std::nullopt;
}

[[nodiscard]] std::optional<uint8_t> raw_buffer_resource_base_for_descriptor_config(
    uint8_t sgpr) {
  if (sgpr >= 3 && (sgpr % 4u) == 3u)
    return static_cast<uint8_t>(sgpr - 3u);
  return std::nullopt;
}

[[nodiscard]] std::optional<uint32_t> rdna4_range_from_gfx1250_descriptor_word2_units(
    uint64_t units) {
  if (units == 0)
    return kRdna4RawBufferUnboundedRange;
  if (units >= std::numeric_limits<uint32_t>::max() / 128u)
    return kRdna4RawBufferUnboundedRange;
  const uint64_t inclusive_last_block_bytes = (units + 1u) * 128u;
  if (units <= 16u)
    return static_cast<uint32_t>(std::max(units * 256u, inclusive_last_block_bytes));
  return static_cast<uint32_t>(inclusive_last_block_bytes);
}

[[nodiscard]] std::optional<uint32_t> rdna4_range_from_gfx1250_descriptor_word2_src(
    uint16_t src) {
  if (src < scalar_positive_inline_u32(0) || src > scalar_positive_inline_u32(64))
    return std::nullopt;

  return rdna4_range_from_gfx1250_descriptor_word2_units(src - scalar_positive_inline_u32(0));
}

[[nodiscard]] std::optional<uint32_t> rdna4_range_from_gfx1250_descriptor_word2_mov(
    uint16_t src, std::optional<uint32_t> literal32, std::optional<uint64_t> literal64) {
  if (literal64)
    return rdna4_range_from_gfx1250_descriptor_word2_units(*literal64);
  if (literal32)
    return rdna4_range_from_gfx1250_descriptor_word2_units(*literal32);
  return rdna4_range_from_gfx1250_descriptor_word2_src(src);
}

[[nodiscard]] bool gfx1250_descriptor_word2_mov_is_statically_nonzero(
    uint16_t src, std::optional<uint32_t> literal32, std::optional<uint64_t> literal64) {
  if (literal64)
    return *literal64 != 0;
  if (literal32)
    return *literal32 != 0;
  if (src >= scalar_positive_inline_u32(0) && src <= scalar_positive_inline_u32(64))
    return src != scalar_positive_inline_u32(0);
  return false;
}

[[nodiscard]] bool is_raw_buffer_descriptor_word2_sgpr_src(uint16_t src) {
  if (src >= 128u)
    return false;
  return raw_buffer_resource_base_for_descriptor_word2(static_cast<uint8_t>(src)).has_value();
}

[[nodiscard]] bool defines_sgpr(const Instruction &inst, uint8_t sgpr) {
  for (int i = 0; i < inst.num_dst_operands(); ++i) {
    const Operand *operand = inst.dst_operand(i);
    if (!operand)
      continue;
    const auto reg = operand->to_register_ref();
    if (!reg || reg->cls != RegClass::SGPR)
      continue;
    if (reg->index <= sgpr && sgpr < reg->index + reg->width)
      return true;
  }

  RegisterSet implicit_defs;
  inst.implicit_defs(implicit_defs);
  return implicit_defs.contains(RegisterRef{RegClass::SGPR, sgpr, 1});
}

[[nodiscard]] bool uses_sgpr(const Instruction &inst, uint8_t sgpr) {
  for (int i = 0; i < inst.num_src_operands(); ++i) {
    const Operand *operand = inst.src_operand(i);
    if (!operand)
      continue;
    const auto reg = operand->to_register_ref();
    if (!reg || reg->cls != RegClass::SGPR)
      continue;
    if (reg->index <= sgpr && sgpr < reg->index + reg->width)
      return true;
  }

  RegisterSet implicit_uses;
  inst.implicit_uses(implicit_uses);
  if (implicit_uses.contains(RegisterRef{RegClass::SGPR, sgpr, 1}))
    return true;

  const uint32_t *raw = inst.raw_encoding();
  if (!raw)
    return false;
  if (is_sop2_encoding(inst)) {
    const auto src = std::bit_cast<gfx1250::Sop2MachineInst>(raw[0]);
    const auto overlaps_scalar_src = [&inst, sgpr](uint16_t ssrc, uint8_t operand_index) {
      if (ssrc >= 128u)
        return false;
      const Operand *operand = inst.src_operand(operand_index);
      const auto width = static_cast<uint8_t>(
          std::max(1, operand ? operand->size_bits() / 32 : 1));
      return ssrc <= sgpr && sgpr < ssrc + width;
    };
    return overlaps_scalar_src(src.ssrc0, 0) || overlaps_scalar_src(src.ssrc1, 1);
  }
  return false;
}

[[nodiscard]] std::optional<uint32_t> vbuffer_access_size_bytes(uint32_t op) {
  switch (op) {
  case 16: // buffer_load_u8
  case 17: // buffer_load_i8
    return 1;
  case 18: // buffer_load_u16
  case 25: // buffer_store_b16
    return 2;
  case 20: // buffer_load_b32
  case 26: // buffer_store_b32
    return 4;
  case 21: // buffer_load_b64
  case 27: // buffer_store_b64
    return 8;
  case 23: // buffer_load_b128
  case 29: // buffer_store_b128
    return 16;
  case 24: // buffer_store_b8
    return 1;
  default:
    return std::nullopt;
  }
}

[[nodiscard]] std::optional<uint32_t> gfx1250_vbuffer_access_size_bytes(
    const Instruction &inst, uint8_t resource_base) {
  if (!inst.mnemonic().starts_with("buffer_"))
    return std::nullopt;
  if ((inst.encoding_id() & 0x1FCu) != kEnc_VBUFFER ||
      inst.size() < static_cast<int>(3 * sizeof(uint32_t)))
    return std::nullopt;
  const uint32_t *raw = inst.raw_encoding();
  if (!raw)
    return std::nullopt;
  if (((raw[0] >> 24u) & 0xFFu) == 0xEEu)
    return std::nullopt;

  const auto fields = gfx1250_to_rdna4::decode_vbuffer_gfx1250(raw[0], raw[1], raw[2]);
  if (fields.rsrc != resource_base)
    return std::nullopt;
  return vbuffer_access_size_bytes(fields.op);
}

enum class DescriptorUseScanResult : uint8_t {
  FoundUse,
  UnsafeUse,
  Defined,
  ReachedEnd,
  Terminated,
};

[[nodiscard]] bool defines_any_sgpr(const Instruction &inst, std::span<const uint8_t> sgprs) {
  for (const uint8_t sgpr : sgprs) {
    if (defines_sgpr(inst, sgpr))
      return true;
  }
  return false;
}

[[nodiscard]] bool uses_any_sgpr(const Instruction &inst, std::span<const uint8_t> sgprs) {
  for (const uint8_t sgpr : sgprs) {
    if (uses_sgpr(inst, sgpr))
      return true;
  }
  return false;
}

[[nodiscard]] bool gfx1250_mov_src_is_zero(uint16_t src, std::optional<uint32_t> literal32,
                                           std::optional<uint64_t> literal64) {
  if (literal64)
    return *literal64 == 0;
  if (literal32)
    return *literal32 == 0;
  return src == scalar_positive_inline_u32(0);
}

[[nodiscard]] bool nearest_gfx1250_sgpr_def_is_zero_mov(InstructionList::Iterator block_begin,
                                                        InstructionList::Iterator inst_it,
                                                        uint8_t sgpr) {
  for (auto scan_it = inst_it; scan_it != block_begin;) {
    --scan_it;
    const Instruction &prev = *scan_it;
    if (!defines_sgpr(prev, sgpr))
      continue;

    if (const auto mov = decode_gfx1250_s_mov_b32(prev)) {
      return mov->sdst == sgpr &&
             gfx1250_mov_src_is_zero(mov->ssrc0, mov->literal32, mov->literal64);
    }
    if (const auto mov = decode_gfx1250_s_mov_b64(prev)) {
      return mov->sdst <= sgpr && sgpr < mov->sdst + 2u &&
             gfx1250_mov_src_is_zero(mov->ssrc0, mov->literal32, mov->literal64);
    }
    return false;
  }
  return false;
}

[[nodiscard]] bool zero_sgpr_was_used_as_descriptor_word2_after_def(
    InstructionList::Iterator block_begin, InstructionList::Iterator inst_it, uint8_t sgpr) {
  const auto resource_base = raw_buffer_resource_base_for_descriptor_word2(sgpr);
  if (!resource_base)
    return false;

  for (auto scan_it = inst_it; scan_it != block_begin;) {
    --scan_it;
    const Instruction &prev = *scan_it;
    if (!defines_sgpr(prev, sgpr))
      continue;

    bool zero_def = false;
    if (const auto mov = decode_gfx1250_s_mov_b32(prev)) {
      zero_def = mov->sdst == sgpr &&
                 gfx1250_mov_src_is_zero(mov->ssrc0, mov->literal32, mov->literal64);
    } else if (const auto mov = decode_gfx1250_s_mov_b64(prev)) {
      zero_def = mov->sdst <= sgpr && sgpr < mov->sdst + 2u &&
                 gfx1250_mov_src_is_zero(mov->ssrc0, mov->literal32, mov->literal64);
    }
    if (!zero_def)
      return false;

    for (auto use_it = scan_it;;) {
      ++use_it;
      if (use_it == inst_it)
        return false;
      if (defines_sgpr(*use_it, sgpr))
        return false;
      if (gfx1250_vbuffer_access_size_bytes(*use_it, *resource_base))
        return true;
    }
  }
  return false;
}

[[nodiscard]] bool zero_sgpr_was_used_as_descriptor_word2_in_scope(
    std::span<BasicBlock *const> scope_blocks, uint64_t inst_offset, uint8_t sgpr) {
  const auto resource_base = raw_buffer_resource_base_for_descriptor_word2(sgpr);
  if (!resource_base)
    return false;

  std::optional<uint64_t> def_offset;
  bool zero_def = false;
  for (BasicBlock *block : scope_blocks) {
    if (!block)
      continue;
    uint64_t offset = block->start_offset();
    for (const Instruction &candidate : block->instructions()) {
      if (offset >= inst_offset)
        break;
      if (defines_sgpr(candidate, sgpr) && (!def_offset || offset > *def_offset)) {
        def_offset = offset;
        zero_def = false;
        if (const auto mov = decode_gfx1250_s_mov_b32(candidate)) {
          zero_def = mov->sdst == sgpr &&
                     gfx1250_mov_src_is_zero(mov->ssrc0, mov->literal32, mov->literal64);
        } else if (const auto mov = decode_gfx1250_s_mov_b64(candidate)) {
          zero_def = mov->sdst <= sgpr && sgpr < mov->sdst + 2u &&
                     gfx1250_mov_src_is_zero(mov->ssrc0, mov->literal32, mov->literal64);
        }
      }
      offset += candidate.size();
    }
  }
  if (!def_offset || !zero_def)
    return false;

  for (BasicBlock *block : scope_blocks) {
    if (!block)
      continue;
    uint64_t offset = block->start_offset();
    for (const Instruction &candidate : block->instructions()) {
      if (offset > *def_offset && offset < inst_offset &&
          gfx1250_vbuffer_access_size_bytes(candidate, *resource_base)) {
        return true;
      }
      offset += candidate.size();
    }
  }
  return false;
}

void rewrite_gfx1250_zero_sgpr_v_mov_sources(std::vector<uint32_t> &words,
                                             const Instruction &inst,
                                             InstructionList::Iterator block_begin,
                                             InstructionList::Iterator inst_it) {
  constexpr uint8_t kRdna4VMovB32Op = 1;
  for (uint32_t &word : words) {
    auto vop1 = std::bit_cast<rdna4::Vop1MachineInst>(word);
    if (vop1.encoding != 0x3Fu || vop1.op != kRdna4VMovB32Op || vop1.src0 >= 128u)
      continue;

    const auto sgpr = static_cast<uint8_t>(vop1.src0);
    if (!uses_sgpr(inst, sgpr))
      continue;
    if (!nearest_gfx1250_sgpr_def_is_zero_mov(block_begin, inst_it, sgpr))
      continue;

    vop1.src0 = scalar_positive_inline_u32(0);
    word = std::bit_cast<uint32_t>(vop1);
  }
}

[[nodiscard]] bool is_descriptor_setup_copy_from_tracked_sgpr(
    const Instruction &inst, std::span<const uint8_t> tracked_sgprs);

void rewrite_gfx1250_zero_sgpr_scalar_sources(std::vector<uint32_t> &words,
                                              const Instruction &inst,
                                              InstructionList::Iterator block_begin,
                                              InstructionList::Iterator inst_it,
                                              uint64_t inst_offset,
                                              std::span<BasicBlock *const> scope_blocks) {
  if (words.empty())
    return;

  const auto rewrite_descriptor_zero_src = [&](uint32_t src) -> uint32_t {
    if (src >= 128u)
      return src;
    if (!is_raw_buffer_descriptor_word2_sgpr_src(src))
      return src;

    const auto sgpr = static_cast<uint8_t>(src);
    if (!uses_sgpr(inst, sgpr))
      return src;
    if (is_descriptor_setup_copy_from_tracked_sgpr(inst, std::span<const uint8_t>(&sgpr, 1)))
      return src;
    const bool descriptor_zero_promoted =
        zero_sgpr_was_used_as_descriptor_word2_after_def(block_begin, inst_it, sgpr) ||
        zero_sgpr_was_used_as_descriptor_word2_in_scope(scope_blocks, inst_offset, sgpr);
    if (!descriptor_zero_promoted)
      return src;

    return scalar_positive_inline_u32(0);
  };

  if (is_sop2_encoding(inst)) {
    auto sop2 = std::bit_cast<rdna4::Sop2MachineInst>(words[0]);
    sop2.ssrc0 = rewrite_descriptor_zero_src(sop2.ssrc0);
    sop2.ssrc1 = rewrite_descriptor_zero_src(sop2.ssrc1);
    words[0] = std::bit_cast<uint32_t>(sop2);
    return;
  }

  if (inst.encoding_id() == kEnc_SOPC) {
    auto sopc = std::bit_cast<rdna4::SopcMachineInst>(words[0]);
    sopc.ssrc0 = rewrite_descriptor_zero_src(sopc.ssrc0);
    sopc.ssrc1 = rewrite_descriptor_zero_src(sopc.ssrc1);
    words[0] = std::bit_cast<uint32_t>(sopc);
  }
}

[[nodiscard]] bool is_descriptor_setup_copy_from_tracked_sgpr(
    const Instruction &inst, std::span<const uint8_t> tracked_sgprs) {
  const auto mov = decode_gfx1250_s_mov_b32(inst);
  if (!mov || mov->ssrc0 >= 128u)
    return false;

  const auto src = static_cast<uint8_t>(mov->ssrc0);
  const auto src_overlaps_tracked = [src](uint8_t sgpr) {
    return src <= sgpr && sgpr <= src + 1u;
  };
  if (!std::ranges::any_of(tracked_sgprs, src_overlaps_tracked))
    return false;

  return raw_buffer_resource_base_for_descriptor_word2(mov->sdst).has_value() ||
         raw_buffer_resource_base_for_descriptor_config(mov->sdst).has_value();
}

[[nodiscard]] bool is_zero_scalar_src(const Instruction &inst, uint16_t src,
                                      uint8_t operand_index) {
  if (src == scalar_positive_inline_u32(0))
    return true;

  const Operand *operand = inst.src_operand(operand_index);
  if (!operand)
    return false;
  if (src == 255u)
    return static_cast<uint32_t>(operand->encoding_value()) == 0;
  if (src == 254u) {
    const auto literal64 = operand->literal64_value();
    return literal64 && *literal64 == 0;
  }
  return false;
}

[[nodiscard]] bool is_allowed_nonzero_descriptor_zero_compare(
    const Instruction &inst, std::span<const uint8_t> zero_compare_safe_sgprs) {
  constexpr uint8_t kSCmpEqU32 = 6;
  constexpr uint8_t kSCmpLgU32 = 7;
  if (inst.encoding_id() != kEnc_SOPC ||
      (inst.opcode() != kSCmpEqU32 && inst.opcode() != kSCmpLgU32))
    return false;

  const uint32_t *raw = inst.raw_encoding();
  if (!raw)
    return false;
  const auto sopc = std::bit_cast<gfx1250::SopcMachineInst>(raw[0]);
  const auto is_safe_descriptor_word2 = [zero_compare_safe_sgprs](uint16_t src) {
    if (src >= 128u)
      return false;
    const auto sgpr = static_cast<uint8_t>(src);
    if (!raw_buffer_resource_base_for_descriptor_word2(sgpr))
      return false;
    return std::ranges::find(zero_compare_safe_sgprs, sgpr) != zero_compare_safe_sgprs.end();
  };

  return (is_safe_descriptor_word2(sopc.ssrc0) && is_zero_scalar_src(inst, sopc.ssrc1, 1)) ||
         (is_safe_descriptor_word2(sopc.ssrc1) && is_zero_scalar_src(inst, sopc.ssrc0, 0));
}

[[nodiscard]] bool has_no_fallthrough(const Instruction &inst) {
  return (inst.flags() & (BRANCH | INDIRECT_BRANCH | PROGRAM_TERMINATOR)) &&
         !(inst.flags() & COND_BRANCH);
}

[[nodiscard]] bool is_direct_conditional_branch(const Instruction &inst) {
  return (inst.flags() & COND_BRANCH) && inst.branch_offset_bytes().has_value();
}

[[nodiscard]] DescriptorUseScanResult scan_vbuffer_use_before_any_def(
    InstructionList::Iterator scan_it, InstructionList::Iterator end, uint8_t resource_base,
    std::span<const uint8_t> tracked_sgprs,
    std::span<const uint8_t> zero_compare_safe_sgprs, size_t &scanned) {
  for (; scan_it != end && scanned < kGfx1250DescriptorUseScanLimit; ++scan_it, ++scanned) {
    const Instruction &future = *scan_it;
    if (auto access_size = gfx1250_vbuffer_access_size_bytes(future, resource_base))
      return DescriptorUseScanResult::FoundUse;
    if (uses_any_sgpr(future, tracked_sgprs) &&
        !is_direct_conditional_branch(future) &&
        !is_descriptor_setup_copy_from_tracked_sgpr(future, tracked_sgprs) &&
        !is_allowed_nonzero_descriptor_zero_compare(future, zero_compare_safe_sgprs))
      return DescriptorUseScanResult::UnsafeUse;
    if (defines_any_sgpr(future, tracked_sgprs))
      return DescriptorUseScanResult::Defined;
    if (has_no_fallthrough(future))
      return DescriptorUseScanResult::Terminated;
  }
  return scanned < kGfx1250DescriptorUseScanLimit ? DescriptorUseScanResult::ReachedEnd
                                                  : DescriptorUseScanResult::UnsafeUse;
}

[[nodiscard]] bool successor_vbuffer_uses_resource_before_any_def(
    const BasicBlock *block, uint8_t resource_base, std::span<const uint8_t> tracked_sgprs,
    std::span<const uint8_t> zero_compare_safe_sgprs,
    std::span<BasicBlock *const> scope_blocks, size_t scanned_before_successors) {
  if (!block)
    return false;

  std::vector<std::pair<BasicBlock *, size_t>> worklist;
  worklist.reserve(block->successors().size() + 1);
  const auto enqueue = [&](BasicBlock *successor) {
    if (!successor)
      return;
    if (std::ranges::none_of(worklist, [successor](const auto &entry) {
          return entry.first == successor;
        })) {
      worklist.emplace_back(successor, scanned_before_successors);
    }
  };
  for (BasicBlock *successor : block->successors())
    enqueue(successor);

  const Instruction *terminator = block->terminator();
  if ((!terminator || !has_no_fallthrough(*terminator)) && !scope_blocks.empty()) {
    const uint64_t fallthrough_offset = block->end_offset();
    const auto fallthrough = std::ranges::find_if(scope_blocks, [&](const BasicBlock *candidate) {
      return candidate && candidate->start_offset() == fallthrough_offset;
    });
    if (fallthrough != scope_blocks.end())
      enqueue(*fallthrough);
  }

  std::unordered_set<const BasicBlock *> visited;
  bool found_use = false;
  while (!worklist.empty()) {
    const auto [successor, scanned_so_far] = worklist.back();
    worklist.pop_back();
    if (!successor || !visited.insert(successor).second)
      continue;

    size_t scanned = scanned_so_far;
    auto result = scan_vbuffer_use_before_any_def(successor->instructions().begin(),
                                                  successor->instructions().end(), resource_base,
                                                  tracked_sgprs, zero_compare_safe_sgprs,
                                                  scanned);
    if (result == DescriptorUseScanResult::FoundUse) {
      found_use = true;
      continue;
    }
    if (result == DescriptorUseScanResult::UnsafeUse)
      return false;
    if (result == DescriptorUseScanResult::Defined || result == DescriptorUseScanResult::Terminated)
      continue;

    for (BasicBlock *next : successor->successors())
      worklist.emplace_back(next, scanned);
  }
  return found_use;
}

[[nodiscard]] bool future_vbuffer_uses_resource_before_any_def(
    InstructionList::Iterator inst_it, InstructionList::Iterator end, uint8_t resource_base,
    std::span<const uint8_t> tracked_sgprs,
    std::span<const uint8_t> zero_compare_safe_sgprs = {},
    const BasicBlock *block = nullptr, std::span<BasicBlock *const> scope_blocks = {}) {
  auto scan_it = inst_it;
  ++scan_it;
  size_t scanned = 0;
  auto result = scan_vbuffer_use_before_any_def(scan_it, end, resource_base, tracked_sgprs,
                                                zero_compare_safe_sgprs, scanned);
  if (result == DescriptorUseScanResult::FoundUse)
    return true;
  if (result == DescriptorUseScanResult::UnsafeUse || result == DescriptorUseScanResult::Defined)
    return false;
  return successor_vbuffer_uses_resource_before_any_def(block, resource_base, tracked_sgprs,
                                                        zero_compare_safe_sgprs, scope_blocks,
                                                        scanned);
}

[[nodiscard]] bool future_vbuffer_uses_resource_before_def(
    InstructionList::Iterator inst_it, InstructionList::Iterator end, uint8_t resource_base,
    uint8_t tracked_sgpr, const BasicBlock *block = nullptr,
    std::span<BasicBlock *const> scope_blocks = {}, bool allow_zero_compare = false) {
  const std::span<const uint8_t> zero_compare_safe_sgprs =
      allow_zero_compare ? std::span<const uint8_t>(&tracked_sgpr, 1)
                         : std::span<const uint8_t>();
  return future_vbuffer_uses_resource_before_any_def(
      inst_it, end, resource_base, std::span<const uint8_t>(&tracked_sgpr, 1),
      zero_compare_safe_sgprs, block, scope_blocks);
}

[[nodiscard]] std::optional<uint8_t> raw_buffer_resource_base_for_descriptor_base(
    uint8_t sgpr) {
  if ((sgpr % 4u) == 0u)
    return sgpr;
  return std::nullopt;
}

[[nodiscard]] bool is_gfx1250_flat_address_alignment_low_mask(uint32_t mask) {
  return mask == 0xFFFF'E000u || mask == 0xFFFF'C000u;
}

[[nodiscard]] bool is_gfx1250_flat_address_alignment_high_mask(uint32_t mask) {
  return mask == 0x7FFF'FFFFu || mask == 0x1FFF'FFFFu;
}

[[nodiscard]] std::vector<uint32_t> lower_gfx1250_contextual_s_and_b32_address_mask_high(
    InstructionList::Iterator block_begin, InstructionList::Iterator inst_it,
    rj_code_arch_t host_arch) {
  const auto high = decode_gfx1250_sop2_literal32(*inst_it, kGfx1250SOpAndB32);
  if (!high || !is_gfx1250_flat_address_alignment_high_mask(high->literal) ||
      inst_it == block_begin)
    return {};

  auto prev_it = inst_it;
  --prev_it;
  const auto low = decode_gfx1250_sop2_literal32(*prev_it, kGfx1250SOpAndB32);
  if (!low || !is_gfx1250_flat_address_alignment_low_mask(low->literal) ||
      low->sdst + 1u != high->sdst || low->non_literal_src + 1u != high->non_literal_src)
    return {};

  return {build_s_mov_b32(high->sdst, scalar_positive_inline_u32(0), host_arch),
          build_s_nop(0, host_arch)};
}

[[nodiscard]] std::optional<uint32_t> descriptor_range_from_gfx1250_high_literal(
    uint32_t literal_hi) {
  if ((literal_hi & 0xFFFF0000u) == 0)
    return std::nullopt;
  return literal_hi >> 25u;
}

[[nodiscard]] std::vector<uint32_t> lower_gfx1250_contextual_raw_buffer_descriptor_base(
    InstructionList::Iterator inst_it, InstructionList::Iterator end,
    rj_code_arch_t host_arch, const BasicBlock *block = nullptr,
    std::span<BasicBlock *const> scope_blocks = {}) {
  if (const auto or32 = decode_gfx1250_sop2_literal32(*inst_it, kGfx1250SOpOrB32)) {
    if (or32->sdst == 0)
      return {};
    const auto range = descriptor_range_from_gfx1250_high_literal(or32->literal);
    const auto resource_base =
        raw_buffer_resource_base_for_descriptor_base(static_cast<uint8_t>(or32->sdst - 1u));
    if (!range || !resource_base ||
        !future_vbuffer_uses_resource_before_def(inst_it, end, *resource_base, or32->sdst, block,
                                                 scope_blocks))
      return {};
    return {pack_sop2(kGfx1250SOpAndB32, or32->sdst, or32->non_literal_src, 255), 0xFFFFu};
  }

  if (const auto or64 = decode_gfx1250_sop2_literal64(*inst_it, kGfx1250SOpOrB64)) {
    const uint32_t literal_lo = static_cast<uint32_t>(or64->literal);
    const uint32_t literal_hi = static_cast<uint32_t>(or64->literal >> 32u);
    const auto range = descriptor_range_from_gfx1250_high_literal(literal_hi);
    const auto resource_base = raw_buffer_resource_base_for_descriptor_base(or64->sdst);
    if (literal_lo != 0 || !range || !resource_base || or64->non_literal_src >= 123u ||
        !future_vbuffer_uses_resource_before_def(inst_it, end, *resource_base,
                                                 static_cast<uint8_t>(or64->sdst + 1u), block,
                                                 scope_blocks))
      return {};
    return {build_s_mov_b32(or64->sdst, or64->non_literal_src, host_arch),
            pack_sop2(kGfx1250SOpAndB32, static_cast<uint8_t>(or64->sdst + 1u),
                      static_cast<uint16_t>(or64->non_literal_src + 1u), 255),
            0xFFFFu};
  }

  return {};
}

[[nodiscard]] std::vector<uint32_t> lower_gfx1250_contextual_raw_buffer_descriptor_mov(
    InstructionList::Iterator inst_it, InstructionList::Iterator end,
    rj_code_arch_t host_arch, const BasicBlock *block = nullptr,
    std::span<BasicBlock *const> scope_blocks = {}) {
  if (const auto mov64 = decode_gfx1250_s_mov_b64(*inst_it)) {
    if (const auto resource_base = raw_buffer_resource_base_for_descriptor_word2(mov64->sdst)) {
      const std::array<uint8_t, 2> tracked_sgprs = {
          mov64->sdst, static_cast<uint8_t>(mov64->sdst + 1u)};
      const std::span<const uint8_t> zero_compare_safe_sgprs =
          gfx1250_descriptor_word2_mov_is_statically_nonzero(mov64->ssrc0, mov64->literal32,
                                                             mov64->literal64)
              ? std::span<const uint8_t>(tracked_sgprs.data(), 1)
              : std::span<const uint8_t>();
      if (future_vbuffer_uses_resource_before_any_def(inst_it, end, *resource_base, tracked_sgprs,
                                                      zero_compare_safe_sgprs, block,
                                                      scope_blocks)) {
        const uint32_t range =
            rdna4_range_from_gfx1250_descriptor_word2_mov(mov64->ssrc0, mov64->literal32,
                                                          mov64->literal64)
                .value_or(kRdna4RawBufferUnboundedRange);
        return {build_s_mov_b32(mov64->sdst, 255, host_arch), range,
                build_s_mov_b32(static_cast<uint8_t>(mov64->sdst + 1u), 255, host_arch),
                kRdna4RawBufferConfigWord};
      }
    }
  }

  if (const auto movk = decode_gfx1250_s_movk_i32(*inst_it)) {
    if (const auto resource_base = raw_buffer_resource_base_for_descriptor_word2(movk->sdst);
        resource_base && future_vbuffer_uses_resource_before_def(
                             inst_it, end, *resource_base, movk->sdst, block, scope_blocks,
                             movk->simm16 != 0)) {
      const uint32_t range = rdna4_range_from_gfx1250_descriptor_word2_units(movk->simm16)
                                 .value_or(kRdna4RawBufferUnboundedRange);
      return {build_s_mov_b32(movk->sdst, 255, host_arch), range};
    }
  }

  const auto mov = decode_gfx1250_s_mov_b32(*inst_it);
  if (!mov)
    return {};

  if (const auto resource_base = raw_buffer_resource_base_for_descriptor_config(mov->sdst);
      resource_base &&
      future_vbuffer_uses_resource_before_def(inst_it, end, *resource_base, mov->sdst, block,
                                              scope_blocks)) {
    return {build_s_mov_b32(mov->sdst, 255, host_arch), kRdna4RawBufferConfigWord};
  }

  if (const auto resource_base = raw_buffer_resource_base_for_descriptor_word2(mov->sdst);
      resource_base &&
      future_vbuffer_uses_resource_before_def(
          inst_it, end, *resource_base, mov->sdst, block, scope_blocks,
          gfx1250_descriptor_word2_mov_is_statically_nonzero(mov->ssrc0, mov->literal32,
                                                             mov->literal64))) {
    if (is_raw_buffer_descriptor_word2_sgpr_src(mov->ssrc0))
      return {};
    const uint32_t range = rdna4_range_from_gfx1250_descriptor_word2_mov(
                               mov->ssrc0, mov->literal32, mov->literal64)
                               .value_or(kRdna4RawBufferUnboundedRange);
    return {build_s_mov_b32(mov->sdst, 255, host_arch), range};
  }

  const bool zero_or_canonical_zero_copy =
      mov->ssrc0 == scalar_positive_inline_u32(0) || mov->ssrc0 == 2;
  if (!zero_or_canonical_zero_copy)
    return {};

  return {};
}

[[nodiscard]] std::vector<uint32_t> lower_gfx1250_smem_nv_to_rdna4(
    const Instruction &inst, rj_code_arch_t) {
  if (inst.size() != static_cast<int>(2 * sizeof(uint32_t)))
    return {};
  const uint32_t *raw = inst.raw_encoding();
  if (!raw)
    return {};
  if ((raw[0] >> 26u) != 0x3Du)
    return {};

  gfx1250::SmemMachineInst src{};
  std::memcpy(&src, raw, sizeof(src));
  if (src.nv == 0)
    return {};
  src.nv = 0;

  std::array<uint32_t, 2> words{};
  std::memcpy(words.data(), &src, sizeof(src));
  return {words[0], words[1]};
}

[[nodiscard]] std::vector<uint32_t> lower_gfx1250_s_wait_xcnt_to_rdna4(
    const Instruction &inst, rj_code_arch_t host_arch) {
  if (host_arch != ROCJITSU_CODE_ARCH_RDNA4 || inst.encoding_id() != kEnc_SOPP ||
      inst.opcode() != kGfx1250SoppWaitXcnt)
    return {};

  // GFX1250 XCNT tracks memory progress across VMEM, LDS/scratch, and SMEM
  // groups. RDNA4 has no single equivalent counter, so conservatively drain the
  // split counters.
  // For nonzero source waits this is stricter than the original instruction but
  // preserves ordering.
  return {pack_sopp(kSoppWaitLoadcnt, 0), pack_sopp(kSoppWaitStorecnt, 0),
          pack_sopp(kSoppWaitDscnt, 0), pack_sopp(kSoppWaitKmcnt, 0)};
}

[[nodiscard]] constexpr uint16_t build_hwreg(uint8_t reg_id, uint8_t offset, uint8_t size) {
  return static_cast<uint16_t>((reg_id & 0x3Fu) | ((offset & 0x1Fu) << 6) |
                               (((size - 1u) & 0x1Fu) << 11));
}

[[nodiscard]] constexpr uint32_t build_sopk(uint8_t op, uint16_t simm16, uint8_t sdst = 0) {
  return 0xB0000000u | (simm16 & 0xFFFFu) | ((sdst & 0x7Fu) << 16) | ((op & 0x1Fu) << 23);
}

[[nodiscard]] constexpr std::array<uint32_t, 3> build_scratch_store_b32(uint8_t vdata,
                                                                        uint32_t offset) {
  return {0xED06807Cu, static_cast<uint32_t>(vdata) << 23, (offset & 0xFFFFFFu) << 8};
}

[[nodiscard]] constexpr std::array<uint32_t, 3> build_scratch_load_b32(uint8_t vdst,
                                                                       uint32_t offset) {
  return {0xED05007Cu, static_cast<uint32_t>(vdst), (offset & 0xFFFFFFu) << 8};
}

void append_scratch_store_b32(std::vector<uint32_t> &words, uint8_t vdata, uint32_t offset) {
  const auto encoded = build_scratch_store_b32(vdata, offset);
  words.insert(words.end(), encoded.begin(), encoded.end());
}

void append_scratch_load_b32(std::vector<uint32_t> &words, uint8_t vdst, uint32_t offset) {
  const auto encoded = build_scratch_load_b32(vdst, offset);
  words.insert(words.end(), encoded.begin(), encoded.end());
}

[[nodiscard]] constexpr uint16_t vgpr_msb_mode_hwreg() {
  return build_hwreg(1, amdgpu::VGPR_MSB_MODE_SHIFT, 8);
}

void append_raw_s_get_vgpr_msb_mode(std::vector<uint32_t> &words, uint8_t sdst) {
  constexpr uint8_t kOpSGetregB32 = 17;
  words.push_back(build_sopk(kOpSGetregB32, vgpr_msb_mode_hwreg(), sdst));
}

void append_raw_s_set_vgpr_msb_mode_from_sgpr(std::vector<uint32_t> &words, uint8_t ssrc) {
  constexpr uint8_t kOpSSetregB32 = 18;
  words.push_back(build_sopk(kOpSSetregB32, vgpr_msb_mode_hwreg(), ssrc));
}

void append_raw_s_set_vgpr_msb_mode(std::vector<uint32_t> &words, uint8_t mode) {
  constexpr uint8_t kOpSSetregImm32B32 = 19;
  const uint32_t mode_literal = amdgpu::set_vgpr_msb_to_mode_layout(mode);
  words.push_back(build_sopk(kOpSSetregImm32B32, vgpr_msb_mode_hwreg()));
  words.push_back(mode_literal);
}

void grow_required_vgpr_count_for_src(uint32_t &minimum_vgprs, uint16_t src) {
  if (src >= 256u && src < 512u)
    minimum_vgprs = std::max(minimum_vgprs, static_cast<uint32_t>(src - 256u + 1u));
}

[[nodiscard]] constexpr uint16_t scalar_negative_inline_i32(int16_t value) {
  return static_cast<uint16_t>(192 - value);
}

[[nodiscard]] constexpr std::optional<uint16_t> scalar_inline_i32(int32_t value) {
  if (value >= 0 && value <= 64)
    return scalar_positive_inline_u32(static_cast<uint16_t>(value));
  if (value >= -16 && value <= -1)
    return scalar_negative_inline_i32(static_cast<int16_t>(value));
  return std::nullopt;
}

[[nodiscard]] constexpr std::optional<uint8_t> raw_vgpr_index(uint16_t src) {
  if (src < 256u || src >= 512u)
    return std::nullopt;
  return static_cast<uint8_t>(src - 256u);
}

[[nodiscard]] constexpr bool raw_overlaps_vdst_pair(uint8_t vgpr, uint8_t vdst) {
  return vgpr == vdst || vgpr == static_cast<uint8_t>(vdst + 1u);
}

[[nodiscard]] bool raw_overlaps_vgpr_run(uint16_t run_base, uint16_t count, uint8_t vgpr) {
  return vgpr >= run_base && vgpr < run_base + count;
}

void add_raw_avoid_vgpr(std::vector<uint8_t> &avoid, uint8_t vgpr) {
  if (std::find(avoid.begin(), avoid.end(), vgpr) == avoid.end())
    avoid.push_back(vgpr);
}

void add_raw_avoid_vgpr_run(std::vector<uint8_t> &avoid, uint8_t base, uint8_t count) {
  for (uint8_t i = 0; i < count; ++i)
    add_raw_avoid_vgpr(avoid, static_cast<uint8_t>(base + i));
}

void add_raw_avoid_src_vgpr(std::vector<uint8_t> &avoid, uint16_t src) {
  if (auto vgpr = raw_vgpr_index(src))
    add_raw_avoid_vgpr(avoid, *vgpr);
}

[[nodiscard]] std::optional<uint16_t>
raw_pair_hi_src_with_literal(uint16_t src, std::optional<uint32_t> literal) {
  if (src == 255) {
    if (!literal)
      return std::nullopt;
    return scalar_inline_i32(static_cast<int32_t>(*literal) < 0 ? -1 : 0);
  }
  if (src == 254)
    return std::nullopt;
  if (src < 128u || src >= 256u)
    return static_cast<uint16_t>(src + 1u);
  if (src >= scalar_positive_inline_u32(0) && src <= scalar_positive_inline_u32(64))
    return scalar_positive_inline_u32(0);
  if (src >= scalar_negative_inline_i32(-1) && src <= scalar_negative_inline_i32(-16))
    return scalar_negative_inline_i32(-1);
  return std::nullopt;
}

[[nodiscard]] bool raw_source_pair_reads_vdst_pair(uint8_t vdst, uint16_t src_lo, uint16_t src_hi) {
  const auto lo = raw_vgpr_index(src_lo);
  const auto hi = raw_vgpr_index(src_hi);
  return (lo && raw_overlaps_vdst_pair(*lo, vdst)) || (hi && raw_overlaps_vdst_pair(*hi, vdst));
}

void add_raw_avoid_source_pair_vgprs(std::vector<uint8_t> &avoid, uint16_t src_lo,
                                     uint16_t src_hi) {
  add_raw_avoid_src_vgpr(avoid, src_lo);
  add_raw_avoid_src_vgpr(avoid, src_hi);
}

std::optional<uint16_t> find_raw_free_vgpr_run_avoiding(const Instruction &inst,
                                                        const LivenessAnalysis &liveness,
                                                        uint16_t count,
                                                        const std::vector<uint8_t> &avoid) {
  uint16_t search_start = 0;
  while (true) {
    auto tmp_base = liveness.find_free_run(&inst, count, search_start);
    if (!tmp_base || *tmp_base + count - 1u > 255u)
      return std::nullopt;
    bool overlaps = false;
    for (uint8_t vgpr : avoid)
      overlaps |= raw_overlaps_vgpr_run(*tmp_base, count, vgpr);
    if (!overlaps)
      return tmp_base;
    search_start = static_cast<uint16_t>(*tmp_base + 1u);
  }
}

std::optional<uint8_t> find_raw_borrowable_low_vgpr_run(uint8_t count, uint8_t alignment,
                                                        const std::vector<uint8_t> &avoid) {
  for (uint16_t base = 0; base + count <= 128u; ++base) {
    if ((base % alignment) != 0)
      continue;
    bool overlaps = false;
    for (uint8_t vgpr : avoid)
      overlaps |= raw_overlaps_vgpr_run(base, count, vgpr);
    if (!overlaps)
      return static_cast<uint8_t>(base);
  }
  return std::nullopt;
}

std::optional<uint16_t> find_raw_v_mul_u64_low_scratch(uint8_t vdst, uint16_t src0,
                                                       uint16_t src0_hi, uint16_t src1,
                                                       uint16_t src1_hi, const Instruction &inst,
                                                       const LivenessAnalysis &liveness) {
  std::vector<uint8_t> avoid;
  add_raw_avoid_vgpr_run(avoid, vdst, 2);
  add_raw_avoid_source_pair_vgprs(avoid, src0, src0_hi);
  add_raw_avoid_source_pair_vgprs(avoid, src1, src1_hi);
  return find_raw_free_vgpr_run_avoiding(inst, liveness, 2, avoid);
}

void append_raw_vop3(std::vector<uint32_t> &words, uint16_t op, uint8_t vdst, uint16_t src0,
                     uint16_t src1, uint16_t src2 = 0,
                     std::optional<uint32_t> literal = std::nullopt) {
  auto [w0, w1] = build_rdna4_vop3(op, vdst, src0, src1, src2);
  words.push_back(w0);
  words.push_back(w1);
  if (literal && (src0 == 255 || src1 == 255 || src2 == 255))
    words.push_back(*literal);
}

void append_raw_vop3_sdst(std::vector<uint32_t> &words, uint16_t op, uint8_t vdst, uint8_t sdst,
                          uint16_t src0, uint16_t src1, uint16_t src2 = 0,
                          std::optional<uint32_t> literal = std::nullopt) {
  auto [w0, w1] = build_rdna4_vop3_sdst(op, vdst, sdst, src0, src1, src2);
  words.push_back(w0);
  words.push_back(w1);
  if (literal && (src0 == 255 || src1 == 255 || src2 == 255))
    words.push_back(*literal);
}

void append_raw_vop1(std::vector<uint32_t> &words, uint8_t op, uint8_t vdst, uint16_t src0) {
  words.push_back(build_rdna4_vop1(op, vdst, src0));
}

void append_raw_vop2(std::vector<uint32_t> &words, uint8_t op, uint8_t vdst, uint16_t src0,
                     uint8_t vsrc1, std::optional<uint32_t> literal = std::nullopt) {
  words.push_back(build_rdna4_vop2(op, vdst, src0, vsrc1));
  if (literal && src0 == 255)
    words.push_back(*literal);
}

[[nodiscard]] bool starts_with(std::string_view value, std::string_view prefix);
[[nodiscard]] std::vector<uint32_t> nop_words(uint32_t size, rj_code_arch_t host_arch);
void append_raw_v_mul_u64_low64(std::vector<uint32_t> &words, uint8_t out_lo, uint8_t out_hi,
                                uint16_t src0_lo, uint16_t src0_hi, uint16_t src1_lo,
                                uint16_t src1_hi, std::optional<uint32_t> literal);

enum class HighBankRole : uint8_t {
  Src0,
  Src1,
  Src2,
  Dst,
};

struct HighBankShadowPlan {
  uint8_t base = 0;
  uint8_t count = 0;
  bool spill_to_private = false;
};

struct HighBankShadowLowSave {
  uint8_t physical = 0;
  uint8_t slot = 0;
};

struct HighBankShadowState {
  uint8_t mode = 0;
};

enum class HighBankShadowLoweringKind {
  NotApplicable,
  Lowered,
  Unsupported,
};

struct HighBankShadowLowering {
  HighBankShadowLoweringKind kind = HighBankShadowLoweringKind::NotApplicable;
  std::vector<uint32_t> words;
  std::string message;
};

[[nodiscard]] constexpr uint8_t high_bank_role_shift(HighBankRole role) {
  switch (role) {
  case HighBankRole::Src0:
    return 0;
  case HighBankRole::Src1:
    return 2;
  case HighBankRole::Src2:
    return 4;
  case HighBankRole::Dst:
    return 6;
  }
  return 0;
}

[[nodiscard]] constexpr uint8_t high_bank_selector(uint8_t mode, HighBankRole role) {
  return static_cast<uint8_t>((mode >> high_bank_role_shift(role)) & 0x3u);
}

[[nodiscard]] bool is_gfx1250_s_set_vgpr_msb(uint32_t word, uint16_t &simm16) {
  const auto sopp = std::bit_cast<gfx1250::SoppMachineInst>(word);
  constexpr uint8_t kOpSSetVgprMsb = 6;
  if (sopp.encoding != 0x17Fu || sopp.op != kOpSSetVgprMsb)
    return false;
  simm16 = static_cast<uint16_t>(sopp.simm16);
  return true;
}

[[nodiscard]] std::optional<uint8_t> shadow_vgpr(uint8_t logical,
                                                 const std::optional<HighBankShadowPlan> &plan) {
  if (!plan || logical >= plan->count)
    return std::nullopt;
  const uint16_t physical = static_cast<uint16_t>(plan->base) + logical;
  if (physical > 255u)
    return std::nullopt;
  return static_cast<uint8_t>(physical);
}

[[nodiscard]] uint32_t high_bank_shadow_private_scratch_bytes(const HighBankShadowPlan &plan) {
  if (!plan.spill_to_private)
    return 0;
  return kGfx1250K128Fp8PrivateScratchBytes + plan.count * sizeof(uint32_t) +
         kGfx1250HighBankShadowLowSaveBytes;
}

[[nodiscard]] std::optional<uint32_t>
high_bank_shadow_private_base(const LivenessAnalysis &liveness,
                              const std::optional<HighBankShadowPlan> &plan) {
  if (!plan || !plan->spill_to_private)
    return std::nullopt;
  const auto private_base = liveness.private_spill_base();
  if (!private_base)
    return std::nullopt;
  const uint32_t required_bytes = high_bank_shadow_private_scratch_bytes(*plan);
  if (liveness.private_spill_bytes() < required_bytes)
    return std::nullopt;
  return *private_base + kGfx1250K128Fp8PrivateScratchBytes;
}

[[nodiscard]] std::optional<uint32_t>
high_bank_shadow_low_save_base(const LivenessAnalysis &liveness,
                               const std::optional<HighBankShadowPlan> &plan) {
  const auto private_base = high_bank_shadow_private_base(liveness, plan);
  if (!private_base || !plan)
    return std::nullopt;
  return *private_base + plan->count * sizeof(uint32_t);
}

void add_unique_logical_vgpr(std::vector<uint8_t> &regs, uint8_t logical) {
  if (std::find(regs.begin(), regs.end(), logical) == regs.end())
    regs.push_back(logical);
}

[[nodiscard]] bool collect_high_bank_vgpr_spill(std::vector<uint8_t> &regs, uint8_t logical,
                                                uint16_t width, uint8_t mode, HighBankRole role,
                                                const std::optional<HighBankShadowPlan> &plan) {
  if (!plan || !plan->spill_to_private)
    return true;
  const uint8_t selector = high_bank_selector(mode, role);
  if (selector == 0)
    return true;
  if (selector != 1 || width == 0 || logical + width > plan->count)
    return false;
  for (uint16_t i = 0; i < width; ++i)
    add_unique_logical_vgpr(regs, static_cast<uint8_t>(logical + i));
  return true;
}

[[nodiscard]] bool collect_high_bank_src_spill(std::vector<uint8_t> &regs, uint16_t src,
                                               uint16_t width, uint8_t mode, HighBankRole role,
                                               const std::optional<HighBankShadowPlan> &plan) {
  if (!plan || !plan->spill_to_private)
    return true;
  const uint8_t selector = high_bank_selector(mode, role);
  if (selector == 0)
    return true;
  if (selector != 1)
    return false;
  const auto logical = raw_vgpr_index(src);
  if (!logical)
    return true;
  return collect_high_bank_vgpr_spill(regs, *logical, width, mode, role, plan);
}

[[nodiscard]] std::optional<std::vector<uint32_t>>
wrap_high_bank_shadow_private_spills(const Instruction &inst, std::vector<uint32_t> words,
                                     const std::vector<uint8_t> &loads,
                                     const std::vector<uint8_t> &stores,
                                     const LivenessAnalysis &liveness,
                                     const std::optional<HighBankShadowPlan> &plan) {
  if (!plan || !plan->spill_to_private || (loads.empty() && stores.empty()))
    return words;

  const auto private_base = high_bank_shadow_private_base(liveness, plan);
  if (!private_base)
    return std::nullopt;

  std::vector<uint8_t> touched_logicals;
  touched_logicals.reserve(loads.size() + stores.size());
  for (const uint8_t logical : loads)
    add_unique_logical_vgpr(touched_logicals, logical);
  for (const uint8_t logical : stores)
    add_unique_logical_vgpr(touched_logicals, logical);

  const RegisterSet &live = liveness.live_before(inst);
  std::vector<HighBankShadowLowSave> low_saves;
  low_saves.reserve(touched_logicals.size());
  for (const uint8_t logical : touched_logicals) {
    const auto physical = shadow_vgpr(logical, plan);
    if (!physical)
      return std::nullopt;
    if (!live.contains({RegClass::VGPR, *physical, 1}))
      continue;
    if (low_saves.size() >= kGfx1250HighBankShadowLowSaveVgprCount)
      return std::nullopt;
    low_saves.push_back(
        HighBankShadowLowSave{*physical, static_cast<uint8_t>(low_saves.size())});
  }

  const auto low_save_base =
      low_saves.empty() ? std::optional<uint32_t>{}
                        : high_bank_shadow_low_save_base(liveness, plan);
  if (!low_saves.empty() && !low_save_base)
    return std::nullopt;

  std::vector<uint32_t> wrapped;
  wrapped.reserve(low_saves.size() * 6 + loads.size() * 3 + words.size() +
                  stores.size() * 3 + 16);
  if (!low_saves.empty()) {
    wrapped.push_back(build_s_wait_alu(kWaitAluDepctrVaVdst0, ROCJITSU_CODE_ARCH_RDNA4));
    wrapped.push_back(pack_sopp(kSoppWaitLoadcnt, 0));
    wrapped.push_back(pack_sopp(kSoppWaitDscnt, 0));
    for (const HighBankShadowLowSave &save : low_saves) {
      append_scratch_store_b32(wrapped, save.physical,
                               *low_save_base + save.slot * sizeof(uint32_t));
    }
    wrapped.push_back(pack_sopp(kSoppWaitStorecnt, 0));
  }

  for (const uint8_t logical : loads) {
    const auto physical = shadow_vgpr(logical, plan);
    if (!physical)
      return std::nullopt;
    append_scratch_load_b32(wrapped, *physical, *private_base + logical * sizeof(uint32_t));
  }
  if (!loads.empty())
    wrapped.push_back(pack_sopp(kSoppWaitLoadcnt, 0));

  wrapped.insert(wrapped.end(), words.begin(), words.end());

  if (!stores.empty()) {
    wrapped.push_back(build_s_wait_alu(kWaitAluDepctrVaSdst0, ROCJITSU_CODE_ARCH_RDNA4));
    wrapped.push_back(pack_sopp(kSoppWaitLoadcnt, 0));
    wrapped.push_back(pack_sopp(kSoppWaitDscnt, 0));
    for (const uint8_t logical : stores) {
      const auto physical = shadow_vgpr(logical, plan);
      if (!physical)
        return std::nullopt;
      append_scratch_store_b32(wrapped, *physical, *private_base + logical * sizeof(uint32_t));
    }
    wrapped.push_back(pack_sopp(kSoppWaitStorecnt, 0));
  }

  if (!low_saves.empty()) {
    for (const HighBankShadowLowSave &save : low_saves) {
      append_scratch_load_b32(wrapped, save.physical,
                              *low_save_base + save.slot * sizeof(uint32_t));
    }
    wrapped.push_back(pack_sopp(kSoppWaitLoadcnt, 0));
  }

  return wrapped;
}

[[nodiscard]] std::optional<uint8_t>
remap_high_bank_vgpr(uint8_t logical, uint8_t mode, HighBankRole role,
                     const std::optional<HighBankShadowPlan> &plan, bool &changed) {
  const uint8_t selector = high_bank_selector(mode, role);
  if (selector == 0)
    return logical;
  if (selector != 1)
    return std::nullopt;
  auto mapped = shadow_vgpr(logical, plan);
  if (!mapped)
    return std::nullopt;
  changed = true;
  return *mapped;
}

[[nodiscard]] std::optional<uint16_t>
remap_high_bank_src(uint16_t src, uint8_t mode, HighBankRole role,
                    const std::optional<HighBankShadowPlan> &plan, bool &changed) {
  const uint8_t selector = high_bank_selector(mode, role);
  if (selector == 0)
    return src;
  if (selector != 1)
    return std::nullopt;
  auto vgpr = raw_vgpr_index(src);
  if (!vgpr)
    return src;
  auto mapped = shadow_vgpr(*vgpr, plan);
  if (!mapped)
    return std::nullopt;
  changed = true;
  return static_cast<uint16_t>(256u + *mapped);
}

struct RemappedSrcPair {
  uint16_t lo = 0;
  uint16_t hi = 0;
};

[[nodiscard]] std::optional<RemappedSrcPair>
remap_high_bank_src_pair(uint16_t src_lo, std::optional<uint32_t> literal, uint8_t mode,
                         HighBankRole role, const std::optional<HighBankShadowPlan> &plan,
                         bool &changed) {
  auto src_hi = raw_pair_hi_src_with_literal(src_lo, literal);
  if (!src_hi)
    return std::nullopt;
  auto lo = remap_high_bank_src(src_lo, mode, role, plan, changed);
  auto hi = remap_high_bank_src(*src_hi, mode, role, plan, changed);
  if (!lo || !hi)
    return std::nullopt;
  return RemappedSrcPair{*lo, *hi};
}

[[nodiscard]] bool source_pair_overlaps_any_vgpr(const RemappedSrcPair &pair,
                                                 const std::vector<uint8_t> &vgprs) {
  const auto lo = raw_vgpr_index(pair.lo);
  const auto hi = raw_vgpr_index(pair.hi);
  return (lo && std::find(vgprs.begin(), vgprs.end(), *lo) != vgprs.end()) ||
         (hi && std::find(vgprs.begin(), vgprs.end(), *hi) != vgprs.end());
}

[[nodiscard]] std::optional<std::pair<uint8_t, uint8_t>>
private_shadow_source_pair_logicals(uint16_t src_lo, std::optional<uint32_t> literal, uint8_t mode,
                                    HighBankRole role,
                                    const std::optional<HighBankShadowPlan> &plan) {
  if (!plan || !plan->spill_to_private || high_bank_selector(mode, role) != 1)
    return std::nullopt;
  const auto src_hi = raw_pair_hi_src_with_literal(src_lo, literal);
  if (!src_hi)
    return std::nullopt;
  const auto lo = raw_vgpr_index(src_lo);
  const auto hi = raw_vgpr_index(*src_hi);
  if (!lo || !hi)
    return std::nullopt;
  return std::pair<uint8_t, uint8_t>{*lo, *hi};
}

struct ShadowFootprint {
  bool needed = false;
  bool unsupported = false;
  uint8_t max_logical_vgpr = 0;
  std::bitset<256> used_low_vgprs;
  std::vector<const Instruction *> high_mode_insts;
};

void record_high_bank_logical_vgpr(uint16_t logical, uint16_t width, ShadowFootprint &footprint) {
  if (width == 0 || logical + width - 1u > 255u) {
    footprint.unsupported = true;
    return;
  }
  footprint.needed = true;
  footprint.max_logical_vgpr =
      std::max<uint8_t>(footprint.max_logical_vgpr, static_cast<uint8_t>(logical + width - 1u));
}

void record_high_bank_src(uint16_t src, uint16_t width, uint8_t mode, HighBankRole role,
                          ShadowFootprint &footprint) {
  const uint8_t selector = high_bank_selector(mode, role);
  if (selector == 0)
    return;
  if (selector != 1) {
    footprint.unsupported = true;
    return;
  }
  if (auto vgpr = raw_vgpr_index(src))
    record_high_bank_logical_vgpr(*vgpr, width, footprint);
}

void record_high_bank_vgpr(uint8_t logical, uint16_t width, uint8_t mode, HighBankRole role,
                           ShadowFootprint &footprint) {
  const uint8_t selector = high_bank_selector(mode, role);
  if (selector == 0)
    return;
  if (selector != 1) {
    footprint.unsupported = true;
    return;
  }
  record_high_bank_logical_vgpr(logical, width, footprint);
}

struct DsHighBankOperands {
  bool recognized = false;
  bool uses_addr = false;
  uint16_t vdst_width = 0;
  uint16_t data0_width = 0;
};

[[nodiscard]] uint16_t ds_vgpr_width_from_mnemonic(std::string_view mnemonic) {
  if (mnemonic.find("_b128") != std::string_view::npos)
    return 4;
  if (mnemonic.find("_b96") != std::string_view::npos)
    return 3;
  if (mnemonic.find("_b64") != std::string_view::npos)
    return 2;
  return 1;
}

[[nodiscard]] DsHighBankOperands describe_high_bank_ds_operands(std::string_view mnemonic) {
  DsHighBankOperands ops;
  if (starts_with(mnemonic, "ds_load_b") || starts_with(mnemonic, "ds_load_i") ||
      starts_with(mnemonic, "ds_load_u")) {
    ops.recognized = true;
    ops.uses_addr = true;
    ops.vdst_width = ds_vgpr_width_from_mnemonic(mnemonic);
    return ops;
  }
  if (starts_with(mnemonic, "ds_store_b")) {
    ops.recognized = true;
    ops.uses_addr = true;
    ops.data0_width = ds_vgpr_width_from_mnemonic(mnemonic);
    return ops;
  }
  if (mnemonic == "ds_permute_b32" || mnemonic == "ds_bpermute_b32") {
    ops.recognized = true;
    ops.uses_addr = true;
    ops.vdst_width = 1;
    ops.data0_width = 1;
    return ops;
  }
  return ops;
}

void record_high_bank_vop_footprint(const Instruction &inst, uint8_t mode,
                                    ShadowFootprint &footprint) {
  if (mode == 0)
    return;
  const std::string_view mnemonic = inst.mnemonic();
  if (starts_with(mnemonic, "v_") && !starts_with(mnemonic, "v_nop"))
    footprint.high_mode_insts.push_back(&inst);
  if (starts_with(mnemonic, "ds_"))
    footprint.high_mode_insts.push_back(&inst);
  const uint32_t *raw = inst.raw_encoding();
  if (raw == nullptr || inst.size() < static_cast<int>(sizeof(uint32_t)))
    return;

  const uint32_t w0 = raw[0];
  if (((w0 >> 25) & 0x3Fu) == kGfx1250VAddF16E32Opcode &&
      inst.size() == sizeof(uint32_t)) {
    footprint.high_mode_insts.push_back(&inst);
    const auto op = std::bit_cast<gfx1250::Vop2MachineInst>(w0);
    record_high_bank_vgpr(static_cast<uint8_t>(op.vdst), 1, mode, HighBankRole::Dst, footprint);
    record_high_bank_src(static_cast<uint16_t>(op.src0), 1, mode, HighBankRole::Src0, footprint);
    record_high_bank_vgpr(static_cast<uint8_t>(op.vsrc1), 1, mode, HighBankRole::Src1, footprint);
    return;
  }
  if (starts_with(mnemonic, "ds_")) {
    const DsHighBankOperands ops = describe_high_bank_ds_operands(mnemonic);
    if (!ops.recognized || inst.size() < static_cast<int>(sizeof(gfx1250::VdsMachineInst))) {
      footprint.unsupported = true;
      return;
    }
    gfx1250::VdsMachineInst op{};
    std::memcpy(&op, raw, sizeof(op));
    if (ops.uses_addr)
      record_high_bank_vgpr(static_cast<uint8_t>(op.addr), 1, mode, HighBankRole::Src0, footprint);
    if (ops.vdst_width != 0)
      record_high_bank_vgpr(static_cast<uint8_t>(op.vdst), ops.vdst_width, mode, HighBankRole::Dst,
                            footprint);
    if (ops.data0_width != 0)
      record_high_bank_vgpr(static_cast<uint8_t>(op.data0), ops.data0_width, mode,
                            HighBankRole::Src1, footprint);
    return;
  }
  if (mnemonic == "v_cvt_u32_f32_e32" || mnemonic == "v_mov_b32_e32") {
    const auto op = std::bit_cast<gfx1250::Vop1MachineInst>(w0);
    record_high_bank_vgpr(static_cast<uint8_t>(op.vdst), 1, mode, HighBankRole::Dst, footprint);
    record_high_bank_src(static_cast<uint16_t>(op.src0), 1, mode, HighBankRole::Src0, footprint);
    return;
  }
  if (starts_with(mnemonic, "v_readfirstlane_b32")) {
    const auto op = std::bit_cast<gfx1250::Vop1MachineInst>(w0);
    record_high_bank_src(static_cast<uint16_t>(op.src0), 1, mode, HighBankRole::Src0, footprint);
    return;
  }
  if (mnemonic == "v_mov_b64_e32") {
    const auto op = std::bit_cast<gfx1250::Vop1MachineInst>(w0);
    record_high_bank_vgpr(static_cast<uint8_t>(op.vdst), 2, mode, HighBankRole::Dst, footprint);
    record_high_bank_src(static_cast<uint16_t>(op.src0), 2, mode, HighBankRole::Src0, footprint);
    return;
  }
  if (mnemonic == "v_add_nc_u64_e32" || mnemonic == "v_sub_nc_u64_e32" ||
      mnemonic == "v_mul_u64_e32" || mnemonic == "v_lshlrev_b64_e32") {
    const auto op = std::bit_cast<gfx1250::Vop2MachineInst>(w0);
    record_high_bank_vgpr(static_cast<uint8_t>(op.vdst), 2, mode, HighBankRole::Dst, footprint);
    record_high_bank_src(static_cast<uint16_t>(op.src0), 2, mode, HighBankRole::Src0, footprint);
    record_high_bank_vgpr(static_cast<uint8_t>(op.vsrc1), 2, mode, HighBankRole::Src1, footprint);
    return;
  }
  if (mnemonic == "v_or_b32_e32" || mnemonic == "v_add_f16_e32" ||
      (((w0 >> 25) & 0x3Fu) == kGfx1250VAddF16E32Opcode &&
       inst.size() == sizeof(uint32_t))) {
    const auto op = std::bit_cast<gfx1250::Vop2MachineInst>(w0);
    record_high_bank_vgpr(static_cast<uint8_t>(op.vdst), 1, mode, HighBankRole::Dst, footprint);
    record_high_bank_src(static_cast<uint16_t>(op.src0), 1, mode, HighBankRole::Src0, footprint);
    record_high_bank_vgpr(static_cast<uint8_t>(op.vsrc1), 1, mode, HighBankRole::Src1, footprint);
    return;
  }
  if (starts_with(mnemonic, "v_cmp") && (mnemonic.find("_u64_e32") != std::string_view::npos ||
                                         mnemonic.find("_i64_e32") != std::string_view::npos)) {
    const auto op = std::bit_cast<gfx1250::VopcMachineInst>(w0);
    record_high_bank_src(static_cast<uint16_t>(op.src0), 2, mode, HighBankRole::Src0, footprint);
    record_high_bank_vgpr(static_cast<uint8_t>(op.vsrc1), 2, mode, HighBankRole::Src1, footprint);
    return;
  }
  if (inst.size() >= static_cast<int>(2 * sizeof(uint32_t)) && (w0 >> 26) == kVop3Encoding) {
    const uint16_t op = static_cast<uint16_t>((w0 >> 16) & 0x3FFu);
    const uint8_t vdst = static_cast<uint8_t>(w0 & 0xFFu);
    const uint32_t w1 = raw[1];
    const uint16_t src0 = static_cast<uint16_t>(w1 & 0x1FFu);
    const uint16_t src1 = static_cast<uint16_t>((w1 >> 9) & 0x1FFu);
    const uint16_t src2 = static_cast<uint16_t>((w1 >> 18) & 0x1FFu);
    if (op == 594u) {
      record_high_bank_vgpr(vdst, 2, mode, HighBankRole::Dst, footprint);
      record_high_bank_src(src0, 2, mode, HighBankRole::Src0, footprint);
      record_high_bank_src(src1, 1, mode, HighBankRole::Src1, footprint);
      record_high_bank_src(src2, 2, mode, HighBankRole::Src2, footprint);
      return;
    }
    if (op == 762u) {
      record_high_bank_vgpr(vdst, 2, mode, HighBankRole::Dst, footprint);
      record_high_bank_src(src0, 1, mode, HighBankRole::Src0, footprint);
      record_high_bank_src(src1, 1, mode, HighBankRole::Src1, footprint);
      record_high_bank_src(src2, 2, mode, HighBankRole::Src2, footprint);
      return;
    }
    if (op == 597u) {
      record_high_bank_vgpr(vdst, 1, mode, HighBankRole::Dst, footprint);
      record_high_bank_src(src0, 1, mode, HighBankRole::Src0, footprint);
      record_high_bank_src(src1, 1, mode, HighBankRole::Src1, footprint);
      record_high_bank_src(src2, 1, mode, HighBankRole::Src2, footprint);
      return;
    }
    if (op == 599u || op == 600u) {
      record_high_bank_vgpr(vdst, 1, mode, HighBankRole::Dst, footprint);
      record_high_bank_src(src0, 1, mode, HighBankRole::Src0, footprint);
      record_high_bank_src(src1, 1, mode, HighBankRole::Src1, footprint);
      record_high_bank_src(src2, 1, mode, HighBankRole::Src2, footprint);
      return;
    }
    if (op == 812u || op == 813u) {
      record_high_bank_vgpr(vdst, 1, mode, HighBankRole::Dst, footprint);
      record_high_bank_src(src0, 1, mode, HighBankRole::Src0, footprint);
      record_high_bank_src(src1, 1, mode, HighBankRole::Src1, footprint);
      return;
    }
  }

  if (starts_with(mnemonic, "v_"))
    footprint.unsupported = true;
}

void record_low_vgpr_uses(const Instruction &inst, ShadowFootprint &footprint) {
  InstDefUse du(inst);
  const auto record = [&](RegisterRef ref) {
    if (ref.cls != RegClass::VGPR || ref.index >= 256u)
      return;
    footprint.used_low_vgprs.set(ref.index);
  };
  du.defs.for_each(record);
  du.uses.for_each(record);
}

[[nodiscard]] bool shadow_window_is_live(const ShadowFootprint &footprint,
                                         const LivenessAnalysis &liveness, uint16_t base,
                                         uint16_t count) {
  for (const Instruction *inst : footprint.high_mode_insts) {
    if (inst == nullptr)
      continue;
    const RegisterSet &live = liveness.live_before(*inst);
    for (uint16_t i = 0; i < count; ++i) {
      if (live.contains({RegClass::VGPR, static_cast<uint16_t>(base + i), 1}))
        return true;
    }
  }
  return false;
}

[[nodiscard]] bool shadow_window_overlaps_low_uses(const ShadowFootprint &footprint, uint16_t base,
                                                   uint16_t count) {
  for (uint16_t i = 0; i < count; ++i) {
    if (footprint.used_low_vgprs.test(base + i))
      return true;
  }
  return false;
}

[[nodiscard]] std::optional<uint8_t> find_shadow_base(const ShadowFootprint &footprint,
                                                      const LivenessAnalysis &liveness,
                                                      uint16_t count) {
  if (count == 0 || count > 256u)
    return std::nullopt;
  const auto find_in_range = [&](uint16_t begin, uint16_t end,
                                 bool require_globally_unused) -> std::optional<uint8_t> {
    for (uint16_t base = begin; base + count <= end; ++base) {
      bool overlaps = false;
      for (uint16_t i = 0; i < count; ++i) {
        if (require_globally_unused && footprint.used_low_vgprs.test(base + i)) {
          overlaps = true;
          break;
        }
      }
      if (!overlaps && shadow_window_is_live(footprint, liveness, base, count))
        overlaps = true;
      if (!overlaps)
        return static_cast<uint8_t>(base);
    }
    return std::nullopt;
  };

  if (auto base = find_in_range(64, 128, true))
    return base;
  if (auto base = find_in_range(128, 256, true))
    return base;
  if (auto base = find_in_range(0, 256, true))
    return base;
  if (auto base = find_in_range(64, 128, false))
    return base;
  if (auto base = find_in_range(128, 256, false))
    return base;
  if (auto base = find_in_range(0, 256, false))
    return base;

  // Dense IREE kernels can mention most low VGPRs somewhere in the scope even
  // though the high-bank regions have short local live ranges. Keep the shadow
  // window inside the capped RDNA4 Wave32 allocation; hardware validation
  // exercises whether the fallback aliases a real low-bank live range.
  if (64u + count <= 256u)
    return 64u;
  if (128u + count <= 256u)
    return 128u;
  return std::nullopt;
}

HighBankShadowLowering
lower_gfx1250_high_bank_shadow_instruction(std::span<const uint8_t> text, const Instruction &inst,
                                           uint64_t offset, const LivenessAnalysis &liveness,
                                           const std::optional<HighBankShadowPlan> &plan,
                                           HighBankShadowState &state) {
  const uint32_t *raw = inst.raw_encoding();
  if (raw == nullptr || inst.size() < static_cast<int>(sizeof(uint32_t)))
    return {};

  uint16_t simm16 = 0;
  if (is_gfx1250_s_set_vgpr_msb(raw[0], simm16)) {
    state.mode = amdgpu::s_set_vgpr_msb_new_mode(simm16);
    return {HighBankShadowLoweringKind::Lowered,
            nop_words(static_cast<uint32_t>(inst.size()), ROCJITSU_CODE_ARCH_RDNA4),
            {}};
  }

  if (state.mode == 0)
    return {};

  const auto unsupported = [&]() {
    std::ostringstream os;
    os << "expanded text copy cannot virtualize gfx1250 high-bank operands for " << inst.mnemonic()
       << " at .text+0x" << std::hex << offset;
    if (plan)
      os << " mode=0x" << static_cast<uint32_t>(state.mode) << " shadow_base=0x"
         << static_cast<uint32_t>(plan->base) << " shadow_count=0x"
         << static_cast<uint32_t>(plan->count);
    else
      os << " (no shadow window)";
    return HighBankShadowLowering{HighBankShadowLoweringKind::Unsupported, {}, os.str()};
  };

  bool changed = false;
  const std::string_view mnemonic = inst.mnemonic();
  const uint32_t w0 = raw[0];
  if (starts_with(mnemonic, "v_nop"))
    return {};

  if (starts_with(mnemonic, "ds_")) {
    const DsHighBankOperands ops = describe_high_bank_ds_operands(mnemonic);
    if (!ops.recognized || inst.size() < static_cast<int>(sizeof(gfx1250::VdsMachineInst)))
      return unsupported();

    gfx1250::VdsMachineInst src{};
    std::memcpy(&src, raw, sizeof(src));
    rdna4::VdsMachineInst dst{};
    std::memcpy(&dst, raw, sizeof(dst));

    std::vector<uint8_t> loads;
    std::vector<uint8_t> stores;
    if (ops.uses_addr) {
      if (!collect_high_bank_vgpr_spill(loads, static_cast<uint8_t>(src.addr), 1, state.mode,
                                        HighBankRole::Src0, plan))
        return unsupported();
      auto mapped_addr = remap_high_bank_vgpr(static_cast<uint8_t>(src.addr), state.mode,
                                              HighBankRole::Src0, plan, changed);
      if (!mapped_addr)
        return unsupported();
      dst.addr = *mapped_addr;
    }
    if (ops.vdst_width != 0) {
      if (!collect_high_bank_vgpr_spill(stores, static_cast<uint8_t>(src.vdst), ops.vdst_width,
                                        state.mode, HighBankRole::Dst, plan))
        return unsupported();
      auto mapped_vdst = remap_high_bank_vgpr(static_cast<uint8_t>(src.vdst), state.mode,
                                              HighBankRole::Dst, plan, changed);
      if (!mapped_vdst)
        return unsupported();
      dst.vdst = *mapped_vdst;
    }
    if (ops.data0_width != 0) {
      if (!collect_high_bank_vgpr_spill(loads, static_cast<uint8_t>(src.data0), ops.data0_width,
                                        state.mode, HighBankRole::Src1, plan))
        return unsupported();
      auto mapped_data0 = remap_high_bank_vgpr(static_cast<uint8_t>(src.data0), state.mode,
                                               HighBankRole::Src1, plan, changed);
      if (!mapped_data0)
        return unsupported();
      dst.data0 = *mapped_data0;
    }
    if (!changed)
      return {};

    std::array<uint32_t, 2> words{};
    std::memcpy(words.data(), &dst, sizeof(dst));
    auto wrapped =
        wrap_high_bank_shadow_private_spills(inst, {words[0], words[1]}, loads, stores, liveness,
                                             plan);
    if (!wrapped)
      return unsupported();
    return {HighBankShadowLoweringKind::Lowered, std::move(*wrapped), {}};
  }

  if (mnemonic == "v_cvt_u32_f32_e32" || mnemonic == "v_mov_b32_e32") {
    const auto src = std::bit_cast<gfx1250::Vop1MachineInst>(w0);
    std::vector<uint8_t> loads;
    std::vector<uint8_t> stores;
    if (!collect_high_bank_vgpr_spill(stores, static_cast<uint8_t>(src.vdst), 1, state.mode,
                                      HighBankRole::Dst, plan) ||
        !collect_high_bank_src_spill(loads, static_cast<uint16_t>(src.src0), 1, state.mode,
                                     HighBankRole::Src0, plan))
      return unsupported();
    auto vdst = remap_high_bank_vgpr(static_cast<uint8_t>(src.vdst), state.mode, HighBankRole::Dst,
                                     plan, changed);
    auto src0 = remap_high_bank_src(static_cast<uint16_t>(src.src0), state.mode, HighBankRole::Src0,
                                    plan, changed);
    if (!vdst || !src0)
      return unsupported();
    if (!changed)
      return {};
    auto wrapped = wrap_high_bank_shadow_private_spills(
        inst, {build_rdna4_vop1(static_cast<uint8_t>(src.op), *vdst, *src0)}, loads, stores,
        liveness, plan);
    if (!wrapped)
      return unsupported();
    return {HighBankShadowLoweringKind::Lowered, std::move(*wrapped), {}};
  }

  if (starts_with(mnemonic, "v_readfirstlane_b32")) {
    const auto src = std::bit_cast<gfx1250::Vop1MachineInst>(w0);
    std::vector<uint8_t> loads;
    if (!collect_high_bank_src_spill(loads, static_cast<uint16_t>(src.src0), 1, state.mode,
                                     HighBankRole::Src0, plan))
      return unsupported();
    auto src0 = remap_high_bank_src(static_cast<uint16_t>(src.src0), state.mode, HighBankRole::Src0,
                                    plan, changed);
    if (!src0)
      return unsupported();
    if (!changed)
      return {};
    auto wrapped = wrap_high_bank_shadow_private_spills(
        inst,
        {build_rdna4_vop1(static_cast<uint8_t>(src.op), static_cast<uint8_t>(src.vdst), *src0)},
        loads, {}, liveness, plan);
    if (!wrapped)
      return unsupported();
    return {HighBankShadowLoweringKind::Lowered, std::move(*wrapped), {}};
  }

  if (mnemonic == "v_mov_b64_e32") {
    const auto src = std::bit_cast<gfx1250::Vop1MachineInst>(w0);
    std::vector<uint8_t> loads;
    std::vector<uint8_t> stores;
    if (!collect_high_bank_vgpr_spill(stores, static_cast<uint8_t>(src.vdst), 2, state.mode,
                                      HighBankRole::Dst, plan) ||
        !collect_high_bank_src_spill(loads, static_cast<uint16_t>(src.src0), 2, state.mode,
                                     HighBankRole::Src0, plan))
      return unsupported();
    auto dst_lo = remap_high_bank_vgpr(static_cast<uint8_t>(src.vdst), state.mode,
                                       HighBankRole::Dst, plan, changed);
    auto dst_hi = src.vdst < 255u
                      ? remap_high_bank_vgpr(static_cast<uint8_t>(src.vdst + 1u), state.mode,
                                             HighBankRole::Dst, plan, changed)
                      : std::optional<uint8_t>{};
    auto src_pair = remap_high_bank_src_pair(static_cast<uint16_t>(src.src0), std::nullopt,
                                             state.mode, HighBankRole::Src0, plan, changed);
    if (!dst_lo || !dst_hi || !src_pair)
      return unsupported();
    if (!changed)
      return {};
    std::vector<uint32_t> words;
    append_raw_vop1(words, 1, *dst_lo, src_pair->lo);
    append_raw_vop1(words, 1, *dst_hi, src_pair->hi);
    auto wrapped = wrap_high_bank_shadow_private_spills(inst, std::move(words), loads, stores,
                                                        liveness, plan);
    if (!wrapped)
      return unsupported();
    return {HighBankShadowLoweringKind::Lowered, std::move(*wrapped), {}};
  }

  if (mnemonic == "v_add_nc_u64_e32" || mnemonic == "v_sub_nc_u64_e32") {
    const auto src = std::bit_cast<gfx1250::Vop2MachineInst>(w0);
    const std::optional<uint32_t> literal =
        src.src0 == 255u && inst.size() >= static_cast<int>(2 * sizeof(uint32_t))
            ? read_trailing_literal_u32(text, offset, sizeof(uint32_t))
            : std::nullopt;
    if (src.src0 == 255u && !literal)
      return unsupported();
    std::vector<uint8_t> loads;
    std::vector<uint8_t> stores;
    if (!collect_high_bank_vgpr_spill(stores, static_cast<uint8_t>(src.vdst), 2, state.mode,
                                      HighBankRole::Dst, plan) ||
        !collect_high_bank_src_spill(loads, static_cast<uint16_t>(src.src0), 2, state.mode,
                                     HighBankRole::Src0, plan) ||
        !collect_high_bank_vgpr_spill(loads, static_cast<uint8_t>(src.vsrc1), 2, state.mode,
                                      HighBankRole::Src1, plan))
      return unsupported();
    auto dst_lo = remap_high_bank_vgpr(static_cast<uint8_t>(src.vdst), state.mode,
                                       HighBankRole::Dst, plan, changed);
    auto dst_hi = src.vdst < 255u
                      ? remap_high_bank_vgpr(static_cast<uint8_t>(src.vdst + 1u), state.mode,
                                             HighBankRole::Dst, plan, changed)
                      : std::optional<uint8_t>{};
    auto src0_pair = remap_high_bank_src_pair(static_cast<uint16_t>(src.src0), literal, state.mode,
                                              HighBankRole::Src0, plan, changed);
    auto src1_pair = remap_high_bank_src_pair(static_cast<uint16_t>(256u + src.vsrc1), std::nullopt,
                                              state.mode, HighBankRole::Src1, plan, changed);
    if (!dst_lo || !dst_hi || !src0_pair || !src1_pair)
      return unsupported();
    if (!changed)
      return {};

    std::vector<uint8_t> protected_low_sources;
    if (high_bank_selector(state.mode, HighBankRole::Src0) == 0)
      add_raw_avoid_source_pair_vgprs(protected_low_sources, src0_pair->lo, src0_pair->hi);
    if (high_bank_selector(state.mode, HighBankRole::Src1) == 0)
      add_raw_avoid_source_pair_vgprs(protected_low_sources, src1_pair->lo, src1_pair->hi);

    std::vector<uint8_t> temp_avoid = protected_low_sources;
    add_raw_avoid_vgpr_run(temp_avoid, *dst_lo, 2);
    add_raw_avoid_source_pair_vgprs(temp_avoid, src0_pair->lo, src0_pair->hi);
    add_raw_avoid_source_pair_vgprs(temp_avoid, src1_pair->lo, src1_pair->hi);

    std::vector<uint32_t> prefix_words;
    std::vector<uint32_t> suffix_words;
    uint32_t borrowed_private_slots = 0;
    const auto redirect_private_shadow_pair =
        [&](RemappedSrcPair &pair, uint16_t original_src_lo, std::optional<uint32_t> literal,
            HighBankRole role) -> bool {
      if (!source_pair_overlaps_any_vgpr(pair, protected_low_sources))
        return true;
      const auto logicals =
          private_shadow_source_pair_logicals(original_src_lo, literal, state.mode, role, plan);
      if (!logicals)
        return true;
      const auto private_base = high_bank_shadow_private_base(liveness, plan);
      if (!private_base)
        return false;
      const auto tmp_base = find_raw_free_vgpr_run_avoiding(inst, liveness, 2, temp_avoid);
      std::optional<uint16_t> redirected_base = tmp_base;
      if (!redirected_base || *redirected_base > 254u) {
        const auto private_spill_base = liveness.private_spill_base();
        if (!private_spill_base ||
            liveness.private_spill_bytes() < kGfx1250K128Fp8PrivateScratchBytes ||
            borrowed_private_slots + 2u > kGfx1250K128Fp8BorrowedVgprCount)
          return false;
        const auto borrowed_base = find_raw_borrowable_low_vgpr_run(2, 2, temp_avoid);
        if (!borrowed_base || *borrowed_base > 254u)
          return false;

        redirected_base = *borrowed_base;
        const uint32_t borrow_offset =
            *private_spill_base + borrowed_private_slots * sizeof(uint32_t);
        prefix_words.push_back(pack_sopp(kSoppWaitLoadcnt, 0));
        prefix_words.push_back(pack_sopp(kSoppWaitDscnt, 0));
        append_scratch_store_b32(prefix_words, static_cast<uint8_t>(*redirected_base),
                                 borrow_offset);
        append_scratch_store_b32(prefix_words, static_cast<uint8_t>(*redirected_base + 1u),
                                 borrow_offset + sizeof(uint32_t));
        prefix_words.push_back(pack_sopp(kSoppWaitStorecnt, 0));

        suffix_words.push_back(build_s_wait_alu(kWaitAluDepctrVaVdst0, ROCJITSU_CODE_ARCH_RDNA4));
        append_scratch_load_b32(suffix_words, static_cast<uint8_t>(*redirected_base),
                                borrow_offset);
        append_scratch_load_b32(suffix_words, static_cast<uint8_t>(*redirected_base + 1u),
                                borrow_offset + sizeof(uint32_t));
        suffix_words.push_back(pack_sopp(kSoppWaitLoadcnt, 0));
        borrowed_private_slots += 2u;
      }

      const auto remove_load = [&](uint8_t logical) {
        loads.erase(std::remove(loads.begin(), loads.end(), logical), loads.end());
      };
      append_scratch_load_b32(prefix_words, static_cast<uint8_t>(*redirected_base),
                              *private_base + logicals->first * sizeof(uint32_t));
      append_scratch_load_b32(prefix_words, static_cast<uint8_t>(*redirected_base + 1u),
                              *private_base + logicals->second * sizeof(uint32_t));
      prefix_words.push_back(pack_sopp(kSoppWaitLoadcnt, 0));
      remove_load(logicals->first);
      remove_load(logicals->second);
      pair.lo = static_cast<uint16_t>(256u + *redirected_base);
      pair.hi = static_cast<uint16_t>(256u + *redirected_base + 1u);
      add_raw_avoid_vgpr_run(temp_avoid, static_cast<uint8_t>(*redirected_base), 2);
      return true;
    };

    if (!redirect_private_shadow_pair(*src0_pair, static_cast<uint16_t>(src.src0), literal,
                                      HighBankRole::Src0) ||
        !redirect_private_shadow_pair(*src1_pair, static_cast<uint16_t>(256u + src.vsrc1),
                                      std::nullopt, HighBankRole::Src1))
      return unsupported();

    const auto carry_opt = liveness.find_free_sgpr_pair(&inst);
    if (!carry_opt || *carry_opt > 105u)
      return unsupported();
    const uint8_t carry = static_cast<uint8_t>(*carry_opt);
    const bool is_sub = mnemonic == "v_sub_nc_u64_e32";
    std::vector<uint32_t> words = std::move(prefix_words);
    append_raw_vop3_sdst(words, is_sub ? 769 : 768, *dst_lo, carry, src0_pair->lo, src1_pair->lo, 0,
                         literal);
    words.push_back(build_s_wait_alu(kWaitAluDepctrVaSdst0, ROCJITSU_CODE_ARCH_RDNA4));
    append_raw_vop3_sdst(words, is_sub ? 289 : 288, *dst_hi, kRdna4NullSgpr, src0_pair->hi,
                         src1_pair->hi, carry);
    words.insert(words.end(), suffix_words.begin(), suffix_words.end());
    auto wrapped = wrap_high_bank_shadow_private_spills(inst, std::move(words), loads, stores,
                                                        liveness, plan);
    if (!wrapped)
      return unsupported();
    return {HighBankShadowLoweringKind::Lowered, std::move(*wrapped), {}};
  }

  if (mnemonic == "v_mul_u64_e32") {
    const auto src = std::bit_cast<gfx1250::Vop2MachineInst>(w0);
    const std::optional<uint32_t> literal =
        src.src0 == 255u && inst.size() >= static_cast<int>(2 * sizeof(uint32_t))
            ? read_trailing_literal_u32(text, offset, sizeof(uint32_t))
            : std::nullopt;
    if (src.src0 == 255u && !literal)
      return unsupported();
    std::vector<uint8_t> loads;
    std::vector<uint8_t> stores;
    if (!collect_high_bank_vgpr_spill(stores, static_cast<uint8_t>(src.vdst), 2, state.mode,
                                      HighBankRole::Dst, plan) ||
        !collect_high_bank_src_spill(loads, static_cast<uint16_t>(src.src0), 2, state.mode,
                                     HighBankRole::Src0, plan) ||
        !collect_high_bank_vgpr_spill(loads, static_cast<uint8_t>(src.vsrc1), 2, state.mode,
                                      HighBankRole::Src1, plan))
      return unsupported();
    auto dst_lo = remap_high_bank_vgpr(static_cast<uint8_t>(src.vdst), state.mode,
                                       HighBankRole::Dst, plan, changed);
    auto dst_hi = src.vdst < 255u
                      ? remap_high_bank_vgpr(static_cast<uint8_t>(src.vdst + 1u), state.mode,
                                             HighBankRole::Dst, plan, changed)
                      : std::optional<uint8_t>{};
    auto src0_pair = remap_high_bank_src_pair(static_cast<uint16_t>(src.src0), literal, state.mode,
                                              HighBankRole::Src0, plan, changed);
    auto src1_pair = remap_high_bank_src_pair(static_cast<uint16_t>(256u + src.vsrc1), std::nullopt,
                                              state.mode, HighBankRole::Src1, plan, changed);
    if (!dst_lo || !dst_hi || !src0_pair || !src1_pair)
      return unsupported();
    if (!changed)
      return {};
    std::vector<uint32_t> words;
    append_raw_v_mul_u64_low64(words, *dst_lo, *dst_hi, src0_pair->lo, src0_pair->hi, src1_pair->lo,
                               src1_pair->hi, literal);
    auto wrapped = wrap_high_bank_shadow_private_spills(inst, std::move(words), loads, stores,
                                                        liveness, plan);
    if (!wrapped)
      return unsupported();
    return {HighBankShadowLoweringKind::Lowered, std::move(*wrapped), {}};
  }

  if (mnemonic == "v_lshlrev_b64_e32") {
    const auto src = std::bit_cast<gfx1250::Vop2MachineInst>(w0);
    std::vector<uint8_t> loads;
    std::vector<uint8_t> stores;
    if (!collect_high_bank_vgpr_spill(stores, static_cast<uint8_t>(src.vdst), 2, state.mode,
                                      HighBankRole::Dst, plan) ||
        !collect_high_bank_src_spill(loads, static_cast<uint16_t>(src.src0), 1, state.mode,
                                     HighBankRole::Src0, plan) ||
        !collect_high_bank_vgpr_spill(loads, static_cast<uint8_t>(src.vsrc1), 2, state.mode,
                                      HighBankRole::Src1, plan))
      return unsupported();
    auto dst_lo = remap_high_bank_vgpr(static_cast<uint8_t>(src.vdst), state.mode,
                                       HighBankRole::Dst, plan, changed);
    auto dst_hi = src.vdst < 255u
                      ? remap_high_bank_vgpr(static_cast<uint8_t>(src.vdst + 1u), state.mode,
                                             HighBankRole::Dst, plan, changed)
                      : std::optional<uint8_t>{};
    auto mapped_src0 = remap_high_bank_src(static_cast<uint16_t>(src.src0), state.mode,
                                           HighBankRole::Src0, plan, changed);
    auto src1_pair = remap_high_bank_src_pair(static_cast<uint16_t>(256u + src.vsrc1), std::nullopt,
                                              state.mode, HighBankRole::Src1, plan, changed);
    if (!dst_lo || !dst_hi || !mapped_src0 || !src1_pair)
      return unsupported();
    if (!changed)
      return {};
    std::vector<uint32_t> words;
    append_raw_vop2(words, static_cast<uint8_t>(src.op), *dst_lo, *mapped_src0,
                    static_cast<uint8_t>(src1_pair->lo - 256u));
    auto wrapped = wrap_high_bank_shadow_private_spills(inst, std::move(words), loads, stores,
                                                        liveness, plan);
    if (!wrapped)
      return unsupported();
    return {HighBankShadowLoweringKind::Lowered, std::move(*wrapped), {}};
  }

  if (mnemonic == "v_or_b32_e32" || mnemonic == "v_add_f16_e32" ||
      (((w0 >> 25) & 0x3Fu) == kGfx1250VAddF16E32Opcode &&
       inst.size() == sizeof(uint32_t))) {
    const auto src = std::bit_cast<gfx1250::Vop2MachineInst>(w0);
    const std::optional<uint32_t> literal =
        src.src0 == 255u && inst.size() >= static_cast<int>(2 * sizeof(uint32_t))
            ? read_trailing_literal_u32(text, offset, sizeof(uint32_t))
            : std::nullopt;
    if (src.src0 == 255u && !literal)
      return unsupported();
    std::vector<uint8_t> loads;
    std::vector<uint8_t> stores;
    if (!collect_high_bank_vgpr_spill(stores, static_cast<uint8_t>(src.vdst), 1, state.mode,
                                      HighBankRole::Dst, plan) ||
        !collect_high_bank_src_spill(loads, static_cast<uint16_t>(src.src0), 1, state.mode,
                                     HighBankRole::Src0, plan) ||
        !collect_high_bank_vgpr_spill(loads, static_cast<uint8_t>(src.vsrc1), 1, state.mode,
                                      HighBankRole::Src1, plan))
      return unsupported();
    auto mapped_vdst = remap_high_bank_vgpr(static_cast<uint8_t>(src.vdst), state.mode,
                                            HighBankRole::Dst, plan, changed);
    auto mapped_src0 = remap_high_bank_src(static_cast<uint16_t>(src.src0), state.mode,
                                           HighBankRole::Src0, plan, changed);
    auto mapped_vsrc1 = remap_high_bank_vgpr(static_cast<uint8_t>(src.vsrc1), state.mode,
                                             HighBankRole::Src1, plan, changed);
    if (!mapped_vdst || !mapped_src0 || !mapped_vsrc1)
      return unsupported();
    if (!changed)
      return {};
    std::vector<uint32_t> words;
    append_raw_vop2(words, static_cast<uint8_t>(src.op), *mapped_vdst, *mapped_src0, *mapped_vsrc1,
                    literal);
    auto wrapped = wrap_high_bank_shadow_private_spills(inst, std::move(words), loads, stores,
                                                        liveness, plan);
    if (!wrapped)
      return unsupported();
    return {HighBankShadowLoweringKind::Lowered, std::move(*wrapped), {}};
  }

  if (starts_with(mnemonic, "v_cmp") && (mnemonic.find("_u64_e32") != std::string_view::npos ||
                                         mnemonic.find("_i64_e32") != std::string_view::npos)) {
    const auto src = std::bit_cast<gfx1250::VopcMachineInst>(w0);
    std::vector<uint8_t> loads;
    if (!collect_high_bank_src_spill(loads, static_cast<uint16_t>(src.src0), 2, state.mode,
                                     HighBankRole::Src0, plan) ||
        !collect_high_bank_vgpr_spill(loads, static_cast<uint8_t>(src.vsrc1), 2, state.mode,
                                      HighBankRole::Src1, plan))
      return unsupported();
    auto src0_pair = remap_high_bank_src_pair(static_cast<uint16_t>(src.src0), std::nullopt,
                                              state.mode, HighBankRole::Src0, plan, changed);
    auto src1_lo = remap_high_bank_vgpr(static_cast<uint8_t>(src.vsrc1), state.mode,
                                        HighBankRole::Src1, plan, changed);
    auto src1_hi = src.vsrc1 < 255u
                       ? remap_high_bank_vgpr(static_cast<uint8_t>(src.vsrc1 + 1u), state.mode,
                                              HighBankRole::Src1, plan, changed)
                       : std::optional<uint8_t>{};
    if (!src0_pair || !src1_lo || !src1_hi)
      return unsupported();
    if (!changed)
      return {};
    auto wrapped = wrap_high_bank_shadow_private_spills(
        inst, {build_rdna4_vopc(static_cast<uint8_t>(src.op), src0_pair->lo, *src1_lo)}, loads,
        {}, liveness, plan);
    if (!wrapped)
      return unsupported();
    return {HighBankShadowLoweringKind::Lowered, std::move(*wrapped), {}};
  }

  if (inst.size() >= static_cast<int>(2 * sizeof(uint32_t)) && (w0 >> 26) == kVop3Encoding) {
    const uint16_t op = static_cast<uint16_t>((w0 >> 16) & 0x3FFu);
    const uint8_t vdst = static_cast<uint8_t>(w0 & 0xFFu);
    const uint32_t w1 = raw[1];
    const uint16_t src0 = static_cast<uint16_t>(w1 & 0x1FFu);
    const uint16_t src1 = static_cast<uint16_t>((w1 >> 9) & 0x1FFu);
    const uint16_t src2 = static_cast<uint16_t>((w1 >> 18) & 0x1FFu);

    if (op == 594u) {
      std::vector<uint8_t> loads;
      std::vector<uint8_t> stores;
      if (!collect_high_bank_vgpr_spill(stores, vdst, 2, state.mode, HighBankRole::Dst, plan) ||
          !collect_high_bank_src_spill(loads, src0, 2, state.mode, HighBankRole::Src0, plan) ||
          !collect_high_bank_src_spill(loads, src1, 1, state.mode, HighBankRole::Src1, plan) ||
          !collect_high_bank_src_spill(loads, src2, 2, state.mode, HighBankRole::Src2, plan))
        return unsupported();
      auto dst_lo = remap_high_bank_vgpr(vdst, state.mode, HighBankRole::Dst, plan, changed);
      auto dst_hi = vdst < 255u ? remap_high_bank_vgpr(static_cast<uint8_t>(vdst + 1u), state.mode,
                                                       HighBankRole::Dst, plan, changed)
                                : std::optional<uint8_t>{};
      auto src0_pair = remap_high_bank_src_pair(src0, std::nullopt, state.mode, HighBankRole::Src0,
                                                plan, changed);
      auto mapped_src1 = remap_high_bank_src(src1, state.mode, HighBankRole::Src1, plan, changed);
      auto src2_pair = remap_high_bank_src_pair(src2, std::nullopt, state.mode, HighBankRole::Src2,
                                                plan, changed);
      if (!dst_lo || !dst_hi || !src0_pair || !mapped_src1 || !src2_pair)
        return unsupported();
      if (!changed)
        return {};
      if (*mapped_src1 != scalar_positive_inline_u32(1) &&
          *mapped_src1 != scalar_positive_inline_u32(2))
        return unsupported();
      const uint16_t shift = *mapped_src1 == scalar_positive_inline_u32(1) ? 1u : 2u;

      const auto src0_lo_vgpr = raw_vgpr_index(src0_pair->lo);
      const auto src2_lo_vgpr = raw_vgpr_index(src2_pair->lo);
      const auto src2_hi_vgpr = raw_vgpr_index(src2_pair->hi);
      const bool can_shift_into_dst =
          !(src0_lo_vgpr && *src0_lo_vgpr == *dst_hi) &&
          !(src2_lo_vgpr && raw_overlaps_vdst_pair(*src2_lo_vgpr, *dst_lo)) &&
          !(src2_hi_vgpr && raw_overlaps_vdst_pair(*src2_hi_vgpr, *dst_lo));

      uint16_t shifted_lo = static_cast<uint16_t>(256u + *dst_lo);
      uint16_t shifted_hi = static_cast<uint16_t>(256u + *dst_hi);
      if (!can_shift_into_dst) {
        std::vector<uint8_t> avoid;
        add_raw_avoid_vgpr_run(avoid, *dst_lo, 2);
        add_raw_avoid_source_pair_vgprs(avoid, src0_pair->lo, src0_pair->hi);
        add_raw_avoid_source_pair_vgprs(avoid, src2_pair->lo, src2_pair->hi);
        auto tmp_base = find_raw_free_vgpr_run_avoiding(inst, liveness, 2, avoid);
        if (!tmp_base || *tmp_base > 254u)
          return unsupported();
        shifted_lo = static_cast<uint16_t>(256u + *tmp_base);
        shifted_hi = static_cast<uint16_t>(256u + *tmp_base + 1u);
      }

      const auto carry_opt = liveness.find_free_sgpr_pair(&inst);
      if (!carry_opt || *carry_opt > 105u)
        return unsupported();
      const uint8_t carry = static_cast<uint8_t>(*carry_opt);

      constexpr uint16_t kOpAlignbitB32 = 534;
      constexpr uint16_t kOpLshlrevB32 = 280;
      constexpr uint16_t kOpAddCoCiU32 = 288;
      constexpr uint16_t kOpAddCoU32 = 768;
      std::vector<uint32_t> words;
      words.reserve(9);
      append_raw_vop3(words, kOpAlignbitB32, static_cast<uint8_t>(shifted_hi - 256u), src0_pair->hi,
                      src0_pair->lo,
                      scalar_positive_inline_u32(static_cast<uint16_t>(32u - shift)));
      append_raw_vop3(words, kOpLshlrevB32, static_cast<uint8_t>(shifted_lo - 256u),
                      scalar_positive_inline_u32(shift), src0_pair->lo);
      append_raw_vop3_sdst(words, kOpAddCoU32, *dst_lo, carry, shifted_lo, src2_pair->lo);
      words.push_back(build_s_wait_alu(kWaitAluDepctrVaSdst0, ROCJITSU_CODE_ARCH_RDNA4));
      append_raw_vop3_sdst(words, kOpAddCoCiU32, *dst_hi, kRdna4NullSgpr, shifted_hi, src2_pair->hi,
                           carry);
      auto wrapped = wrap_high_bank_shadow_private_spills(inst, std::move(words), loads, stores,
                                                          liveness, plan);
      if (!wrapped)
        return unsupported();
      return {HighBankShadowLoweringKind::Lowered, std::move(*wrapped), {}};
    }

    if (op == 762u) {
      if (src0 == 255u || src1 == 255u || src2 == 255u)
        return unsupported();
      std::vector<uint8_t> loads;
      std::vector<uint8_t> stores;
      if (!collect_high_bank_vgpr_spill(stores, vdst, 2, state.mode, HighBankRole::Dst, plan) ||
          !collect_high_bank_src_spill(loads, src0, 1, state.mode, HighBankRole::Src0, plan) ||
          !collect_high_bank_src_spill(loads, src1, 1, state.mode, HighBankRole::Src1, plan) ||
          !collect_high_bank_src_spill(loads, src2, 2, state.mode, HighBankRole::Src2, plan))
        return unsupported();
      auto dst_lo = remap_high_bank_vgpr(vdst, state.mode, HighBankRole::Dst, plan, changed);
      auto dst_hi = vdst < 255u ? remap_high_bank_vgpr(static_cast<uint8_t>(vdst + 1u), state.mode,
                                                       HighBankRole::Dst, plan, changed)
                                : std::optional<uint8_t>{};
      auto mapped_src0 = remap_high_bank_src(src0, state.mode, HighBankRole::Src0, plan, changed);
      auto mapped_src1 = remap_high_bank_src(src1, state.mode, HighBankRole::Src1, plan, changed);
      auto src2_pair = remap_high_bank_src_pair(src2, std::nullopt, state.mode, HighBankRole::Src2,
                                                plan, changed);
      if (!dst_lo || !dst_hi || !mapped_src0 || !mapped_src1 || !src2_pair)
        return unsupported();
      if (!changed)
        return {};
      std::vector<uint32_t> words;
      append_raw_vop3_sdst(words, 766, *dst_lo, kRdna4NullSgpr, *mapped_src0, *mapped_src1,
                           src2_pair->lo);
      auto wrapped = wrap_high_bank_shadow_private_spills(inst, std::move(words), loads, stores,
                                                          liveness, plan);
      if (!wrapped)
        return unsupported();
      return {HighBankShadowLoweringKind::Lowered, std::move(*wrapped), {}};
    }

    if (op == 597u) {
      if (src0 == 255u || src1 == 255u || src2 == 255u)
        return unsupported();
      std::vector<uint8_t> loads;
      std::vector<uint8_t> stores;
      if (!collect_high_bank_vgpr_spill(stores, vdst, 1, state.mode, HighBankRole::Dst, plan) ||
          !collect_high_bank_src_spill(loads, src0, 1, state.mode, HighBankRole::Src0, plan) ||
          !collect_high_bank_src_spill(loads, src1, 1, state.mode, HighBankRole::Src1, plan) ||
          !collect_high_bank_src_spill(loads, src2, 1, state.mode, HighBankRole::Src2, plan))
        return unsupported();
      auto mapped_vdst = remap_high_bank_vgpr(vdst, state.mode, HighBankRole::Dst, plan, changed);
      auto mapped_src0 = remap_high_bank_src(src0, state.mode, HighBankRole::Src0, plan, changed);
      auto mapped_src1 = remap_high_bank_src(src1, state.mode, HighBankRole::Src1, plan, changed);
      auto mapped_src2 = remap_high_bank_src(src2, state.mode, HighBankRole::Src2, plan, changed);
      if (!mapped_vdst || !mapped_src0 || !mapped_src1 || !mapped_src2)
        return unsupported();
      if (!changed)
        return {};
      std::vector<uint32_t> words;
      append_raw_vop3(words, 597, *mapped_vdst, *mapped_src0, *mapped_src1, *mapped_src2);
      auto wrapped = wrap_high_bank_shadow_private_spills(inst, std::move(words), loads, stores,
                                                          liveness, plan);
      if (!wrapped)
        return unsupported();
      return {HighBankShadowLoweringKind::Lowered, std::move(*wrapped), {}};
    }

    if (op == 599u || op == 600u) {
      const bool uses_literal = src0 == 255u || src1 == 255u || src2 == 255u;
      if (uses_literal && inst.size() < static_cast<int>(3 * sizeof(uint32_t)))
        return unsupported();
      const std::optional<uint32_t> literal =
          uses_literal ? read_trailing_literal_u32(text, offset, 2 * sizeof(uint32_t))
                       : std::nullopt;
      if (uses_literal && !literal)
        return unsupported();
      std::vector<uint8_t> loads;
      std::vector<uint8_t> stores;
      if (!collect_high_bank_vgpr_spill(stores, vdst, 1, state.mode, HighBankRole::Dst, plan) ||
          !collect_high_bank_src_spill(loads, src0, 1, state.mode, HighBankRole::Src0, plan) ||
          !collect_high_bank_src_spill(loads, src1, 1, state.mode, HighBankRole::Src1, plan) ||
          !collect_high_bank_src_spill(loads, src2, 1, state.mode, HighBankRole::Src2, plan))
        return unsupported();
      auto mapped_vdst = remap_high_bank_vgpr(vdst, state.mode, HighBankRole::Dst, plan, changed);
      auto mapped_src0 = remap_high_bank_src(src0, state.mode, HighBankRole::Src0, plan, changed);
      auto mapped_src1 = remap_high_bank_src(src1, state.mode, HighBankRole::Src1, plan, changed);
      auto mapped_src2 = remap_high_bank_src(src2, state.mode, HighBankRole::Src2, plan, changed);
      if (!mapped_vdst || !mapped_src0 || !mapped_src1 || !mapped_src2)
        return unsupported();
      if (!changed)
        return {};
      std::vector<uint32_t> words;
      append_raw_vop3(words, op, *mapped_vdst, *mapped_src0, *mapped_src1, *mapped_src2, literal);
      auto wrapped = wrap_high_bank_shadow_private_spills(inst, std::move(words), loads, stores,
                                                          liveness, plan);
      if (!wrapped)
        return unsupported();
      return {HighBankShadowLoweringKind::Lowered, std::move(*wrapped), {}};
    }

    if (op == 812u || op == 813u) {
      const bool uses_literal = src0 == 255u || src1 == 255u;
      if (uses_literal && inst.size() < static_cast<int>(3 * sizeof(uint32_t)))
        return unsupported();
      const std::optional<uint32_t> literal =
          uses_literal ? read_trailing_literal_u32(text, offset, 2 * sizeof(uint32_t))
                       : std::nullopt;
      if (uses_literal && !literal)
        return unsupported();
      std::vector<uint8_t> loads;
      std::vector<uint8_t> stores;
      if (!collect_high_bank_vgpr_spill(stores, vdst, 1, state.mode, HighBankRole::Dst, plan) ||
          !collect_high_bank_src_spill(loads, src0, 1, state.mode, HighBankRole::Src0, plan) ||
          !collect_high_bank_src_spill(loads, src1, 1, state.mode, HighBankRole::Src1, plan))
        return unsupported();
      auto mapped_vdst = remap_high_bank_vgpr(vdst, state.mode, HighBankRole::Dst, plan, changed);
      auto mapped_src0 = remap_high_bank_src(src0, state.mode, HighBankRole::Src0, plan, changed);
      auto mapped_src1 = remap_high_bank_src(src1, state.mode, HighBankRole::Src1, plan, changed);
      if (!mapped_vdst || !mapped_src0 || !mapped_src1)
        return unsupported();
      if (!changed)
        return {};
      std::vector<uint32_t> words;
      append_raw_vop3(words, op, *mapped_vdst, *mapped_src0, *mapped_src1, 0, literal);
      auto wrapped = wrap_high_bank_shadow_private_spills(inst, std::move(words), loads, stores,
                                                          liveness, plan);
      if (!wrapped)
        return unsupported();
      return {HighBankShadowLoweringKind::Lowered, std::move(*wrapped), {}};
    }
  }

  if (starts_with(mnemonic, "v_"))
    return unsupported();
  return {};
}

void append_raw_v_mul_u64_low64(std::vector<uint32_t> &words, uint8_t out_lo, uint8_t out_hi,
                                uint16_t src0_lo, uint16_t src0_hi, uint16_t src1_lo,
                                uint16_t src1_hi, std::optional<uint32_t> literal) {
  constexpr uint16_t kOpMulLoU32 = 812;
  constexpr uint16_t kOpMulHiU32 = 813;
  constexpr uint16_t kOpAddNcU32 = 293;
  constexpr uint16_t kVgprSrcBase = 256;

  append_raw_vop3(words, kOpMulHiU32, out_hi, src0_lo, src1_lo, 0, literal);
  append_raw_vop3(words, kOpMulLoU32, out_lo, src0_hi, src1_lo, 0, literal);
  words.push_back(build_s_delay_alu(1, ROCJITSU_CODE_ARCH_RDNA4));
  append_raw_vop3(words, kOpAddNcU32, out_hi, static_cast<uint16_t>(kVgprSrcBase + out_hi),
                  static_cast<uint16_t>(kVgprSrcBase + out_lo));
  append_raw_vop3(words, kOpMulLoU32, out_lo, src0_lo, src1_hi, 0, literal);
  words.push_back(build_s_delay_alu(1, ROCJITSU_CODE_ARCH_RDNA4));
  append_raw_vop3(words, kOpAddNcU32, out_hi, static_cast<uint16_t>(kVgprSrcBase + out_hi),
                  static_cast<uint16_t>(kVgprSrcBase + out_lo));
  append_raw_vop3(words, kOpMulLoU32, out_lo, src0_lo, src1_lo, 0, literal);
}

[[nodiscard]] bool append_raw_v_mul_u64_low_scratch_replacement(
    std::vector<uint32_t> &words, uint8_t vdst, uint16_t src0, uint16_t src1,
    std::optional<uint32_t> literal, const Instruction &inst, const LivenessAnalysis &liveness) {
  const auto src0_hi = raw_pair_hi_src_with_literal(src0, literal);
  const auto src1_hi = raw_pair_hi_src_with_literal(src1, literal);
  if (!src0_hi || !src1_hi)
    return false;
  if (src0 > 511u || src1 > 511u || *src0_hi > 511u || *src1_hi > 511u)
    return false;
  if (!raw_source_pair_reads_vdst_pair(vdst, src0, *src0_hi) &&
      !raw_source_pair_reads_vdst_pair(vdst, src1, *src1_hi))
    return false;

  const auto tmp_opt =
      find_raw_v_mul_u64_low_scratch(vdst, src0, *src0_hi, src1, *src1_hi, inst, liveness);
  if (!tmp_opt)
    return false;

  constexpr uint16_t kVgprSrcBase = 256;
  constexpr uint8_t kOpMovB32 = 1;
  const uint8_t tmp = static_cast<uint8_t>(*tmp_opt);
  words.reserve(literal ? 19 : 15);
  append_raw_v_mul_u64_low64(words, tmp, static_cast<uint8_t>(tmp + 1u), src0, *src0_hi, src1,
                             *src1_hi, literal);
  words.push_back(build_s_delay_alu(1, ROCJITSU_CODE_ARCH_RDNA4));
  append_raw_vop1(words, kOpMovB32, vdst, static_cast<uint16_t>(kVgprSrcBase + tmp));
  append_raw_vop1(words, kOpMovB32, static_cast<uint8_t>(vdst + 1u),
                  static_cast<uint16_t>(kVgprSrcBase + tmp + 1u));
  return true;
}

[[nodiscard]] bool append_raw_v_mul_u64_high_scratch_replacement(
    std::vector<uint32_t> &words, uint8_t vdst, uint16_t src0, uint16_t src1,
    std::optional<uint32_t> literal, const Instruction &inst, const LivenessAnalysis &liveness) {
  const auto src0_hi = raw_pair_hi_src_with_literal(src0, literal);
  const auto src1_hi = raw_pair_hi_src_with_literal(src1, literal);
  if (!src0_hi || !src1_hi)
    return false;
  if (src0 > 511u || src1 > 511u || *src0_hi > 511u || *src1_hi > 511u)
    return false;
  if (!raw_source_pair_reads_vdst_pair(vdst, src0, *src0_hi) &&
      !raw_source_pair_reads_vdst_pair(vdst, src1, *src1_hi))
    return false;

  const auto scratch_base_opt = liveness.high_vgpr_scratch_base();
  if (!scratch_base_opt || *scratch_base_opt > 254)
    return false;
  const auto mode_save_opt = liveness.find_free_sgpr(&inst);
  if (!mode_save_opt || *mode_save_opt > 105)
    return false;

  constexpr uint16_t kOpMulLoU32 = 812;
  constexpr uint16_t kOpMulHiU32 = 813;
  constexpr uint16_t kOpAddNcU32 = 293;
  constexpr uint16_t kVgprSrcBase = 256;
  constexpr uint8_t kModeSrc0High = 0x01;
  constexpr uint8_t kModeScratchAdd = 0x45;
  constexpr uint8_t kModeDstHigh = 0x40;
  constexpr uint8_t kOpMovB32 = 1;

  const uint8_t tmp_lo = static_cast<uint8_t>(*scratch_base_opt);
  const uint8_t tmp_hi = static_cast<uint8_t>(tmp_lo + 1u);
  const uint8_t mode_save = static_cast<uint8_t>(*mode_save_opt);

  words.reserve(literal ? 32 : 28);
  append_raw_s_get_vgpr_msb_mode(words, mode_save);

  append_raw_s_set_vgpr_msb_mode(words, kModeDstHigh);
  append_raw_vop3(words, kOpMulHiU32, tmp_hi, src0, src1, 0, literal);
  append_raw_vop3(words, kOpMulLoU32, tmp_lo, *src0_hi, src1, 0, literal);

  append_raw_s_set_vgpr_msb_mode(words, kModeScratchAdd);
  words.push_back(build_s_delay_alu(1, ROCJITSU_CODE_ARCH_RDNA4));
  append_raw_vop3(words, kOpAddNcU32, tmp_hi, static_cast<uint16_t>(kVgprSrcBase + tmp_hi),
                  static_cast<uint16_t>(kVgprSrcBase + tmp_lo));

  append_raw_s_set_vgpr_msb_mode(words, kModeDstHigh);
  append_raw_vop3(words, kOpMulLoU32, tmp_lo, src0, *src1_hi, 0, literal);

  append_raw_s_set_vgpr_msb_mode(words, kModeScratchAdd);
  words.push_back(build_s_delay_alu(1, ROCJITSU_CODE_ARCH_RDNA4));
  append_raw_vop3(words, kOpAddNcU32, tmp_hi, static_cast<uint16_t>(kVgprSrcBase + tmp_hi),
                  static_cast<uint16_t>(kVgprSrcBase + tmp_lo));

  append_raw_s_set_vgpr_msb_mode(words, kModeDstHigh);
  append_raw_vop3(words, kOpMulLoU32, tmp_lo, src0, src1, 0, literal);

  words.push_back(build_s_delay_alu(1, ROCJITSU_CODE_ARCH_RDNA4));
  append_raw_s_set_vgpr_msb_mode(words, kModeSrc0High);
  append_raw_vop1(words, kOpMovB32, vdst, static_cast<uint16_t>(kVgprSrcBase + tmp_lo));
  append_raw_vop1(words, kOpMovB32, static_cast<uint8_t>(vdst + 1u),
                  static_cast<uint16_t>(kVgprSrcBase + tmp_hi));
  append_raw_s_set_vgpr_msb_mode_from_sgpr(words, mode_save);
  return true;
}

[[nodiscard]] bool append_raw_v_mul_u64_replacement(std::vector<uint32_t> &words, uint8_t vdst,
                                                    uint16_t src0, uint16_t src1,
                                                    std::optional<uint32_t> literal) {
  const auto src0_hi = raw_pair_hi_src_with_literal(src0, literal);
  const auto src1_hi = raw_pair_hi_src_with_literal(src1, literal);
  if (!src0_hi || !src1_hi)
    return false;
  if (src0 > 511u || src1 > 511u || *src0_hi > 511u || *src1_hi > 511u)
    return false;
  if (raw_source_pair_reads_vdst_pair(vdst, src0, *src0_hi) ||
      raw_source_pair_reads_vdst_pair(vdst, src1, *src1_hi))
    return false;

  words.reserve(literal ? 15 : 11);
  append_raw_v_mul_u64_low64(words, vdst, static_cast<uint8_t>(vdst + 1u), src0, *src0_hi, src1,
                             *src1_hi, literal);
  return true;
}

[[nodiscard]] std::vector<uint32_t>
lower_raw_gfx1250_v_mul_u64_vop3(const uint32_t *raw, uint32_t inst_size,
                                 uint32_t *source_size = nullptr) {
  if (!raw || inst_size < 2 * sizeof(uint32_t))
    return {};

  const uint32_t w0 = raw[0];
  if ((w0 >> 26) != kVop3Encoding || ((w0 >> 16) & 0x3FFu) != 0)
    return {};

  const uint8_t vdst = static_cast<uint8_t>(w0 & 0xFFu);
  if (vdst > 254)
    return {};

  const uint32_t w1 = raw[1];
  const uint16_t src0 = static_cast<uint16_t>(w1 & 0x1FFu);
  const uint16_t src1 = static_cast<uint16_t>((w1 >> 9) & 0x1FFu);
  if (src0 == 254 || src1 == 254)
    return {};
  const bool uses_literal = src0 == 255 || src1 == 255;
  if (uses_literal && inst_size < 3 * sizeof(uint32_t))
    return {};
  if (source_size)
    *source_size = uses_literal ? 3 * sizeof(uint32_t) : 2 * sizeof(uint32_t);

  std::vector<uint32_t> words;
  if (!append_raw_v_mul_u64_replacement(
          words, vdst, src0, src1, uses_literal ? std::optional<uint32_t>(raw[2]) : std::nullopt))
    return {};
  return words;
}

[[nodiscard]] std::vector<uint32_t>
lower_raw_gfx1250_v_mul_u64_vop3(const uint32_t *raw, uint32_t inst_size, const Instruction &inst,
                                 const LivenessAnalysis &liveness,
                                 uint32_t *source_size = nullptr) {
  if (!raw || inst_size < 2 * sizeof(uint32_t))
    return {};

  const uint32_t w0 = raw[0];
  if ((w0 >> 26) != kVop3Encoding || ((w0 >> 16) & 0x3FFu) != 0)
    return {};

  const uint8_t vdst = static_cast<uint8_t>(w0 & 0xFFu);
  if (vdst > 254)
    return {};

  const uint32_t w1 = raw[1];
  const uint16_t src0 = static_cast<uint16_t>(w1 & 0x1FFu);
  const uint16_t src1 = static_cast<uint16_t>((w1 >> 9) & 0x1FFu);
  if (src0 == 254 || src1 == 254)
    return {};
  const bool uses_literal = src0 == 255 || src1 == 255;
  if (uses_literal && inst_size < 3 * sizeof(uint32_t))
    return {};
  if (source_size)
    *source_size = uses_literal ? 3 * sizeof(uint32_t) : 2 * sizeof(uint32_t);

  const std::optional<uint32_t> literal =
      uses_literal ? std::optional<uint32_t>(raw[2]) : std::nullopt;
  std::vector<uint32_t> words;
  if (append_raw_v_mul_u64_replacement(words, vdst, src0, src1, literal))
    return words;
  words.clear();
  if (append_raw_v_mul_u64_low_scratch_replacement(words, vdst, src0, src1, literal, inst,
                                                   liveness))
    return words;
  words.clear();
  if (append_raw_v_mul_u64_high_scratch_replacement(words, vdst, src0, src1, literal, inst,
                                                    liveness))
    return words;
  return {};
}

[[nodiscard]] std::vector<uint32_t>
lower_raw_gfx1250_v_mul_u64_vop3(std::span<const uint8_t> text, uint64_t offset,
                                 const Instruction &inst, const LivenessAnalysis &liveness,
                                 uint32_t &source_size) {
  if (offset + 2 * sizeof(uint32_t) > text.size())
    return {};

  uint32_t raw[3] = {};
  std::memcpy(raw, text.data() + offset, 2 * sizeof(uint32_t));
  const uint16_t src0 = static_cast<uint16_t>(raw[1] & 0x1FFu);
  const uint16_t src1 = static_cast<uint16_t>((raw[1] >> 9) & 0x1FFu);
  const uint32_t available_size =
      (src0 == 255 || src1 == 255) && offset + 3 * sizeof(uint32_t) <= text.size()
          ? 3 * sizeof(uint32_t)
          : 2 * sizeof(uint32_t);
  if (available_size == 3 * sizeof(uint32_t))
    std::memcpy(raw + 2, text.data() + offset + 2 * sizeof(uint32_t), sizeof(uint32_t));

  uint32_t consumed_size = 0;
  auto words =
      lower_raw_gfx1250_v_mul_u64_vop3(raw, available_size, inst, liveness, &consumed_size);
  if (!words.empty())
    source_size = consumed_size;
  return words;
}

[[nodiscard]] std::vector<uint32_t>
lower_raw_gfx1250_v_mul_u64_e32(const uint32_t *raw, uint32_t inst_size, const Instruction &inst,
                                const LivenessAnalysis &liveness, uint32_t *source_size = nullptr) {
  if (!raw || inst_size < sizeof(uint32_t))
    return {};

  const uint32_t w0 = raw[0];
  if ((w0 >> 26) == kVop3Encoding)
    return {};
  if (((w0 >> 25) & 0x3Fu) != kGfx1250VMulU64E32Opcode)
    return {};

  const uint16_t src0 = static_cast<uint16_t>(w0 & 0x1FFu);
  if (src0 == 254)
    return {};
  const bool uses_literal = src0 == 255;
  if (uses_literal && inst_size < 2 * sizeof(uint32_t))
    return {};
  if (source_size)
    *source_size = uses_literal ? 2 * sizeof(uint32_t) : sizeof(uint32_t);

  const uint8_t vsrc1 = static_cast<uint8_t>((w0 >> 9) & 0xFFu);
  const uint8_t vdst = static_cast<uint8_t>((w0 >> 17) & 0xFFu);
  if (vdst > 254)
    return {};
  const uint16_t src1 = static_cast<uint16_t>(256u + vsrc1);
  const std::optional<uint32_t> literal =
      uses_literal ? std::optional<uint32_t>(raw[1]) : std::nullopt;

  std::vector<uint32_t> words;
  if (append_raw_v_mul_u64_replacement(words, vdst, src0, src1, literal))
    return words;
  words.clear();
  if (append_raw_v_mul_u64_low_scratch_replacement(words, vdst, src0, src1, literal, inst,
                                                   liveness))
    return words;
  words.clear();
  if (append_raw_v_mul_u64_high_scratch_replacement(words, vdst, src0, src1, literal, inst,
                                                    liveness))
    return words;
  return {};
}

[[nodiscard]] std::vector<uint32_t>
lower_raw_gfx1250_v_mul_u64_e32(std::span<const uint8_t> text, uint64_t offset,
                                const Instruction &inst, const LivenessAnalysis &liveness,
                                uint32_t &source_size) {
  if (offset + sizeof(uint32_t) > text.size())
    return {};

  uint32_t raw[2] = {};
  std::memcpy(raw, text.data() + offset, sizeof(uint32_t));
  const uint16_t src0 = static_cast<uint16_t>(raw[0] & 0x1FFu);
  const uint32_t available_size = src0 == 255 && offset + 2 * sizeof(uint32_t) <= text.size()
                                      ? 2 * sizeof(uint32_t)
                                      : sizeof(uint32_t);
  if (available_size == 2 * sizeof(uint32_t))
    std::memcpy(raw + 1, text.data() + offset + sizeof(uint32_t), sizeof(uint32_t));

  uint32_t consumed_size = 0;
  auto words = lower_raw_gfx1250_v_mul_u64_e32(raw, available_size, inst, liveness, &consumed_size);
  if (!words.empty())
    source_size = consumed_size;
  return words;
}

void grow_required_vgpr_count_for_raw_gfx1250_v_mul_u64(const Instruction &inst,
                                                        uint32_t &minimum_vgprs) {
  const uint32_t *raw = inst.raw_encoding();
  if (raw == nullptr || inst.size() < static_cast<int>(sizeof(uint32_t)))
    return;

  const uint32_t w0 = raw[0];
  const uint8_t vdst = static_cast<uint8_t>(w0 & 0xFFu);
  if ((w0 >> 26) == kVop3Encoding && ((w0 >> 16) & 0x3FFu) == 0u &&
      inst.size() >= static_cast<int>(2 * sizeof(uint32_t))) {
    if (vdst <= 254u)
      minimum_vgprs = std::max(minimum_vgprs, static_cast<uint32_t>(vdst + 2u));
    const uint32_t w1 = raw[1];
    const uint16_t src0 = static_cast<uint16_t>(w1 & 0x1FFu);
    const uint16_t src1 = static_cast<uint16_t>((w1 >> 9) & 0x1FFu);
    const bool uses_literal = src0 == 255 || src1 == 255;
    const std::optional<uint32_t> literal =
        uses_literal && inst.size() >= static_cast<int>(3 * sizeof(uint32_t))
            ? std::optional<uint32_t>(raw[2])
            : std::nullopt;
    grow_required_vgpr_count_for_src(minimum_vgprs, src0);
    grow_required_vgpr_count_for_src(minimum_vgprs, src1);
    if (auto src0_hi = raw_pair_hi_src_with_literal(src0, literal))
      grow_required_vgpr_count_for_src(minimum_vgprs, *src0_hi);
    if (auto src1_hi = raw_pair_hi_src_with_literal(src1, literal))
      grow_required_vgpr_count_for_src(minimum_vgprs, *src1_hi);
    return;
  }

  if ((w0 >> 26) == kVop3Encoding)
    return;
  if (((w0 >> 25) & 0x3Fu) != kGfx1250VMulU64E32Opcode)
    return;

  const uint8_t e32_vdst = static_cast<uint8_t>((w0 >> 17) & 0xFFu);
  if (e32_vdst <= 254u)
    minimum_vgprs = std::max(minimum_vgprs, static_cast<uint32_t>(e32_vdst + 2u));
  const uint16_t src0 = static_cast<uint16_t>(w0 & 0x1FFu);
  const uint16_t src1 = static_cast<uint16_t>(256u + ((w0 >> 9) & 0xFFu));
  const std::optional<uint32_t> literal =
      src0 == 255 && inst.size() >= static_cast<int>(2 * sizeof(uint32_t))
          ? std::optional<uint32_t>(raw[1])
          : std::nullopt;
  grow_required_vgpr_count_for_src(minimum_vgprs, src0);
  grow_required_vgpr_count_for_src(minimum_vgprs, src1);
  if (auto src0_hi = raw_pair_hi_src_with_literal(src0, literal))
    grow_required_vgpr_count_for_src(minimum_vgprs, *src0_hi);
  if (auto src1_hi = raw_pair_hi_src_with_literal(src1, literal))
    grow_required_vgpr_count_for_src(minimum_vgprs, *src1_hi);
}

[[nodiscard]] bool
gfx1250_v_mul_u64_needs_high_bank_scratch(const Instruction &inst,
                                          const LivenessAnalysis *liveness = nullptr) {
  const uint32_t *raw = inst.raw_encoding();
  if (raw == nullptr || inst.size() < static_cast<int>(sizeof(uint32_t)))
    return false;

  const uint32_t w0 = raw[0];
  if ((w0 >> 26) == kVop3Encoding) {
    if (((w0 >> 16) & 0x3FFu) != 0u || inst.size() < static_cast<int>(2 * sizeof(uint32_t)))
      return false;

    const uint8_t vdst = static_cast<uint8_t>(w0 & 0xFFu);
    if (vdst > 254u)
      return false;

    const uint32_t w1 = raw[1];
    const uint16_t src0 = static_cast<uint16_t>(w1 & 0x1FFu);
    const uint16_t src1 = static_cast<uint16_t>((w1 >> 9) & 0x1FFu);
    const bool uses_literal = src0 == 255 || src1 == 255;
    const std::optional<uint32_t> literal =
        uses_literal && inst.size() >= static_cast<int>(3 * sizeof(uint32_t))
            ? std::optional<uint32_t>(raw[2])
            : std::nullopt;
    const auto src0_hi = raw_pair_hi_src_with_literal(src0, literal);
    const auto src1_hi = raw_pair_hi_src_with_literal(src1, literal);
    if (!src0_hi || !src1_hi)
      return false;
    const bool overlaps = raw_source_pair_reads_vdst_pair(vdst, src0, *src0_hi) ||
                          raw_source_pair_reads_vdst_pair(vdst, src1, *src1_hi);
    if (!overlaps)
      return false;
    return liveness == nullptr ||
           !find_raw_v_mul_u64_low_scratch(vdst, src0, *src0_hi, src1, *src1_hi, inst, *liveness);
  }

  if (((w0 >> 25) & 0x3Fu) != kGfx1250VMulU64E32Opcode)
    return false;

  const uint8_t vdst = static_cast<uint8_t>((w0 >> 17) & 0xFFu);
  if (vdst > 254u)
    return false;

  const uint16_t src0 = static_cast<uint16_t>(w0 & 0x1FFu);
  const uint16_t src1 = static_cast<uint16_t>(256u + ((w0 >> 9) & 0xFFu));
  const std::optional<uint32_t> literal =
      src0 == 255 && inst.size() >= static_cast<int>(2 * sizeof(uint32_t))
          ? std::optional<uint32_t>(raw[1])
          : std::nullopt;
  const auto src0_hi = raw_pair_hi_src_with_literal(src0, literal);
  const auto src1_hi = raw_pair_hi_src_with_literal(src1, literal);
  if (!src0_hi || !src1_hi)
    return false;
  const bool overlaps = raw_source_pair_reads_vdst_pair(vdst, src0, *src0_hi) ||
                        raw_source_pair_reads_vdst_pair(vdst, src1, *src1_hi);
  if (!overlaps)
    return false;
  return liveness == nullptr ||
         !find_raw_v_mul_u64_low_scratch(vdst, src0, *src0_hi, src1, *src1_hi, inst, *liveness);
}

void append_hardware_pending_warning(std::vector<std::string> *warnings,
                                     std::string_view mnemonic) {
  if (!warnings || !has_hardware_pending_semantic_lowering(mnemonic))
    return;
  const std::string warning = "hardware validation pending for " + std::string(mnemonic);
  if (std::find(warnings->begin(), warnings->end(), warning) == warnings->end())
    warnings->push_back(warning);
}

[[nodiscard]] bool cave_diagnostics_enabled() {
  const char *enabled = std::getenv("ROCJITSU_DBT_CAVE_DIAGNOSTICS");
  return enabled != nullptr && enabled[0] != '\0';
}

[[nodiscard]] std::string cave_range_diagnostic(const char *kind, const SemanticReplacement &repl,
                                                uint64_t cave_byte_offset, uint32_t target_size) {
  std::ostringstream os;
  os << kind;
  if (!repl.source_mnemonic.empty())
    os << " source_mnemonic=" << repl.source_mnemonic;
  os << " source_offset=0x" << std::hex << repl.start_offset << " source_size=0x"
     << (repl.end_offset - repl.start_offset) << " target_size=0x" << target_size
     << " cave_offset=0x" << cave_byte_offset;
  return os.str();
}

[[nodiscard]] bool requires_semantic_expansion(rj_code_arch_t guest, const Instruction &inst) {
  if (guest != ROCJITSU_CODE_ARCH_GFX1250)
    return false;
  if (inst.encoding_id() == kGfx1250VopdEncodingId || inst.encoding_id() == kGfx1250Vopd3EncodingId)
    return true;
  if (inst.opcode() == kGfx1250VAddNcU64E32Opcode &&
      inst.encoding_id() >= kGfx1250Vop2AddNcU64EncodingId0 &&
      inst.encoding_id() <= kGfx1250Vop2AddNcU64EncodingId3)
    return true;
  if (inst.opcode() == kGfx1250VSubNcU64E32Opcode &&
      inst.encoding_id() >= kGfx1250Vop2SubNcU64EncodingId0 &&
      inst.encoding_id() <= kGfx1250Vop2SubNcU64EncodingId3)
    return true;
  if (inst.opcode() == kGfx1250VMulU64E32Opcode &&
      inst.encoding_id() >= kGfx1250Vop2MulU64EncodingId0 &&
      inst.encoding_id() <= kGfx1250Vop2MulU64EncodingId3)
    return true;
  return false;
}

[[nodiscard]] bool starts_with(std::string_view value, std::string_view prefix) {
  return value.size() >= prefix.size() && value.substr(0, prefix.size()) == prefix;
}

[[nodiscard]] std::optional<uint16_t> raw_vop3_opcode(uint32_t w0) {
  if ((w0 >> 26) != kVop3Encoding)
    return std::nullopt;
  return static_cast<uint16_t>((w0 >> 16) & 0x3FFu);
}

[[nodiscard]] bool is_gfx1250_vop3_compare_opcode(uint16_t op) {
  return (op >= 1 && op <= 14) || (op >= 17 && op <= 30) || (op >= 33 && op <= 46) ||
         (op >= 49 && op <= 54) || (op >= 57 && op <= 62) || (op >= 65 && op <= 70) ||
         (op >= 73 && op <= 78) || (op >= 81 && op <= 86) || (op >= 89 && op <= 94) ||
         (op >= 129 && op <= 142) || (op >= 145 && op <= 158) || (op >= 161 && op <= 174) ||
         (op >= 177 && op <= 182) || (op >= 185 && op <= 190) || (op >= 193 && op <= 198) ||
         (op >= 201 && op <= 206) || (op >= 209 && op <= 214) || (op >= 217 && op <= 222);
}

[[nodiscard]] bool is_scalar_alu_encoding(uint16_t encoding_id) {
  if (encoding_id == kEnc_SOP1 || encoding_id == kEnc_SOPC)
    return true;
  if ((encoding_id & 0x180u) == kEnc_SOP2)
    return true;
  if ((encoding_id & 0x1E0u) == kEnc_SOPK)
    return true;
  return false;
}

[[nodiscard]] bool has_explicit_sgpr_destination(const Instruction &inst) {
  for (int i = 0; i < inst.num_dst_operands(); ++i) {
    const Operand *operand = inst.dst_operand(i);
    if (!operand)
      continue;
    const auto reg = operand->to_register_ref();
    if (reg && reg->cls == RegClass::SGPR)
      return true;
  }
  return false;
}

[[nodiscard]] std::optional<uint32_t> rdna4_scalar_dependency_barrier_after(
    const Instruction &inst, rj_code_arch_t host_arch) {
  if (inst.is_branch() || inst.is_barrier() || inst.is_waitcnt() || inst.encoding_id() == kEnc_SOPP)
    return std::nullopt;
  if (inst.encoding_id() == kEnc_SOPC)
    return build_s_delay_alu(kDelayAluSaluDep1, host_arch);
  if (is_scalar_alu_encoding(inst.encoding_id()))
    return build_s_wait_alu(kWaitAluDepctrSaSdst0, host_arch);

  const std::string_view mnemonic = inst.mnemonic();
  if (starts_with(mnemonic, "v_cmpx"))
    return build_s_wait_alu(kWaitAluDepctrVaSdst0, host_arch);
  if (starts_with(mnemonic, "v_cmp")) {
    if (has_explicit_sgpr_destination(inst))
      return build_s_wait_alu(kWaitAluDepctrVaSdst0, host_arch);
    return build_s_wait_alu(kWaitAluDepctrVaVcc0, host_arch);
  }
  if (starts_with(mnemonic, "v_readfirstlane_b32") || starts_with(mnemonic, "v_readlane_b32"))
    return build_s_wait_alu(kWaitAluDepctrVaSdst0, host_arch);
  if (starts_with(mnemonic, "v_") && has_explicit_sgpr_destination(inst))
    return build_s_wait_alu(kWaitAluDepctrVaSdst0, host_arch);
  return std::nullopt;
}

[[nodiscard]] bool rdna4_memory_dependency_wait_before(const Instruction &inst) {
  const uint16_t enc = inst.encoding_id();
  if ((enc & 0x1FEu) == kEnc_VFLAT || (enc & 0x1FEu) == kEnc_VSCRATCH ||
      (enc & 0x1FEu) == kEnc_VGLOBAL)
    return true;
  if ((enc & 0x1F8u) == kEnc_VBUFFER || (enc & 0x1F8u) == kEnc_VDS)
    return true;
  return false;
}

void fill_nops(std::vector<uint8_t> &text, uint64_t offset, uint32_t size,
               rj_code_arch_t host_arch) {
  const uint32_t nop = build_s_nop(0, host_arch);
  for (uint32_t i = 0; i < size; i += sizeof(nop))
    std::memcpy(text.data() + offset + i, &nop, sizeof(nop));
}

[[nodiscard]] std::vector<uint32_t> nop_words(uint32_t size, rj_code_arch_t host_arch) {
  assert(size % sizeof(uint32_t) == 0 && "instruction size must be word aligned");
  return std::vector<uint32_t>(size / sizeof(uint32_t), build_s_nop(0, host_arch));
}

[[nodiscard]] std::vector<uint32_t> copy_instruction_words(std::span<const uint8_t> text,
                                                           uint64_t offset, uint32_t size) {
  assert(size % sizeof(uint32_t) == 0 && "instruction size must be word aligned");
  assert(offset + size <= text.size() && "instruction exceeds text bounds");
  std::vector<uint32_t> words(size / sizeof(uint32_t));
  std::memcpy(words.data(), text.data() + offset, size);
  return words;
}

[[nodiscard]] bool ranges_overlap(uint64_t lhs_start, uint64_t lhs_end, uint64_t rhs_start,
                                  uint64_t rhs_end) {
  return lhs_start < rhs_end && rhs_start < lhs_end;
}

[[nodiscard]] bool overlaps_any_range(uint64_t start, uint64_t end,
                                      std::span<const std::pair<uint64_t, uint64_t>> ranges) {
  return std::ranges::any_of(ranges, [&](const auto &range) {
    return ranges_overlap(start, end, range.first, range.second);
  });
}

[[nodiscard]] bool is_local_cave_padding_run(std::span<const uint8_t> text, uint64_t start,
                                             uint64_t size) {
  for (uint64_t off = start; off < start + size; off += sizeof(uint32_t)) {
    const uint32_t word = read_u32(text, off);
    if (word != kSCodeEnd && word != kSNop0)
      return false;
  }
  return true;
}

[[nodiscard]] std::optional<uint64_t>
find_local_text_cave(std::span<const uint8_t> text, const SemanticReplacement &repl,
                     uint64_t cave_size, std::span<const std::pair<uint64_t, uint64_t>> local_caves,
                     std::span<const std::pair<uint64_t, uint64_t>> protected_ranges,
                     bool allow_unreachable_text_caves, int16_t &entry_branch_dwords,
                     int16_t &return_branch_dwords) {
  if (cave_size == 0 || cave_size % sizeof(uint32_t) != 0 || cave_size > text.size())
    return std::nullopt;

  const uint64_t source_size = repl.end_offset - repl.start_offset;
  const uint64_t stub_next = repl.start_offset + source_size;
  const uint64_t last_start = text.size() - cave_size;

  auto try_candidate = [&](uint64_t candidate) -> std::optional<uint64_t> {
    const uint64_t candidate_end = candidate + cave_size;
    if (ranges_overlap(candidate, candidate_end, repl.start_offset, repl.end_offset))
      return std::nullopt;
    if (overlaps_any_range(candidate, candidate_end, local_caves) ||
        overlaps_any_range(candidate, candidate_end, protected_ranges)) {
      return std::nullopt;
    }
    if (!allow_unreachable_text_caves && !is_local_cave_padding_run(text, candidate, cave_size))
      return std::nullopt;

    int16_t fwd_dwords = 0;
    if (!compute_sopp_branch_offset(repl.start_offset, candidate, fwd_dwords))
      return std::nullopt;

    int16_t ret_dwords = 0;
    const uint64_t return_branch_pc = candidate + repl.target_words.size() * sizeof(uint32_t);
    if (!compute_sopp_branch_offset(return_branch_pc, stub_next, ret_dwords))
      return std::nullopt;

    entry_branch_dwords = fwd_dwords;
    return_branch_dwords = ret_dwords;
    return candidate;
  };

  if (allow_unreachable_text_caves) {
    std::vector<std::pair<uint64_t, uint64_t>> gaps;
    uint64_t cursor = 0;
    for (const auto &[range_start, range_end] : protected_ranges) {
      if (range_start > cursor)
        gaps.emplace_back(cursor, range_start);
      cursor = std::max(cursor, range_end);
    }
    if (cursor < text.size())
      gaps.emplace_back(cursor, text.size());

    auto scan_forward = [&](uint64_t start, uint64_t end) -> std::optional<uint64_t> {
      if (end < start || end - start < cave_size)
        return std::nullopt;
      for (uint64_t candidate = align_up_to_word(start); candidate + cave_size <= end;
           candidate += sizeof(uint32_t)) {
        if (auto cave = try_candidate(candidate))
          return cave;
      }
      return std::nullopt;
    };

    auto scan_backward = [&](uint64_t start, uint64_t end) -> std::optional<uint64_t> {
      if (end < start || end - start < cave_size || repl.start_offset < cave_size)
        return std::nullopt;
      uint64_t candidate = std::min(end - cave_size, repl.start_offset - cave_size);
      candidate -= candidate % sizeof(uint32_t);
      while (candidate >= start) {
        if (auto cave = try_candidate(candidate))
          return cave;
        if (candidate < start + sizeof(uint32_t))
          break;
        candidate -= sizeof(uint32_t);
      }
      return std::nullopt;
    };

    const uint64_t forward_start = align_up_to_word(repl.end_offset);
    for (const auto &[gap_start, gap_end] : gaps) {
      if (gap_end <= forward_start)
        continue;
      if (auto cave = scan_forward(std::max(gap_start, forward_start), gap_end))
        return cave;
    }

    for (auto it = gaps.rbegin(); it != gaps.rend(); ++it) {
      if (it->first >= repl.start_offset)
        continue;
      if (auto cave = scan_backward(it->first, std::min(it->second, repl.start_offset)))
        return cave;
    }

    return std::nullopt;
  }

  for (uint64_t candidate = align_up_to_word(repl.end_offset); candidate <= last_start;
       candidate += sizeof(uint32_t)) {
    if (auto cave = try_candidate(candidate))
      return cave;
  }

  if (repl.start_offset >= cave_size) {
    uint64_t candidate = std::min(last_start, repl.start_offset - cave_size);
    candidate -= candidate % sizeof(uint32_t);
    while (true) {
      if (auto cave = try_candidate(candidate))
        return cave;
      if (candidate < sizeof(uint32_t))
        break;
      candidate -= sizeof(uint32_t);
    }
  }

  return std::nullopt;
}

[[nodiscard]] std::vector<uint64_t> kernel_entry_offsets(std::span<const KdTranslation> kernels) {
  std::vector<uint64_t> offsets;
  offsets.reserve(kernels.size());
  for (const KdTranslation &kernel : kernels)
    offsets.push_back(kernel.entry_text_offset);

  std::ranges::sort(offsets);
  offsets.erase(std::ranges::unique(offsets).begin(), offsets.end());
  return offsets;
}

struct KernelTranslationScope {
  KdTranslation *translation = nullptr;
  BasicBlock *entry = nullptr;
  std::vector<BasicBlock *> blocks;
};

using HighBankBlockModeMap = std::unordered_map<BasicBlock *, uint8_t>;

[[nodiscard]] uint8_t gfx1250_high_bank_mode_after_block(BasicBlock &block, uint8_t mode) {
  for (const Instruction &inst : block.instructions()) {
    const uint32_t *raw = inst.raw_encoding();
    if (raw == nullptr || inst.size() < static_cast<int>(sizeof(uint32_t)))
      continue;
    uint16_t simm16 = 0;
    if (is_gfx1250_s_set_vgpr_msb(raw[0], simm16))
      mode = amdgpu::s_set_vgpr_msb_new_mode(simm16);
  }
  return mode;
}

[[nodiscard]] std::optional<HighBankBlockModeMap>
gfx1250_high_bank_entry_modes_for_scope(const KernelTranslationScope &scope) {
  if (scope.entry == nullptr)
    return std::nullopt;

  std::unordered_set<BasicBlock *> scope_blocks;
  scope_blocks.reserve(scope.blocks.size());
  for (BasicBlock *block : scope.blocks) {
    if (block != nullptr)
      scope_blocks.insert(block);
  }

  HighBankBlockModeMap entry_modes;
  std::vector<BasicBlock *> worklist;
  entry_modes.emplace(scope.entry, 0);
  worklist.push_back(scope.entry);

  while (!worklist.empty()) {
    BasicBlock *block = worklist.back();
    worklist.pop_back();
    if (block == nullptr)
      continue;

    const auto mode_it = entry_modes.find(block);
    if (mode_it == entry_modes.end())
      continue;
    const uint8_t exit_mode = gfx1250_high_bank_mode_after_block(*block, mode_it->second);

    for (BasicBlock *succ : block->successors()) {
      if (succ == nullptr || !scope_blocks.contains(succ))
        continue;
      auto [it, inserted] = entry_modes.emplace(succ, exit_mode);
      if (!inserted) {
        if (it->second != exit_mode)
          return std::nullopt;
        continue;
      }
      worklist.push_back(succ);
    }
  }

  for (BasicBlock *block : scope.blocks) {
    if (block != nullptr && !entry_modes.contains(block))
      return std::nullopt;
  }

  return entry_modes;
}

[[nodiscard]] std::optional<HighBankShadowPlan>
gfx1250_high_bank_shadow_plan_for_scope(const KernelTranslationScope &scope,
                                        const LivenessAnalysis &liveness,
                                        const HighBankBlockModeMap &entry_modes) {
  ShadowFootprint footprint;
  for (BasicBlock *block : scope.blocks) {
    if (block == nullptr)
      continue;
    const auto mode_it = entry_modes.find(block);
    if (mode_it == entry_modes.end())
      return std::nullopt;
    uint8_t mode = mode_it->second;
    for (const Instruction &inst : block->instructions()) {
      record_low_vgpr_uses(inst, footprint);
      const uint32_t *raw = inst.raw_encoding();
      if (raw == nullptr || inst.size() < static_cast<int>(sizeof(uint32_t)))
        continue;
      uint16_t simm16 = 0;
      if (is_gfx1250_s_set_vgpr_msb(raw[0], simm16)) {
        mode = amdgpu::s_set_vgpr_msb_new_mode(simm16);
        continue;
      }
      record_high_bank_vop_footprint(inst, mode, footprint);
    }
  }

  if (!footprint.needed)
    return std::nullopt;
  const uint16_t count = static_cast<uint16_t>(footprint.max_logical_vgpr + 1u);
  auto base = find_shadow_base(footprint, liveness, count);
  if (!base)
    return std::nullopt;
  return HighBankShadowPlan{*base, static_cast<uint8_t>(count),
                            shadow_window_overlaps_low_uses(footprint, *base, count)};
}

struct HighBankScratchPlan {
  uint16_t encoded_base = 0;
  uint32_t minimum_vgprs = 0;
};

[[nodiscard]] bool scope_contains_gfx1250_k128_fp8_wmma(const KernelTranslationScope &scope) {
  for (BasicBlock *block : scope.blocks) {
    if (block == nullptr)
      continue;
    for (const Instruction &inst : block->instructions()) {
      if (is_gfx1250_k128_fp8_wmma(inst))
        return true;
    }
  }
  return false;
}

[[nodiscard]] bool scope_contains_gfx1250_k32_f16_wmma(const KernelTranslationScope &scope) {
  for (BasicBlock *block : scope.blocks) {
    if (block == nullptr)
      continue;
    for (const Instruction &inst : block->instructions()) {
      if (is_gfx1250_k32_f16_wmma(inst))
        return true;
    }
  }
  return false;
}

[[nodiscard]] bool
scope_contains_gfx1250_v_mul_u64_needing_high_scratch(const KernelTranslationScope &scope,
                                                      const LivenessAnalysis &liveness) {
  for (BasicBlock *block : scope.blocks) {
    if (block == nullptr)
      continue;
    for (const Instruction &inst : block->instructions()) {
      if (gfx1250_v_mul_u64_needs_high_bank_scratch(inst, &liveness))
        return true;
    }
  }
  return false;
}

[[nodiscard]] uint32_t
gfx1250_high_bank_scratch_count_for_scope(const KernelTranslationScope &scope,
                                          const LivenessAnalysis &liveness) {
  uint32_t scratch_count = 0;
  if (scope_contains_gfx1250_v_mul_u64_needing_high_scratch(scope, liveness))
    scratch_count = std::max(scratch_count, kGfx1250VMulU64HighBankScratchCount);
  return scratch_count;
}

[[nodiscard]] uint32_t gfx1250_private_scratch_bytes_for_scope(
    const KernelTranslationScope &scope,
    const std::optional<HighBankShadowPlan> &high_bank_shadow_plan = std::nullopt) {
  uint32_t bytes =
      scope_contains_gfx1250_k128_fp8_wmma(scope) ? kGfx1250K128Fp8PrivateScratchBytes : 0;
  if (high_bank_shadow_plan)
    bytes = std::max(bytes, high_bank_shadow_private_scratch_bytes(*high_bank_shadow_plan));
  return bytes;
}

[[nodiscard]] std::optional<HighBankScratchPlan>
gfx1250_high_bank_scratch_plan(const KdTranslation &translation, uint32_t scratch_count) {
  if (scratch_count == 0)
    return std::nullopt;
  const uint32_t physical_base = std::max(translation.guest_vgpr_count, kGfx1250HighBankBaseVgpr);
  const uint32_t minimum_vgprs = physical_base + scratch_count;
  if (physical_base < kGfx1250HighBankBaseVgpr || minimum_vgprs > kRdna4MaxVgprsPerWave)
    return std::nullopt;
  return HighBankScratchPlan{static_cast<uint16_t>(physical_base - kGfx1250HighBankBaseVgpr),
                             minimum_vgprs};
}

void grow_required_vgpr_count_for_register_set(const RegisterSet &registers,
                                               uint32_t &minimum_vgprs) {
  registers.for_each([&](RegisterRef ref) {
    if (ref.cls != RegClass::VGPR)
      return;
    minimum_vgprs = std::max(minimum_vgprs, static_cast<uint32_t>(ref.index + 1u));
  });
}

void grow_required_sgpr_count_for_register_set(const RegisterSet &registers,
                                               uint32_t &minimum_sgprs) {
  registers.for_each([&](RegisterRef ref) {
    if (ref.cls != RegClass::SGPR)
      return;
    minimum_sgprs = std::max(minimum_sgprs, static_cast<uint32_t>(ref.index + 1u));
  });
}

[[nodiscard]] uint32_t gfx1250_rdna4_semantic_tmp_vgpr_count(const Instruction &inst) {
  if (inst.encoding_id() != kGfx1250Vop3pEncodingId)
    return 0;

  switch (inst.opcode()) {
  case kGfx1250WmmaI32Iu8K64Opcode:
    // Worst-case K64 i8 lowering uses an aligned scratch accumulator plus A/B
    // relayout and lane-xor address temporaries.
    return 13;
  case kGfx1250SwmmacI32Iu8K128Opcode:
    return 15;
  case kGfx1250SwmmacF32F16K64Opcode:
    return 21;
  default:
    return 0;
  }
}

void grow_gfx1250_rdna4_semantic_tmp_vgpr_count(const Instruction &inst,
                                                uint32_t &max_tmp_count) {
  const uint32_t tmp_count = gfx1250_rdna4_semantic_tmp_vgpr_count(inst);
  if (tmp_count == 0)
    return;
  max_tmp_count = std::max(max_tmp_count, tmp_count);
}

[[nodiscard]] constexpr uint32_t align_up_vgpr_count(uint32_t count, uint32_t alignment) {
  return ((count + alignment - 1u) / alignment) * alignment;
}

[[nodiscard]] uint32_t
required_vgpr_count_for_gfx1250_rdna4_scope(const KernelTranslationScope &scope) {
  uint32_t minimum_vgprs = 0;
  uint32_t semantic_tmp_vgprs = 0;
  for (BasicBlock *block : scope.blocks) {
    if (block == nullptr)
      continue;
    for (const Instruction &inst : block->instructions()) {
      InstDefUse du(inst);
      grow_required_vgpr_count_for_register_set(du.defs, minimum_vgprs);
      grow_required_vgpr_count_for_register_set(du.uses, minimum_vgprs);
      grow_required_vgpr_count_for_raw_gfx1250_v_mul_u64(inst, minimum_vgprs);
      grow_gfx1250_rdna4_semantic_tmp_vgpr_count(inst, semantic_tmp_vgprs);
    }
  }
  if (semantic_tmp_vgprs != 0) {
    // Semantic expansion allocates temporary runs after liveness has reserved
    // source-level live registers. Dense IREE kernels can leave only a high run
    // available, so the launch descriptor must cover that emitted run too.
    minimum_vgprs =
        std::max(minimum_vgprs, align_up_vgpr_count(minimum_vgprs, 8) + semantic_tmp_vgprs);
  }
  return minimum_vgprs;
}

[[nodiscard]] uint32_t
required_sgpr_count_for_gfx1250_rdna4_scope(const KernelTranslationScope &scope) {
  uint32_t minimum_sgprs = 0;
  for (BasicBlock *block : scope.blocks) {
    if (block == nullptr)
      continue;
    for (const Instruction &inst : block->instructions()) {
      InstDefUse du(inst);
      grow_required_sgpr_count_for_register_set(du.defs, minimum_sgprs);
      grow_required_sgpr_count_for_register_set(du.uses, minimum_sgprs);
    }
  }
  return minimum_sgprs;
}

void configure_liveness_scratch(rj_code_arch_t guest_arch, rj_code_arch_t host_arch,
                                const KernelTranslationScope &scope, LivenessAnalysis &liveness) {
  if (guest_arch != ROCJITSU_CODE_ARCH_GFX1250 || host_arch != ROCJITSU_CODE_ARCH_RDNA4 ||
      scope.translation == nullptr)
    return;
  liveness.set_allocatable_sgpr_limit(static_cast<uint16_t>(kRdna4MaxSgprsPerWave));
  const uint32_t reserved_source_sgprs =
      std::max(scope.translation->target_abi_sgpr_count,
               scope.translation->target_source_sgpr_count);
  if (reserved_source_sgprs != 0) {
    const auto reserved_count = static_cast<uint16_t>(
        std::min<uint32_t>(reserved_source_sgprs,
                           static_cast<uint32_t>(REGISTER_SET_ALLOCATABLE_SGPRS)));
    liveness.reserve_scratch_registers({RegClass::SGPR, 0, static_cast<uint8_t>(reserved_count)});
  }
  if (scope.translation->rdna4_grid_x_sgpr >= 0) {
    liveness.reserve_scratch_registers(
        {RegClass::SGPR, static_cast<uint16_t>(scope.translation->rdna4_grid_x_sgpr), 1});
  }
  const uint32_t scratch_count = gfx1250_high_bank_scratch_count_for_scope(scope, liveness);
  if (auto plan = gfx1250_high_bank_scratch_plan(*scope.translation, scratch_count)) {
    liveness.set_high_vgpr_scratch_base(plan->encoded_base);
  }
  if (scope.translation->private_spill_zone_bytes >= kGfx1250K128Fp8PrivateScratchBytes) {
    liveness.set_private_spill_zone(scope.translation->private_spill_zone_base,
                                    scope.translation->private_spill_zone_bytes);
  }
}

[[nodiscard]] std::vector<KernelDescriptorResourceOverride>
descriptor_resource_overrides_for_scopes(rj_code_arch_t guest_arch, rj_code_arch_t host_arch,
                                         std::span<const KernelTranslationScope> scopes) {
  std::vector<KernelDescriptorResourceOverride> overrides;
  if (guest_arch != ROCJITSU_CODE_ARCH_GFX1250 || host_arch != ROCJITSU_CODE_ARCH_RDNA4)
    return overrides;

  const uint32_t lowering_minimum_vgprs =
      conservative_lowering_minimum_vgprs(guest_arch, host_arch);
  for (const KernelTranslationScope &scope : scopes) {
    if (scope.translation == nullptr)
      continue;
    LivenessAnalysis liveness(KernelBlockScope(scope.blocks));
    uint32_t minimum_vgprs = required_vgpr_count_for_gfx1250_rdna4_scope(scope);
    if (scope_contains_gfx1250_k32_f16_wmma(scope))
      minimum_vgprs = std::max(minimum_vgprs, kGfx1250F16WmmaScratchMinimumVgprs);
    const uint32_t minimum_sgprs = required_sgpr_count_for_gfx1250_rdna4_scope(scope);
    const uint32_t scratch_count = gfx1250_high_bank_scratch_count_for_scope(scope, liveness);
    if (const auto plan = gfx1250_high_bank_scratch_plan(*scope.translation, scratch_count)) {
      // Dense IREE kernels frequently leave no contiguous low-VGPR scratch
      // window. Grow only the kernels that need a high-bank scratch fallback.
      minimum_vgprs = std::max(minimum_vgprs, plan->minimum_vgprs);
    }
    std::optional<HighBankShadowPlan> high_bank_shadow_plan;
    if (const auto entry_modes = gfx1250_high_bank_entry_modes_for_scope(scope))
      high_bank_shadow_plan =
          gfx1250_high_bank_shadow_plan_for_scope(scope, liveness, *entry_modes);
    const uint32_t private_scratch_bytes =
        gfx1250_private_scratch_bytes_for_scope(scope, high_bank_shadow_plan);
    if (minimum_vgprs > lowering_minimum_vgprs || minimum_sgprs != 0 ||
        private_scratch_bytes != 0) {
      overrides.push_back({scope.translation->entry_text_offset, minimum_vgprs, minimum_sgprs, 0,
                           private_scratch_bytes});
    }
  }
  return overrides;
}

[[nodiscard]] std::vector<std::pair<uint64_t, uint64_t>>
reachable_code_ranges(std::span<const KernelTranslationScope> scopes) {
  std::vector<std::pair<uint64_t, uint64_t>> ranges;
  for (const KernelTranslationScope &scope : scopes) {
    for (const BasicBlock *block : scope.blocks) {
      if (block == nullptr)
        continue;
      ranges.emplace_back(block->start_offset(), block->end_offset());
    }
  }

  std::ranges::sort(ranges);
  ranges.erase(std::ranges::unique(ranges).begin(), ranges.end());
  return ranges;
}

[[nodiscard]] uint64_t covered_range_bytes(std::span<const std::pair<uint64_t, uint64_t>> ranges) {
  uint64_t covered = 0;
  uint64_t cursor = 0;
  for (const auto &[start, end] : ranges) {
    if (end <= start)
      continue;
    const uint64_t merged_start = std::max(start, cursor);
    if (end > merged_start)
      covered += end - merged_start;
    cursor = std::max(cursor, end);
  }
  return covered;
}

[[nodiscard]] bool
has_reachable_indirect_control_flow(std::span<const KernelTranslationScope> scopes) {
  for (const KernelTranslationScope &scope : scopes) {
    for (BasicBlock *block : scope.blocks) {
      if (block == nullptr)
        continue;
      for (const Instruction &inst : block->instructions()) {
        if ((inst.flags() & (INDIRECT_BRANCH | INDIRECT_CALL)) != 0)
          return true;
      }
    }
  }
  return false;
}

[[nodiscard]] bool
supports_expanded_text_copy(rj_code_arch_t guest_arch, rj_code_arch_t host_arch, uint64_t text_size,
                            uint64_t protected_text_bytes,
                            std::span<const KdTranslation>,
                            std::span<const KernelTranslationScope> scopes) {
  if (guest_arch != ROCJITSU_CODE_ARCH_GFX1250 || host_arch != ROCJITSU_CODE_ARCH_RDNA4)
    return false;
  if (text_size <= kSoppBranchMaxForwardBytes)
    return false;
  if (protected_text_bytes * 2 <= text_size)
    return false;
  if (has_reachable_indirect_control_flow(scopes))
    return false;
  return true;
}

[[nodiscard]] bool has_unprotected_text(std::span<const std::pair<uint64_t, uint64_t>> ranges,
                                        uint64_t text_size) {
  uint64_t cursor = 0;
  for (const auto &[start, end] : ranges) {
    if (start > cursor)
      return true;
    cursor = std::max(cursor, end);
  }
  return cursor < text_size;
}

[[nodiscard]] std::vector<BasicBlock *>
reachable_kernel_blocks(const std::vector<std::unique_ptr<BasicBlock>> &blocks, BasicBlock &entry,
                        const std::unordered_set<uint64_t> &kernel_entries) {
  std::unordered_set<const BasicBlock *> reachable;
  std::vector<BasicBlock *> stack{&entry};

  while (!stack.empty()) {
    BasicBlock *block = stack.back();
    stack.pop_back();
    if (block == nullptr || !reachable.insert(block).second)
      continue;

    for (BasicBlock *succ : block->successors()) {
      if (succ == nullptr)
        continue;
      if (succ->start_offset() != entry.start_offset() &&
          kernel_entries.contains(succ->start_offset()))
        continue;
      stack.push_back(succ);
    }
  }

  std::vector<BasicBlock *> ordered;
  ordered.reserve(reachable.size());
  for (const auto &block : blocks) {
    if (block && reachable.contains(block.get()))
      ordered.push_back(block.get());
  }
  return ordered;
}

[[nodiscard]] std::vector<KernelTranslationScope>
kernel_translation_scopes(const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                          std::span<KdTranslation> kernels) {
  std::vector<KernelTranslationScope> scopes;
  const auto entries = kernel_entry_offsets(kernels);
  if (entries.empty())
    return scopes;

  std::unordered_set<uint64_t> entry_set(entries.begin(), entries.end());
  std::vector<KdTranslation *> ordered_kernels;
  ordered_kernels.reserve(kernels.size());
  std::unordered_set<uint64_t> seen_entries;
  for (KdTranslation &kernel : kernels) {
    if (seen_entries.insert(kernel.entry_text_offset).second)
      ordered_kernels.push_back(&kernel);
  }

  std::ranges::sort(ordered_kernels, [](const auto *lhs, const auto *rhs) {
    return lhs->entry_text_offset < rhs->entry_text_offset;
  });

  scopes.reserve(ordered_kernels.size());
  for (KdTranslation *kernel : ordered_kernels) {
    BasicBlock *entry = block_for_offset(blocks, kernel->entry_text_offset);
    if (entry == nullptr)
      continue;

    scopes.push_back({kernel, entry, reachable_kernel_blocks(blocks, *entry, entry_set)});
  }
  return scopes;
}

} // namespace

BinaryTranslator::~BinaryTranslator() = default;

BinaryTranslator::BinaryTranslator(rj_code_arch_t guest_arch, rj_code_arch_t host_arch,
                                   uint32_t target_mach, BinaryTranslatorOptions options)
    : guest_arch_(guest_arch), host_arch_(host_arch),
      target_mach_(target_mach ? target_mach : elf_mach_for_arch(host_arch)), options_(options),
      encoding_translate_(select_encoding_translator(guest_arch, host_arch)),
      legalization_lookup_(select_legalization(guest_arch, host_arch)),
      semantic_translator_(std::make_unique<SemanticTranslator>(guest_arch, host_arch)) {}

void BinaryTranslator::set_trace_callback(TranslationTraceCallback callback) {
  trace_callback_ = std::move(callback);
}

TranslatedCodeObject BinaryTranslator::translate(const AmdGpuCodeObject &obj) {
  TranslatedCodeObject result;
  result.host_arch = host_arch_;
  warnings_ = &result.warnings;
  diagnostics_ = &result.diagnostics;

  CodeObjectPatcher patcher(obj);
  auto leave_unchanged = [&]() {
    warnings_ = nullptr;
    diagnostics_ = nullptr;
    const auto *image = reinterpret_cast<const uint8_t *>(obj.image_data());
    result.elf_bytes.assign(image, image + obj.image_size());
    return result;
  };
  auto text = patcher.text_bytes();
  if (text.empty()) {
    return leave_unchanged();
  }

  auto decoder = Decoder::create(guest_arch_);
  if (!decoder) {
    append_error(result.diagnostics, DiagnosticKind::UnsupportedGuestArch,
                 "unsupported guest_arch: no decoder available");
    return leave_unchanged();
  }
  KernelDescriptorTranslator descriptor_translator(guest_arch_, host_arch_);
  KernelDescriptorTranslationOptions descriptor_options;
  // Semantic lowerings allocate temporary VGPRs from liveness. Descriptor
  // translation runs before those choices are known, so keep per-lowering
  // headroom for now.
  // TODO: Have lowerings report their actual highest temporary VGPR demand and
  // use that instead of this conservative floor.
  descriptor_options.minimum_vgprs =
      conservative_lowering_minimum_vgprs(guest_arch_, host_arch_);
  auto descriptor_translations = descriptor_translator.translate_image(
      patcher.image_bytes(), patcher.text_offset(), patcher.text_size(), descriptor_options);
  if (descriptor_translations.empty()) {
    append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                 "kernel descriptors are required for kernel-level translation");
    return leave_unchanged();
  }

  auto entry_offsets = kernel_entry_offsets(descriptor_translations);
  auto blocks = BasicBlock::build(obj, *decoder, entry_offsets);
  auto scopes = kernel_translation_scopes(blocks, descriptor_translations);

  if (scopes.size() != entry_offsets.size()) {
    append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                 "kernel descriptor entry offsets are required to map to decoded text blocks");
    return leave_unchanged();
  }

  const auto descriptor_overrides =
      descriptor_resource_overrides_for_scopes(guest_arch_, host_arch_, scopes);
  if (!descriptor_overrides.empty()) {
    descriptor_options.kernel_overrides = std::span<const KernelDescriptorResourceOverride>(
        descriptor_overrides.data(), descriptor_overrides.size());
    descriptor_translations = descriptor_translator.translate_image(
        patcher.image_bytes(), patcher.text_offset(), patcher.text_size(), descriptor_options);
    entry_offsets = kernel_entry_offsets(descriptor_translations);
    scopes = kernel_translation_scopes(blocks, descriptor_translations);
    if (scopes.size() != entry_offsets.size()) {
      append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                   "kernel descriptor entry offsets are required to map to decoded text blocks");
      return leave_unchanged();
    }
  }

  bool descriptors_supported = true;
  for (const auto &translation : descriptor_translations) {
    append_diagnostics(result.diagnostics, translation.diagnostics);
    descriptors_supported &= translation.supported;
  }
  if (!descriptors_supported) {
    append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                 "kernel descriptor translation requires unsupported resource or ABI "
                 "virtualization; leaving code object unchanged");
    return leave_unchanged();
  }

  std::vector<uint8_t> translated_text(text.begin(), text.end());
  const bool continue_after_failure = options_.debug_continue_after_failure;

  auto copy_original_instruction = [&](const Instruction &inst, uint64_t offset) {
    const uint32_t inst_size = inst.size();
    std::memcpy(translated_text.data() + offset, text.data() + offset, inst_size);
    if (!trace_callback_)
      return;

    // Continued-failure mode is diagnostic-only. Emit an explicit copy event so
    // diff reports make it clear which failed source instruction was preserved.
    const auto source_words = raw_words_for_inst(inst);
    trace_callback_({.source_offset = offset,
                     .source_size = inst_size,
                     .source_words = source_words,
                     .legalization = nullptr,
                     .copied_original = true,
                     .semantic_lowering = false,
                     .changed = false,
                     .emitted_in_cave = false,
                     .target_offset = offset,
                     .target_words = source_words});
  };

  auto continue_after_instruction_error = [&](const Instruction &inst, uint64_t offset) {
    if (!continue_after_failure)
      return false;
    copy_original_instruction(inst, offset);
    return true;
  };

  const auto protected_ranges = reachable_code_ranges(scopes);
  const bool allow_unreachable_text_caves = has_unprotected_text(protected_ranges, text.size()) &&
                                            !has_reachable_indirect_control_flow(scopes);
  std::vector<std::pair<uint64_t, uint64_t>> local_caves;

  // Code caves live in a separate executable section that is placed immediately
  // after the original .text bytes. Treating that section as a .text-relative
  // continuation keeps existing instruction addresses stable while avoiding any
  // dependency on compiler-emitted NOP padding after s_endpgm.
  patcher.set_cave_start(text.size());

  if (supports_expanded_text_copy(guest_arch_, host_arch_, text.size(),
                                  covered_range_bytes(protected_ranges), descriptor_translations,
                                  scopes)) {
    struct RelocatedBranch {
      uint64_t word_index = 0;
      uint64_t target_offset = 0;
      std::string mnemonic;
    };

    std::vector<uint32_t> expanded_words;
    std::unordered_map<uint64_t, uint64_t> copied_entry_offsets;
    bool expanded_copy_ok = true;

    auto fail_expanded_copy = [&](std::string message) {
      if (expanded_copy_ok && warnings_)
        warnings_->push_back(std::move(message));
      expanded_copy_ok = false;
    };

    for (const KernelTranslationScope &scope : scopes) {
      if (!expanded_copy_ok)
        break;
      if (scope.blocks.empty() || scope.translation == nullptr)
        continue;

      LivenessAnalysis liveness(KernelBlockScope(scope.blocks));
      configure_liveness_scratch(guest_arch_, host_arch_, scope, liveness);
      std::optional<HighBankBlockModeMap> high_bank_entry_modes;
      if (guest_arch_ == ROCJITSU_CODE_ARCH_GFX1250 && host_arch_ == ROCJITSU_CODE_ARCH_RDNA4) {
        high_bank_entry_modes = gfx1250_high_bank_entry_modes_for_scope(scope);
        if (!high_bank_entry_modes) {
          fail_expanded_copy(
              "expanded text copy cannot determine consistent gfx1250 VGPR MSB mode across CFG");
          break;
        }
      }
      const auto high_bank_shadow_plan =
          guest_arch_ == ROCJITSU_CODE_ARCH_GFX1250 && host_arch_ == ROCJITSU_CODE_ARCH_RDNA4
              ? gfx1250_high_bank_shadow_plan_for_scope(scope, liveness, *high_bank_entry_modes)
              : std::optional<HighBankShadowPlan>{};
      HighBankShadowState high_bank_shadow_state;
      std::vector<uint32_t> scope_words;
      std::vector<uint32_t> scope_word_group_sizes;
      std::vector<RelocatedBranch> direct_branches;
      std::unordered_map<uint64_t, uint64_t> scope_offsets;

      uint64_t consumed_until = 0;
      for (BasicBlock *block : scope.blocks) {
        if (!expanded_copy_ok)
          break;
        if (block == nullptr)
          continue;
        if (guest_arch_ == ROCJITSU_CODE_ARCH_GFX1250 && host_arch_ == ROCJITSU_CODE_ARCH_RDNA4) {
          const auto mode_it = high_bank_entry_modes->find(block);
          if (mode_it == high_bank_entry_modes->end()) {
            fail_expanded_copy(
                "expanded text copy cannot determine gfx1250 VGPR MSB mode for copied block");
            break;
          }
          high_bank_shadow_state.mode = mode_it->second;
        } else {
          high_bank_shadow_state.mode = 0;
        }

        InstructionList &instructions = block->instructions();
        uint64_t offset = block->start_offset();
        for (auto inst_it = instructions.begin(); inst_it != instructions.end(); ++inst_it) {
          const Instruction &inst = *inst_it;
          const uint32_t inst_size = inst.size();
          if (offset < consumed_until) {
            offset += inst_size;
            continue;
          }

          const uint64_t translated_offset = scope_words.size() * sizeof(uint32_t);
          scope_offsets.emplace(offset, translated_offset);

          uint32_t source_size = inst_size;
          std::vector<uint32_t> words;
          if (guest_arch_ == ROCJITSU_CODE_ARCH_GFX1250 && host_arch_ == ROCJITSU_CODE_ARCH_RDNA4) {
            auto shadow_lowering = lower_gfx1250_high_bank_shadow_instruction(
                text, inst, offset, liveness, high_bank_shadow_plan, high_bank_shadow_state);
            if (shadow_lowering.kind == HighBankShadowLoweringKind::Unsupported) {
              fail_expanded_copy(std::move(shadow_lowering.message));
              break;
            }
            if (shadow_lowering.kind == HighBankShadowLoweringKind::Lowered)
              words = std::move(shadow_lowering.words);
            if (words.empty()) {
              if (auto raw_opcode = raw_vop3_opcode(read_u32(text, offset));
                  raw_opcode && is_gfx1250_vop3_compare_opcode(*raw_opcode)) {
                words = copy_instruction_words(text, offset, inst_size);
              }
            }
            if (words.empty())
              words = lower_gfx1250_s_wait_xcnt_to_rdna4(inst, host_arch_);
            if (words.empty())
              words = lower_gfx1250_smem_nv_to_rdna4(inst, host_arch_);
            if (words.empty())
              words = lower_gfx1250_contextual_raw_buffer_descriptor_mov(
                  inst_it, instructions.end(), host_arch_, block, scope.blocks);
            if (words.empty())
              words = lower_gfx1250_contextual_raw_buffer_descriptor_base(
                  inst_it, instructions.end(), host_arch_, block, scope.blocks);
            if (words.empty())
              words = lower_gfx1250_contextual_s_and_b32_address_mask_high(
                  instructions.begin(), inst_it, host_arch_);
            if (words.empty())
              words = lower_raw_gfx1250_v_mul_u64_vop3(text, offset, inst, liveness, source_size);
            if (words.empty())
              words = lower_raw_gfx1250_v_mul_u64_e32(text, offset, inst, liveness, source_size);
          }
          if (words.empty())
            words = translate_instruction_words(inst, offset, liveness, text,
                                                scope.translation->rdna4_grid_x_sgpr);
          if (guest_arch_ == ROCJITSU_CODE_ARCH_GFX1250 && host_arch_ == ROCJITSU_CODE_ARCH_RDNA4) {
            rewrite_gfx1250_zero_sgpr_v_mov_sources(words, inst, instructions.begin(), inst_it);
            rewrite_gfx1250_zero_sgpr_scalar_sources(words, inst, instructions.begin(), inst_it,
                                                     offset, scope.blocks);
            if (words.size() == inst_size / sizeof(uint32_t) &&
                rdna4_memory_dependency_wait_before(inst)) {
              words.insert(words.begin(),
                           build_s_wait_alu(kWaitAluDepctrVaVdstVmVsrc0, host_arch_));
            }
            if (auto barrier = rdna4_scalar_dependency_barrier_after(inst, host_arch_))
              words.push_back(*barrier);
          }

          if (source_size == inst_size) {
            if (auto branch_delta = inst.branch_offset_bytes()) {
              if (words.size() != 1 || inst_size != sizeof(uint32_t)) {
                fail_expanded_copy("expanded text copy cannot relocate non-SOPP direct branch " +
                                   std::string(inst.mnemonic()));
                break;
              }
              const int64_t target =
                  static_cast<int64_t>(offset + inst_size) + static_cast<int64_t>(*branch_delta);
              if (target < 0) {
                fail_expanded_copy("expanded text copy branch target is before .text for " +
                                   std::string(inst.mnemonic()));
                break;
              }
              direct_branches.push_back({scope_words.size(), static_cast<uint64_t>(target),
                                         std::string(inst.mnemonic())});
            }
          }

          if (guest_arch_ == ROCJITSU_CODE_ARCH_GFX1250 &&
              host_arch_ == ROCJITSU_CODE_ARCH_RDNA4) {
            const size_t group_start = scope_words.size();
            scope_word_group_sizes.resize(group_start + words.size(), 0);
            if (!words.empty())
              scope_word_group_sizes[group_start] = static_cast<uint32_t>(words.size());
          }
          scope_words.insert(scope_words.end(), words.begin(), words.end());
          consumed_until = offset + source_size;
          offset += inst_size;
        }
      }

      if (guest_arch_ == ROCJITSU_CODE_ARCH_GFX1250 && host_arch_ == ROCJITSU_CODE_ARCH_RDNA4) {
        const size_t old_word_count = scope_words.size();
        if (scope_word_group_sizes.size() != old_word_count)
          scope_word_group_sizes.assign(old_word_count, 1);
        std::vector<size_t> old_to_new(old_word_count + 1);
        std::vector<uint32_t> rewritten_words;
        rewritten_words.reserve(scope_words.size());

        for (size_t word_index = 0; word_index < old_word_count;) {
          old_to_new[word_index] = rewritten_words.size();
          const size_t group_words =
              scope_word_group_sizes[word_index] == 0 ? 1 : scope_word_group_sizes[word_index];
          if (group_words >= 2 && word_index + group_words <= old_word_count) {
            uint32_t consumed_size = 0;
            const uint32_t available_size = static_cast<uint32_t>(group_words * sizeof(uint32_t));
            auto expansion = lower_raw_gfx1250_v_mul_u64_vop3(scope_words.data() + word_index,
                                                              available_size, &consumed_size);
            const size_t consumed_words = consumed_size / sizeof(uint32_t);
            if (!expansion.empty() && consumed_words != 0 && consumed_words <= group_words) {
              rewritten_words.insert(rewritten_words.end(), expansion.begin(), expansion.end());
              for (size_t consumed = 1; consumed < consumed_words; ++consumed)
                old_to_new[word_index + consumed] = rewritten_words.size();
              for (size_t remaining = consumed_words; remaining < group_words; ++remaining) {
                old_to_new[word_index + remaining] = rewritten_words.size();
                rewritten_words.push_back(scope_words[word_index + remaining]);
              }
              word_index += group_words;
              continue;
            }
          }

          for (size_t group_offset = 0; group_offset < group_words &&
                                        word_index + group_offset < old_word_count;
               ++group_offset) {
            old_to_new[word_index + group_offset] = rewritten_words.size();
            rewritten_words.push_back(scope_words[word_index + group_offset]);
          }
          word_index += group_words;
        }
        old_to_new[old_word_count] = rewritten_words.size();

        if (rewritten_words.size() != scope_words.size()) {
          for (auto &[_, translated_offset] : scope_offsets) {
            const size_t old_word_index = translated_offset / sizeof(uint32_t);
            translated_offset = old_to_new[old_word_index] * sizeof(uint32_t);
          }
          for (RelocatedBranch &branch : direct_branches)
            branch.word_index = old_to_new[branch.word_index];
          scope_words = std::move(rewritten_words);
        }
      }

      for (const RelocatedBranch &branch : direct_branches) {
        if (!expanded_copy_ok)
          break;
        const auto target_it = scope_offsets.find(branch.target_offset);
        if (target_it == scope_offsets.end()) {
          fail_expanded_copy("expanded text copy branch target is outside the copied CFG for " +
                             branch.mnemonic);
          break;
        }

        int16_t branch_dwords = 0;
        const uint64_t branch_pc = branch.word_index * sizeof(uint32_t);
        if (!compute_sopp_branch_offset(branch_pc, target_it->second, branch_dwords)) {
          fail_expanded_copy("expanded text copy branch range exceeds s_branch simm16 for " +
                             branch.mnemonic);
          break;
        }
        scope_words[branch.word_index] =
            (scope_words[branch.word_index] & 0xFFFF0000u) | static_cast<uint16_t>(branch_dwords);
      }

      const auto entry_it = scope_offsets.find(scope.translation->entry_text_offset);
      if (entry_it == scope_offsets.end()) {
        fail_expanded_copy("expanded text copy is missing a translated kernel entry");
        break;
      }

      const uint64_t entry_residue = scope.translation->entry_text_offset % 256;
      if (scope.translation->prologue_words.empty()) {
        const uint64_t entry_without_padding =
            text.size() + expanded_words.size() * sizeof(uint32_t) + entry_it->second;
        const uint64_t padding_bytes =
            (entry_residue + 256 - (entry_without_padding % 256)) % 256;
        assert(padding_bytes % sizeof(uint32_t) == 0 && "entry padding must be word aligned");
        expanded_words.insert(expanded_words.end(), padding_bytes / sizeof(uint32_t),
                              build_s_nop(0, host_arch_));

        const uint64_t scope_base = expanded_words.size() * sizeof(uint32_t);
        expanded_words.insert(expanded_words.end(), scope_words.begin(), scope_words.end());
        copied_entry_offsets.emplace(scope.translation->entry_text_offset,
                                     scope_base + entry_it->second);
      } else {
        const uint64_t stub_without_padding = text.size() + expanded_words.size() * sizeof(uint32_t);
        const uint64_t padding_bytes =
            (entry_residue + 256 - (stub_without_padding % 256)) % 256;
        assert(padding_bytes % sizeof(uint32_t) == 0 && "entry padding must be word aligned");
        expanded_words.insert(expanded_words.end(), padding_bytes / sizeof(uint32_t),
                              build_s_nop(0, host_arch_));

        const uint64_t stub_start = expanded_words.size() * sizeof(uint32_t);
        std::vector<uint32_t> launch_stub(scope.translation->prologue_words.begin(),
                                          scope.translation->prologue_words.end());
        const uint64_t branch_pc = stub_start + launch_stub.size() * sizeof(uint32_t);
        const uint64_t scope_base = branch_pc + sizeof(uint32_t);
        const uint64_t body_entry = scope_base + entry_it->second;
        int16_t entry_branch = 0;
        if (!compute_sopp_branch_offset(branch_pc, body_entry, entry_branch)) {
          fail_expanded_copy("expanded text copy prologue branch range exceeds s_branch simm16");
          break;
        }
        launch_stub.push_back(build_s_branch(entry_branch, host_arch_));

        expanded_words.insert(expanded_words.end(), launch_stub.begin(), launch_stub.end());
        assert(expanded_words.size() * sizeof(uint32_t) == scope_base &&
               "expanded launch stub size mismatch");
        expanded_words.insert(expanded_words.end(), scope_words.begin(), scope_words.end());
        copied_entry_offsets.emplace(scope.translation->entry_text_offset, stub_start);
      }
    }

    if (!expanded_copy_ok)
      return leave_unchanged();

    patcher.append_cave_body(expanded_words);

    std::unordered_set<uint64_t> applied_descriptors;
    for (const KdTranslation &translation : descriptor_translations) {
      if (!applied_descriptors.insert(translation.descriptor_file_offset).second)
        continue;
      const auto copied_entry_it = copied_entry_offsets.find(translation.entry_text_offset);
      if (copied_entry_it == copied_entry_offsets.end()) {
        result.warnings.push_back("expanded text copy could not map a kernel descriptor entry; "
                                  "leaving code object unchanged");
        return leave_unchanged();
      }
      KdTranslation descriptor_patch = translation;
      descriptor_patch.prologue_words.clear();
      if (!patcher.apply_kernel_descriptor_translation(descriptor_patch, host_arch_) ||
          !patcher.redirect_kernel_entry(translation.descriptor_file_offset,
                                         translation.entry_text_offset,
                                         text.size() + copied_entry_it->second)) {
        result.warnings.push_back("kernel descriptor translation could not be applied safely; "
                                  "leaving code object unchanged");
        return leave_unchanged();
      }
    }

    if (!patcher.append_cave_section()) {
      result.warnings.push_back(
          "expanded text copy could not be materialized safely; leaving code object unchanged");
      return leave_unchanged();
    }

    if (const uint32_t metadata_vgprs =
            metadata_vgpr_count_for_in_place_patch(descriptor_translations);
        metadata_vgprs != 0 && !patcher.patch_metadata_vgpr_count(metadata_vgprs)) {
      result.warnings.push_back("AMDGPU metadata VGPR count could not be patched safely; leaving "
                                "code object unchanged");
      return leave_unchanged();
    }
    if (needs_metadata_private_segment_patch(descriptor_translations) &&
        !patcher.patch_metadata_private_segment_fixed_sizes(descriptor_translations)) {
      result.warnings.push_back("AMDGPU metadata private segment size could not be patched safely; "
                                "leaving code object unchanged");
      return leave_unchanged();
    }

    if (target_mach_)
      patcher.update_elf_flags(target_mach_);

    result.elf_bytes = patcher.emit();
    warnings_ = nullptr;
    diagnostics_ = nullptr;
    return result;
  }

  std::unordered_set<const BasicBlock *> translated_blocks;
  for (const KernelTranslationScope &scope : scopes) {
    if (scope.blocks.empty())
      continue;

    TranslationContext kernel_context(
        scope.translation->target_vgpr_count, scope.translation->target_agpr_count,
        scope.translation->target_accvgpr_base, scope.translation->target_sgpr_count);
    LivenessAnalysisOptions liveness_options;
    if (options_.debug_min_free_vgpr)
      liveness_options.min_free_vgpr = *options_.debug_min_free_vgpr;
    LivenessAnalysis liveness(KernelBlockScope(scope.blocks), liveness_options);
    configure_liveness_scratch(guest_arch_, host_arch_, scope, liveness);
    std::optional<HighBankBlockModeMap> high_bank_entry_modes;
    std::optional<HighBankShadowPlan> high_bank_shadow_plan;
    if (guest_arch_ == ROCJITSU_CODE_ARCH_GFX1250 && host_arch_ == ROCJITSU_CODE_ARCH_RDNA4) {
      high_bank_entry_modes = gfx1250_high_bank_entry_modes_for_scope(scope);
      if (high_bank_entry_modes)
        high_bank_shadow_plan =
            gfx1250_high_bank_shadow_plan_for_scope(scope, liveness, *high_bank_entry_modes);
    }

    for (BasicBlock *block : scope.blocks) {
      if (block == nullptr)
        continue;
      if (!translated_blocks.insert(block).second) {
        append_error(result.diagnostics, DiagnosticKind::Legalization,
                     "basic block is reachable from multiple kernel entries; shared kernel text "
                     "translation is not implemented");
        if (continue_after_failure)
          continue;
        return leave_unchanged();
      }

      HighBankShadowState high_bank_shadow_state;
      if (high_bank_entry_modes) {
        const auto mode_it = high_bank_entry_modes->find(block);
        if (mode_it == high_bank_entry_modes->end()) {
          result.warnings.push_back(
              "in-place translation cannot determine gfx1250 VGPR MSB mode for block");
          return leave_unchanged();
        }
        high_bank_shadow_state.mode = mode_it->second;
      }

      InstructionList &instructions = block->instructions();
      uint64_t offset = block->start_offset();
      for (auto it = instructions.begin(); it != instructions.end(); ++it) {
        const auto &inst = *it;
        const uint32_t inst_size = inst.size();
        if (overlaps_any_range(offset, offset + inst_size, local_caves)) {
          offset += inst_size;
          continue;
        }

        const uint32_t *raw = inst.raw_encoding();
        if (!raw) {
          std::memcpy(translated_text.data() + offset, text.data() + offset, inst_size);
          if (trace_callback_) {
            trace_callback_({.source_offset = offset,
                             .source_size = inst_size,
                             .source_words = {},
                             .legalization = nullptr,
                             .copied_original = true,
                             .semantic_lowering = false,
                             .changed = false,
                             .emitted_in_cave = false,
                             .target_offset = offset,
                             .target_words = {}});
          }
          offset += inst_size;
          continue;
        }

        if (high_bank_shadow_plan) {
          auto shadow_lowering = lower_gfx1250_high_bank_shadow_instruction(
              text, inst, offset, liveness, high_bank_shadow_plan, high_bank_shadow_state);
          if (shadow_lowering.kind == HighBankShadowLoweringKind::Unsupported) {
            result.warnings.push_back(std::move(shadow_lowering.message));
            return leave_unchanged();
          }
          if (shadow_lowering.kind == HighBankShadowLoweringKind::Lowered) {
            SemanticReplacement repl{offset, offset + inst_size, std::string(inst.mnemonic()),
                                     std::move(shadow_lowering.words)};
            if (!apply_semantic(repl, translated_text, patcher, local_caves, protected_ranges,
                                allow_unreachable_text_caves))
              return leave_unchanged();
            offset += inst_size;
            continue;
          }
        }

        if (guest_arch_ == ROCJITSU_CODE_ARCH_GFX1250 &&
            host_arch_ == ROCJITSU_CODE_ARCH_RDNA4) {
          auto contextual = lower_gfx1250_s_wait_xcnt_to_rdna4(inst, host_arch_);
          if (!contextual.empty()) {
            SemanticReplacement repl{offset, offset + inst_size, std::string(inst.mnemonic()),
                                     std::move(contextual)};
            if (!apply_semantic(repl, translated_text, patcher, local_caves, protected_ranges,
                                allow_unreachable_text_caves))
              return leave_unchanged();
            offset += inst_size;
            continue;
          }
          contextual = lower_gfx1250_smem_nv_to_rdna4(inst, host_arch_);
          if (!contextual.empty()) {
            SemanticReplacement repl{offset, offset + inst_size, std::string(inst.mnemonic()),
                                     std::move(contextual)};
            if (!apply_semantic(repl, translated_text, patcher, local_caves, protected_ranges,
                                allow_unreachable_text_caves))
              return leave_unchanged();
            offset += inst_size;
            continue;
          }
          contextual =
              lower_gfx1250_contextual_raw_buffer_descriptor_mov(it, instructions.end(), host_arch_,
                                                                 block, scope.blocks);
          if (!contextual.empty()) {
            SemanticReplacement repl{offset, offset + inst_size, std::string(inst.mnemonic()),
                                     std::move(contextual)};
            if (!apply_semantic(repl, translated_text, patcher, local_caves, protected_ranges,
                                allow_unreachable_text_caves))
              return leave_unchanged();
            offset += inst_size;
            continue;
          }
          contextual = lower_gfx1250_contextual_raw_buffer_descriptor_base(
              it, instructions.end(), host_arch_, block, scope.blocks);
          if (!contextual.empty()) {
            SemanticReplacement repl{offset, offset + inst_size, std::string(inst.mnemonic()),
                                     std::move(contextual)};
            if (!apply_semantic(repl, translated_text, patcher, local_caves, protected_ranges,
                                allow_unreachable_text_caves))
              return leave_unchanged();
            offset += inst_size;
            continue;
          }
          contextual = lower_gfx1250_contextual_s_and_b32_address_mask_high(
              instructions.begin(), it, host_arch_);
          if (!contextual.empty()) {
            SemanticReplacement repl{offset, offset + inst_size, std::string(inst.mnemonic()),
                                     std::move(contextual)};
            if (!apply_semantic(repl, translated_text, patcher, local_caves, protected_ranges,
                                allow_unreachable_text_caves))
              return leave_unchanged();
            offset += inst_size;
            continue;
          }
        }

        const InstructionLegalization *leg = nullptr;
        if (legalization_lookup_)
          leg = legalization_lookup_(inst.encoding_id(), inst.opcode());

        const uint16_t dst_opcode = leg ? leg->target_opcode : inst.opcode();

        // Try semantic lowering before raw encoding translation. A matched
        // semantic rule that cannot safely emit code is a translation error:
        // falling through would silently preserve guest semantics on the wrong
        // host ISA.
        {
          auto expansion =
              semantic_translator_->try_lower_expand(inst, offset, liveness, kernel_context);
          if (expansion.status == ExpandStatus::Failed) {
            append_error(result.diagnostics, DiagnosticKind::ExpandFailed,
                         expansion.message.empty()
                             ? "semantic EXPAND rule matched, but could not safely lower"
                             : expansion.message,
                         offset, std::string(inst.mnemonic()), std::move(expansion.required_work));
            if (continue_after_instruction_error(inst, offset)) {
              offset += inst_size;
              continue;
            }
            return leave_unchanged();
          }

          if (expansion.status == ExpandStatus::Success) {
            if (guest_arch_ == ROCJITSU_CODE_ARCH_GFX1250 &&
                host_arch_ == ROCJITSU_CODE_ARCH_RDNA4) {
              rewrite_gfx1250_zero_sgpr_v_mov_sources(expansion.words, inst, instructions.begin(),
                                                       it);
              rewrite_gfx1250_zero_sgpr_scalar_sources(expansion.words, inst, instructions.begin(),
                                                       it, offset, scope.blocks);
            }
            append_hardware_pending_warning(&result.warnings, inst.mnemonic());
            const bool emitted_in_cave = expansion.words.size() * sizeof(uint32_t) > inst_size;
            const uint64_t target_offset =
                emitted_in_cave ? patcher.cave_start() + patcher.cave_body_size() : offset;
            SemanticReplacement repl{offset, offset + inst_size, std::string(inst.mnemonic()),
                                     std::move(expansion.words)};
            if (!apply_semantic(repl, translated_text, patcher, local_caves, protected_ranges,
                                allow_unreachable_text_caves)) {
              if (continue_after_instruction_error(inst, offset)) {
                offset += inst_size;
                continue;
              }
              return leave_unchanged();
            }
            if (trace_callback_) {
              const auto source_words = raw_words_for_inst(inst);
              trace_callback_({.source_offset = offset,
                               .source_size = inst_size,
                               .source_words = source_words,
                               .legalization = leg,
                               .copied_original = false,
                               .semantic_lowering = true,
                               .changed = true,
                               .emitted_in_cave = emitted_in_cave,
                               .target_offset = target_offset,
                               .target_words = repl.target_words});
            }
            offset += inst_size;
            continue;
          }
        }

        if (leg && leg->action == Action::Expand) {
          if (guest_arch_ == ROCJITSU_CODE_ARCH_GFX1250 &&
              host_arch_ == ROCJITSU_CODE_ARCH_RDNA4) {
            result.warnings.push_back("EXPAND not yet implemented for " +
                                      std::string(inst.mnemonic()));
            fill_nops(translated_text, offset, inst_size, host_arch_);
            offset += inst_size;
            continue;
          }

          append_error(result.diagnostics, DiagnosticKind::ExpandMissing,
                       "legalization requires EXPAND, but no expansion rule is implemented", offset,
                       std::string(inst.mnemonic()),
                       {"Add a semantic expansion rule for this mnemonic."});
          if (continue_after_instruction_error(inst, offset)) {
            offset += inst_size;
            continue;
          }
          return leave_unchanged();
        }

        if (requires_semantic_expansion(guest_arch_, inst)) {
          result.warnings.push_back("EXPAND not yet implemented for " +
                                    std::string(inst.mnemonic()));
          fill_nops(translated_text, offset, inst_size, host_arch_);
          offset += inst_size;
          continue;
        }

        if (!handle_encoding(inst, offset, translated_text, dst_opcode, patcher, text, local_caves,
                             protected_ranges, allow_unreachable_text_caves,
                             scope.translation ? scope.translation->rdna4_grid_x_sgpr : -1,
                             instructions.begin(), it, scope.blocks)) {
          if (continue_after_instruction_error(inst, offset)) {
            offset += inst_size;
            continue;
          }
          return leave_unchanged();
        }
        offset += inst_size;
      }
    }

    if (continue_after_failure && has_error_diagnostic(result.diagnostics))
      continue;

    if (kernel_context.required_vgpr_count > kernel_context.num_vgprs)
      scope.translation->target_vgpr_count = kernel_context.required_vgpr_count;
    if (kernel_context.required_sgpr_count > kernel_context.num_sgprs)
      scope.translation->target_sgpr_count = kernel_context.required_sgpr_count;

    if (scope.translation->target_vgpr_count != kernel_context.num_vgprs ||
        scope.translation->target_sgpr_count != kernel_context.num_sgprs) {
      // Semantic rules may allocate descriptor-backed scratch registers beyond
      // the kernel's original SGPR/VGPR counts. Recompute the descriptor with
      // those larger minimums before patching it into the output image.
      KernelDescriptorTranslationOptions descriptor_options;
      descriptor_options.minimum_vgprs = scope.translation->target_vgpr_count;
      descriptor_options.minimum_sgprs = scope.translation->target_sgpr_count;

      // Descriptor growth is intentionally done after instruction lowering so
      // each kernel is translated once. Only descriptors that enter this code
      // scope need the larger floor; rescanning the whole image would also
      // recompute unrelated kernels and risks mixing diagnostics across scopes.
      bool recomputed_descriptor = false;
      for (KdTranslation &translation : descriptor_translations) {
        if (translation.entry_text_offset != scope.translation->entry_text_offset)
          continue;

        auto updated = descriptor_translator.translate_descriptor(
            patcher.image_bytes(), translation.descriptor_file_offset,
            translation.entry_text_offset, descriptor_options);
        if (!updated) {
          append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                       "kernel descriptor translation could not be recomputed; leaving code object "
                       "unchanged");
          return leave_unchanged();
        }

        append_diagnostics(result.diagnostics, updated->diagnostics);
        if (!updated->supported) {
          append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                       "kernel descriptor translation requires unsupported resource or ABI "
                       "virtualization; leaving code object unchanged");
          return leave_unchanged();
        }

        translation = std::move(*updated);
        recomputed_descriptor = true;
      }

      if (!recomputed_descriptor) {
        append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                     "kernel descriptor translation could not be recomputed; leaving code object "
                     "unchanged");
        return leave_unchanged();
      }
    }
  }

  if (continue_after_failure && has_error_diagnostic(result.diagnostics))
    return leave_unchanged();

  std::unordered_set<uint64_t> applied_descriptors;
  for (const KdTranslation &translation : descriptor_translations) {
    if (applied_descriptors.insert(translation.descriptor_file_offset).second) {
      if (!patcher.apply_kernel_descriptor_translation(translation, host_arch_)) {
        append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                     "kernel descriptor translation could not be applied safely; leaving code "
                     "object unchanged");
        return leave_unchanged();
      }
    }
  }

  patcher.overwrite_text(translated_text);
  if (!patcher.append_cave_section()) {
    append_error(result.diagnostics, DiagnosticKind::ResourceLimit,
                 "code cave section could not be materialized safely; leaving code object "
                 "unchanged");
    return leave_unchanged();
  }

  if (const uint32_t metadata_vgprs =
          metadata_vgpr_count_for_in_place_patch(descriptor_translations);
      metadata_vgprs != 0 && !patcher.patch_metadata_vgpr_count(metadata_vgprs)) {
    result.warnings.push_back(
        "AMDGPU metadata VGPR count could not be patched safely; leaving code object unchanged");
    return leave_unchanged();
  }
  if (needs_metadata_private_segment_patch(descriptor_translations) &&
      !patcher.patch_metadata_private_segment_fixed_sizes(descriptor_translations)) {
    result.warnings.push_back("AMDGPU metadata private segment size could not be patched safely; "
                              "leaving code object unchanged");
    return leave_unchanged();
  }

  if (target_mach_)
    patcher.update_elf_flags(target_mach_);

  warnings_ = nullptr;
  diagnostics_ = nullptr;
  result.elf_bytes = patcher.emit();
  return result;
}

std::vector<uint32_t>
BinaryTranslator::translate_instruction_words(const Instruction &inst, uint64_t offset,
                                              const LivenessAnalysis &liveness,
                                              std::span<const uint8_t> orig_text,
                                              int16_t rdna4_grid_x_sgpr) {
  const uint32_t inst_size = inst.size();
  const uint32_t *raw = inst.raw_encoding();
  auto finish_words = [&](std::vector<uint32_t> words) {
    if (guest_arch_ == ROCJITSU_CODE_ARCH_GFX1250 && host_arch_ == ROCJITSU_CODE_ARCH_RDNA4)
      remap_gfx1250_ttmp9_reads_to_sgpr(inst, rdna4_grid_x_sgpr, words);
    return words;
  };
  if (!raw)
    return finish_words(copy_instruction_words(orig_text, offset, inst_size));

  const uint32_t w0 = read_u32(orig_text, offset);
  const uint32_t w1 = inst_size > 4 ? read_u32(orig_text, offset + sizeof(uint32_t)) : 0;
  const uint32_t w2 = inst_size > 8 ? read_u32(orig_text, offset + 2 * sizeof(uint32_t)) : 0;

  if (guest_arch_ == ROCJITSU_CODE_ARCH_GFX1250 && host_arch_ == ROCJITSU_CODE_ARCH_RDNA4 &&
      encoding_translate_) {
    if (auto raw_opcode = raw_vop3_opcode(w0);
        raw_opcode && is_gfx1250_vop3_compare_opcode(*raw_opcode)) {
      std::vector<uint32_t> words{w0, w1};
      if (inst_size > 2 * sizeof(uint32_t))
        words.push_back(w2);
      return finish_words(std::move(words));
    }
  }

  if (guest_arch_ == ROCJITSU_CODE_ARCH_GFX1250 && host_arch_ == ROCJITSU_CODE_ARCH_RDNA4) {
    uint32_t source_size = inst_size;
    auto raw_expansion =
        lower_raw_gfx1250_v_mul_u64_vop3(orig_text, offset, inst, liveness, source_size);
    if (raw_expansion.empty())
      raw_expansion =
          lower_raw_gfx1250_v_mul_u64_e32(orig_text, offset, inst, liveness, source_size);
    if (!raw_expansion.empty())
      return finish_words(std::move(raw_expansion));
  }

  uint16_t source_opcode = inst.opcode();
  if (guest_arch_ == ROCJITSU_CODE_ARCH_GFX1250 && host_arch_ == ROCJITSU_CODE_ARCH_RDNA4) {
    if (auto raw_opcode = raw_vop3_opcode(w0))
      source_opcode = *raw_opcode;
  }

  const InstructionLegalization *leg = nullptr;
  if (legalization_lookup_)
    leg = legalization_lookup_(inst.encoding_id(), source_opcode);

  const uint16_t dst_opcode = leg ? leg->target_opcode : source_opcode;

  TranslationContext context;
  auto expansion = semantic_translator_->try_lower_expand_with_opcode(inst, offset, liveness,
                                                                      context, source_opcode);
  if (expansion.status == ExpandStatus::Failed) {
    if (warnings_)
      warnings_->push_back(expansion.message.empty()
                               ? "semantic EXPAND rule matched, but could not safely lower"
                               : expansion.message);
    return finish_words(nop_words(inst_size, host_arch_));
  }
  if (expansion.status == ExpandStatus::Success) {
    append_hardware_pending_warning(warnings_, inst.mnemonic());
    return finish_words(std::move(expansion.words));
  }

  if (leg && leg->action == Action::Expand) {
    if (warnings_)
      warnings_->push_back("EXPAND not yet implemented for " + std::string(inst.mnemonic()));
    return finish_words(nop_words(inst_size, host_arch_));
  }

  if (requires_semantic_expansion(guest_arch_, inst)) {
    if (warnings_)
      warnings_->push_back("EXPAND not yet implemented for " + std::string(inst.mnemonic()));
    return finish_words(nop_words(inst_size, host_arch_));
  }

  if (!encoding_translate_)
    return finish_words(copy_instruction_words(orig_text, offset, inst_size));

  auto tr = encoding_translate_(inst.encoding_id(), w0, w1, w2, dst_opcode);
  if (tr.word_count == 0)
    return finish_words(copy_instruction_words(orig_text, offset, inst_size));

  const uint32_t translated_bytes = tr.word_count * sizeof(uint32_t);
  if (inst_size >= translated_bytes && inst_size - translated_bytes == sizeof(uint32_t) &&
      tr.word_count < 3) {
    uint32_t lit_word = 0;
    std::memcpy(&lit_word, orig_text.data() + offset + translated_bytes, sizeof(lit_word));
    tr.words[tr.word_count++] = lit_word;
  }

  return finish_words({tr.words, tr.words + tr.word_count});
}

bool BinaryTranslator::apply_semantic(
    const SemanticReplacement &repl, std::vector<uint8_t> &text, CodeObjectPatcher &patcher,
    std::vector<std::pair<uint64_t, uint64_t>> &local_caves,
    std::span<const std::pair<uint64_t, uint64_t>> protected_ranges,
    bool allow_unreachable_text_caves) {
  assert(repl.matched() && "apply_semantic called with unmatched replacement");
  assert(repl.start_offset < repl.end_offset && "invalid replacement range");
  assert(repl.end_offset <= text.size() && "replacement exceeds text bounds");

  const uint32_t source_size = repl.end_offset - repl.start_offset;
  const uint32_t target_size = repl.target_words.size() * 4;

  if (target_size <= source_size) {
    std::memcpy(text.data() + repl.start_offset, repl.target_words.data(), target_size);
    if (target_size < source_size)
      fill_nops(text, repl.start_offset + target_size, source_size - target_size, host_arch_);
    return true;
  }

  const uint64_t stub_next = repl.start_offset + source_size;
  const uint64_t branch_pc = repl.start_offset;

  auto patch_source_branch = [&](int16_t fwd_dwords) {
    const uint32_t stub = build_s_branch(fwd_dwords, host_arch_);
    std::memcpy(text.data() + repl.start_offset, &stub, sizeof(stub));
    const uint32_t nop = build_s_nop(0, host_arch_);
    for (uint64_t off = repl.start_offset + sizeof(uint32_t); off < repl.end_offset;
         off += sizeof(uint32_t))
      std::memcpy(text.data() + off, &nop, sizeof(nop));
  };

  const uint64_t local_cave_size = target_size + sizeof(uint32_t);
  int16_t local_fwd_dwords = 0;
  int16_t local_ret_dwords = 0;

  auto apply_local_cave = [&](uint64_t local_cave_offset) {
    auto cave_words = repl.target_words;
    cave_words.push_back(build_s_branch(local_ret_dwords, host_arch_));

    patch_source_branch(local_fwd_dwords);
    std::memcpy(text.data() + local_cave_offset, cave_words.data(),
                cave_words.size() * sizeof(uint32_t));
    local_caves.emplace_back(local_cave_offset,
                             local_cave_offset + cave_words.size() * sizeof(uint32_t));
  };

  if (auto local_cave_offset =
          find_local_text_cave(text, repl, local_cave_size, local_caves, protected_ranges, false,
                               local_fwd_dwords, local_ret_dwords)) {
    apply_local_cave(*local_cave_offset);
    return true;
  }
  if (allow_unreachable_text_caves) {
    local_fwd_dwords = 0;
    local_ret_dwords = 0;
    if (auto local_cave_offset =
            find_local_text_cave(text, repl, local_cave_size, local_caves, protected_ranges, true,
                                 local_fwd_dwords, local_ret_dwords)) {
      apply_local_cave(*local_cave_offset);
      return true;
    }
  }

  const uint64_t cave_byte_offset = patcher.cave_start() + patcher.cave_body_size();

  // s_branch simm16 targets (PC + 4 + simm16*4).
  int16_t fwd_dwords = 0;
  if (!compute_sopp_branch_offset(branch_pc, cave_byte_offset, fwd_dwords)) {
    const bool can_defer = has_unimplemented_expand_gap(warnings_);
    if (warnings_) {
      warnings_->push_back(
          can_defer ? "code cave branch range exceeds s_branch simm16; leaving source instruction "
                      "unchanged"
                    : "code cave branch range exceeds s_branch simm16; leaving code object "
                      "unchanged");
      if (cave_diagnostics_enabled())
        warnings_->push_back(cave_range_diagnostic("code cave branch range diagnostic", repl,
                                                   cave_byte_offset, target_size));
    }
    if (can_defer)
      return true;
    if (diagnostics_)
      append_error(*diagnostics_, DiagnosticKind::ResourceLimit,
                   "code cave branch range exceeds s_branch simm16; leaving code object unchanged",
                   repl.start_offset);
    return false;
  }

  int16_t ret_dwords = 0;
  const uint64_t return_branch_pc = cave_byte_offset + repl.target_words.size() * sizeof(uint32_t);
  if (!compute_sopp_branch_offset(return_branch_pc, stub_next, ret_dwords)) {
    const bool can_defer = has_unimplemented_expand_gap(warnings_);
    if (warnings_) {
      warnings_->push_back(
          can_defer ? "code cave return branch range exceeds s_branch simm16; leaving source "
                      "instruction unchanged"
                    : "code cave return branch range exceeds s_branch simm16; leaving code object "
                      "unchanged");
      if (cave_diagnostics_enabled())
        warnings_->push_back(cave_range_diagnostic("code cave return range diagnostic", repl,
                                                   cave_byte_offset, target_size));
    }
    if (can_defer)
      return true;
    if (diagnostics_)
      append_error(*diagnostics_, DiagnosticKind::ResourceLimit,
                   "code cave return branch range exceeds s_branch simm16; leaving code object "
                   "unchanged",
                   repl.start_offset);
    return false;
  }

  patch_source_branch(fwd_dwords);
  auto cave_words = repl.target_words;
  cave_words.push_back(build_s_branch(ret_dwords, host_arch_));

  patcher.append_cave_body(cave_words);
  return true;
}

bool BinaryTranslator::handle_encoding(
    const Instruction &inst, uint64_t offset, std::vector<uint8_t> &text, uint16_t dst_opcode,
    CodeObjectPatcher &patcher, std::span<const uint8_t> orig_text,
    std::vector<std::pair<uint64_t, uint64_t>> &local_caves,
    std::span<const std::pair<uint64_t, uint64_t>> protected_ranges,
    bool allow_unreachable_text_caves, int16_t rdna4_grid_x_sgpr,
    InstructionList::Iterator block_begin, InstructionList::Iterator inst_it,
    std::span<BasicBlock *const> scope_blocks) {
  const uint32_t *raw = inst.raw_encoding();
  assert(raw && "handle_encoding called without raw encoding");
  const bool tracing = static_cast<bool>(trace_callback_);
  const auto source_words = tracing ? raw_words_for_inst(inst) : std::vector<uint32_t>{};

  auto emit_trace = [&](bool copied_original, bool changed, bool emitted_in_cave,
                        uint64_t target_offset, std::span<const uint32_t> target_words) {
    if (!trace_callback_)
      return;
    trace_callback_({.source_offset = offset,
                     .source_size = static_cast<uint32_t>(inst.size()),
                     .source_words = source_words,
                     .legalization = nullptr,
                     .copied_original = copied_original,
                     .semantic_lowering = false,
                     .changed = changed,
                     .emitted_in_cave = emitted_in_cave,
                     .target_offset = target_offset,
                     .target_words = target_words});
  };

  const uint32_t w0 = raw[0];
  const uint32_t w1 = inst.size() > 4 ? raw[1] : 0;
  const uint32_t w2 = inst.size() > 8 ? raw[2] : 0;

  std::vector<uint32_t> replacement_words;
  if (!encoding_translate_) {
    replacement_words = copy_instruction_words(orig_text, offset, inst.size());
  } else {
    auto tr = encoding_translate_(inst.encoding_id(), w0, w1, w2, dst_opcode);

    if (tr.word_count == 0) {
      replacement_words = copy_instruction_words(orig_text, offset, inst.size());
    } else {
      // Append trailing literal constant when the source instruction is larger
      // than the translated encoding. This handles single-word formats (SOP1,
      // SOP2, VOP1, VOP2, etc.) with a 32-bit literal appended when a source
      // operand is 0xFF. The encoding translator returns the format's native
      // word count; the literal is always one extra word beyond that.
      // Guard: only append if the gap is exactly one word (the literal). Larger
      // gaps would indicate a format mismatch, not a trailing literal.
      const uint32_t translated_bytes = tr.word_count * 4u;
      const uint32_t orig_bytes = inst.size();
      if (orig_bytes >= translated_bytes && orig_bytes - translated_bytes == 4 &&
          tr.word_count < 3) {
        uint32_t lit_word;
        std::memcpy(&lit_word, orig_text.data() + offset + translated_bytes, 4);
        tr.words[tr.word_count++] = lit_word;
      }

      replacement_words.assign(tr.words, tr.words + tr.word_count);
    }
  }

  const uint32_t orig_bytes = inst.size();
  if (guest_arch_ == ROCJITSU_CODE_ARCH_GFX1250 && host_arch_ == ROCJITSU_CODE_ARCH_RDNA4 &&
      !replacement_words.empty()) {
    remap_gfx1250_ttmp9_reads_to_sgpr(inst, rdna4_grid_x_sgpr, replacement_words);
    rewrite_gfx1250_zero_sgpr_v_mov_sources(replacement_words, inst, block_begin, inst_it);
    rewrite_gfx1250_zero_sgpr_scalar_sources(replacement_words, inst, block_begin, inst_it,
                                             offset, scope_blocks);
    if (rdna4_memory_dependency_wait_before(inst)) {
      replacement_words.insert(replacement_words.begin(),
                               build_s_wait_alu(kWaitAluDepctrVaVdstVmVsrc0, host_arch_));
    }
    if (auto barrier = rdna4_scalar_dependency_barrier_after(inst, host_arch_))
      replacement_words.push_back(*barrier);
  }

  const uint32_t target_size = replacement_words.size() * sizeof(uint32_t);
  const bool emitted_in_cave = target_size > orig_bytes;
  const uint64_t target_offset =
      emitted_in_cave ? patcher.cave_start() + patcher.cave_body_size() : offset;
  const bool copied_original = !encoding_translate_ ||
                               words_changed(source_words, replacement_words) == false;
  const bool changed = tracing && words_changed(source_words, replacement_words);
  const std::vector<uint32_t> target_words_for_trace =
      tracing ? replacement_words : std::vector<uint32_t>{};

  if (target_size <= orig_bytes) {
    std::memcpy(text.data() + offset, replacement_words.data(), target_size);
    if (target_size < orig_bytes)
      fill_nops(text, offset + target_size, orig_bytes - target_size, host_arch_);
  } else {
    SemanticReplacement repl{offset,
                             offset + inst.size(),
                             std::string(inst.mnemonic()),
                             std::move(replacement_words)};
    if (!apply_semantic(repl, text, patcher, local_caves, protected_ranges,
                        allow_unreachable_text_caves))
      return false;
  }
  emit_trace(copied_original, changed, emitted_in_cave, target_offset, target_words_for_trace);
  return true;
}

} // namespace rocjitsu
