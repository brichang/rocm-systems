# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from types import SimpleNamespace

from amdisa.codegen import CodeGenerator
from amdisa.codegen.execute.vector_special import gen_vector_movrel


def test_simm64_literals_require_operand_type():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(operand_types=['OPR_SIMM32'])
    assert not codegen._supports_simm64_literal_operands()

    codegen.isa_spec = SimpleNamespace(operand_types=['OPR_SIMM32', 'OPR_SIMM64'])
    assert codegen._supports_simm64_literal_operands()


def test_movrel_uses_two_arg_resolver_without_vgpr_msb_indexing():
    src_body = gen_vector_movrel(
        ['vdst'], ['src0'], 'src', uses_vgpr_msb_indexing=False
    )
    dst_body = gen_vector_movrel(
        ['vdst'], ['src0'], 'dst', uses_vgpr_msb_indexing=False
    )

    assert 'Isa::resolved_vgpr_offset(src0.opr_type_, src0.encoding_value_)' in src_body
    assert 'Isa::resolved_vgpr_offset(wf, src0.opr_type_' not in src_body
    assert 'Isa::resolved_vgpr_offset(vdst.opr_type_, vdst.encoding_value_)' in dst_body
    assert 'Isa::resolved_vgpr_offset(wf, vdst.opr_type_' not in dst_body


def test_movrel_uses_wavefront_resolver_with_vgpr_msb_indexing():
    body = gen_vector_movrel(['vdst'], ['src0'], 'src', uses_vgpr_msb_indexing=True)
    assert 'Isa::resolved_vgpr_offset(wf, src0.opr_type_' in body
    assert 'src0.vgpr_msb_role()' in body
