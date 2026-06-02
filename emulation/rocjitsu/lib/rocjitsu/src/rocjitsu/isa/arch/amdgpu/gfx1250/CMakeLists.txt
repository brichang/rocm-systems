# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

rj_add_object_library(rocjitsu_isa_gfx1250
    addr_calc.cpp
    decoder.cpp
    encodings.cpp
    operand.cpp
    smem.cpp
    sop1.cpp
    sop2.cpp
    sopc.cpp
    sopk.cpp
    sopp.cpp
    vbuffer.cpp
    vds.cpp
    vflat.cpp
    vglobal.cpp
    vimage.cpp
    vop1.cpp
    vop2.cpp
    vop3_part1.cpp
    vop3_part2.cpp
    vop3_part3.cpp
    vop3_part4.cpp
    vop3p.cpp
    vopd.cpp
    vopc_part1.cpp
    vopc_part2.cpp
    vscratch.cpp
)
