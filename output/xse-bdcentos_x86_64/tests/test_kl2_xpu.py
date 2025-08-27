"""
Test basic function of kunlun2 cluster
"""
import ctypes
import random
import logging
import unittest
import os
import time

import math
import numpy as np

from xpu.base import XpuException
from xpu.kunlun2.simulator import Kl2ClusterSimulator
from xpu.ctypes_util import to_array
from xpu.ctypes_util import reinterpret_cast
from xpu.ctypes_util import to_bytes

class InstructionTestCase(unittest.TestCase):
    """ Define Test Cases """
    def setUp(self):
        os.environ['XPUSIM_GM_BASE_ADDR'] = "0"
        os.environ['XPUSIM_GM_SIZE'] = "0xFFFFFFFF"
        os.environ['XPUSIM_L3_BASE_ADDR'] = "0x100000000"

    def tearDown(self):
        del os.environ['XPUSIM_GM_BASE_ADDR']
        del os.environ['XPUSIM_GM_SIZE']
        del os.environ['XPUSIM_L3_BASE_ADDR']

    def test_addi(self):
        """ test addi instruction """
        # Kl2ClusterSimulator:: first param is cluster num, second param is mode, 0: func, 1 : perf
        sim = Kl2ClusterSimulator(1, 0)
        perf_sim = Kl2ClusterSimulator(1, 1)
        asm = """
        .MODEL KUNLUN2
        addi r5, r2, {}
        exit
        """
        for i in range(50):
            sim.lock_hw()
            sim.set_scalar_register(2, i)
            func_ret = sim.get_scalar_register(2)
            assert func_ret == i
            sim.set_asm_source(code=asm.format(i))
            sim.run()
            func_ret = sim.get_scalar_register(5)
            sim.unlock_hw()
            perf_sim.lock_hw()
            perf_sim.set_scalar_register(2, i)
            perf_sim.set_asm_source(code=asm.format(i))
            perf_sim.run()
            perf_ret = perf_sim.get_scalar_register(5)
            perf_sim.unlock_hw()
            assert func_ret == 2 * i
            assert perf_ret == 2 * i

    def test_addi2(self):
        sim = Kl2ClusterSimulator()
        asm = """
        .MODEL KUNLUN2
        xor r0, r1, r1
        xor r2, r1, r1
        addi r5, r2, {}
        exit
        """
        for i in range(50):
            sim.set_asm_source(code=asm.format(i))
            sim.run()
            assert sim.get_scalar_register(5) == i

    def test_read_write_interface(self):
        func_sim = Kl2ClusterSimulator()
        perf_sim = Kl2ClusterSimulator(1, 1)
        for i in range(255):
            func_sim.write_lm(20, i)
            assert func_sim.read_lm(20) == i
            perf_sim.write_lm(20, i)
            assert perf_sim.read_lm(20) == i

    def test_read_write_float(self):
        a = ctypes.c_float(1.0002)
        func_sim = Kl2ClusterSimulator()
        func_sim.write_lm_block(20, a, ctypes.sizeof(a))
        b = ctypes.c_float()
        func_sim.read_lm_block(20, b, ctypes.sizeof(b))
        assert a.value == b.value

    def test_read_write_uint32(self):
        a = ctypes.c_uint(5000)
        func_sim = Kl2ClusterSimulator()
        func_sim.write_lm_block(20, a, ctypes.sizeof(a))
        b = ctypes.c_uint()
        func_sim.read_lm_block(20, b, ctypes.sizeof(b))
        assert a.value == b.value

    def test_get_set_float_to_register(self):
        a = ctypes.c_float(1.30)
        func_sim = Kl2ClusterSimulator()
        b = reinterpret_cast(a, ctypes.c_uint)
        func_sim.set_scalar_register(15, b)
        c = func_sim.get_scalar_register(15)
        d = reinterpret_cast(ctypes.c_uint(c), ctypes.c_float)
        assert d.value == a.value

# test func_simulator
    def test_scalar_alu(self):
        func_sim = Kl2ClusterSimulator()
        for i in range(31):
            func_sim.set_scalar_register(i,0)
        for i in range(256):
            func_sim.write_lm(i, 0)
        asm = """
        .MODEL KUNLUN2
        sub.s r0, r0, r0
        addi r1, r0, 1
        add.u r2, r1, r1
        add.u r3, r2, r1
        mul.s r4, r2, r2
        sub.s r5, r4, r3
        fmac.u r5, r1, r4
        mul.u r6, r2, r3
        addi r7, r3, 0
        fmac.s r7, r1, r4
        sub.u r9, r9, r9
        xor r10, r10, r10
        add.u r9, r9, r3
        add.u r10, r10, r1
        bneq r10, r3, -8
        div.u r8, r9, r3
        add.u r8, r8, r5
        addi r31, r9, -7
        add.u r31, r31, r9
        addi r30, r31, -1
        min.u r10, r30, r31
        max.u r11, r30, r31
        div.s r2, r6, r3
        mod.s r1, r9, r8
        setneq r1, r8, r10
        add.u r12, r1, r11
        xor r13, r13, r13
        or r13, r12, r13
        sll r13, r13, r1
        sub.u r13, r13, r11
        slli r14, r7, 1
        srai r30, r6, 8
        srli r15, r3, 4
        add.u r15, r15, r3
        addi r30, r30, 1
        blt.u r30, r5, -8
        mfence
        store.w r0, 0(r0)
        store.w r1, 4(r0)
        store.w r2, 8(r0)
        store.b r3, 12(r0)
        store.b r4, 16(r0)
        store.w r5, 20(r0)
        store.b r6, 24(r0)
        store.b r7, 28(r0)
        store.w r8, 32(r0)
        store.w r9, 36(r0)
        store.w r10, 40(r0)
        store.w r11, 44(r0)
        store.b r12, 48(r0)
        store.b r13, 52(r0)
        store.w r14, 56(r0)
        store.w r15, 60(r0)
        load.w r16, 0(r0)
        load.w r17, 4(r0)
        load.w r18, 8(r0)
        load.bu r19, 12(r0)
        load.bu r20, 16(r0)
        load.w r21, 20(r0)
        load.w r22, 24(r0)
        load.b r23, 28(r0)
        load.b r24, 32(r0)
        load.w r25, 36(r0)
        load.w r26, 40(r0)
        load.b r27, 44(r0)
        load.bu r28, 48(r0)
        load.w r29, 52(r0)
        load.w r30, 56(r0)
        load.w r31, 60(r0)
        nop
        exit
        """
        func_sim.set_asm_source(code=asm)
        func_sim.run()
        for i in range(31):
            assert func_sim.get_scalar_register(i) == (i % 16)

    def test_scalar_fp_data(self):
        func_sim = Kl2ClusterSimulator()
        for i in range(31):
            func_sim.set_scalar_register(i,0)
        for i in range(10):
            a = ctypes.c_float(0.123*i+i)
            func_sim.write_lm_block(i*4, a, ctypes.sizeof(a))
        asm = """
        .MODEL KUNLUN2
        nop
        sub.u r0, r0, r0
        addi r31, r0, 64
        load.w r1, 0(r0)
        load.w r2, 4(r0)
        load.w r3, 8(r0)
        load.w r4, 12(r0)
        load.w r5, 16(r0)
        load.w r6, 20(r0)
        load.w r7, 24(r0)
        load.w r8, 28(r0)
        load.w r9, 32(r0)
        load.w r10, 36(r0)
        store.w r1, 0(r31)
        store.w r2, 4(r31)
        store.w r3, 8(r31)
        store.w r4, 12(r31)
        store.w r5, 16(r31)
        store.w r6, 20(r31)
        store.w r7, 24(r31)
        store.w r8, 28(r31)
        store.w r9, 32(r31)
        store.w r10, 36(r31)
        add.f.rn r11, r1, r2
        sub.f.rn r12, r3, r4
        mul.f.rn r13, r2, r7
        min.f r14, r8, r9
        max.f r15, r8, r9
        or r16, r10, r0
        fmac.f r16, r8, r2
        neg.f r17, r16
        abs.f r18, r17
        setlt.f r31, r17, r13
        setle.f r30, r12, r10
        float2fix.rn r29, r10
        fix2float.rn r19, r29
        float2fix_u.rn r28, r5
        fix2float_u.rn r20, r28
        float2bfloat.rn r21, r14
        bfloat2float.rn r22, r21
        fix2bfloat.rn r23, r28
        bfloat2fix.rn r27, r23
        fix2bfloat_u.rn r24, r27
        bfloat2fix_u.rn r26, r24
        blt.f r8, r2, -4
        exit
        """
        func_sim.set_asm_source(code=asm)
        func_sim.run()
        list = [1.123, -1.123, 7.56677, 7.861, 8.984, 18.9349, -18.9349, 18.9349, 10, 4, 0, 7.85938]
        for i in range(64,74):
            b = ctypes.c_float()
            func_sim.read_lm_block((i-64)*4, b, ctypes.sizeof(b))
            a = ctypes.c_float(0.123*(i-64)+(i-64))
            assert a.value == b.value
        # not check bfp16 result, because we can't display them.
        for i in range(11,22):
            c = func_sim.get_scalar_register(i)
            d = reinterpret_cast(ctypes.c_uint(c), ctypes.c_float)
            e = list[i-11]
            f = reinterpret_cast(ctypes.c_float(e), ctypes.c_float)
            g = d.value - f.value
            assert abs(g) < 0.1

    def test_scalar_memory(self):
        func_sim = Kl2ClusterSimulator()
        for i in range(256):
            func_sim.write_lm(i, i)
        asm = """
        .MODEL KUNLUN2
        nop
        sub.u r0, r0, r0
        addi r1, r0, 1
        slli r1, r1, 14
        addi r2, r0, 256
        addi r3, r2, 256
        s_lm2gm r2, r0, 0(r0)
        l_gm2sm r2, r1, 0(r0)
        s_sm2gm r2, r1, 0(r2)
        l_gm2lm r2, r2, 0(r2)
        bl_gm2lm r2, r3, 0(r2)
        get_clock r4
        get_clockh r5
        core_id r6
        cluster_id r7
        exit
        """
        func_sim.set_asm_source(code=asm)
        func_sim.run()
        # lm -> gm -> sm -> gm -> lm
        # Only one core
        for i in range(768):
            assert func_sim.read_lm(i) == (i % 256)

if __name__ == '__main__':
    unittest.main()
