"""
Test basic function of kunlun2 cdnn
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
from xpu.assembler import XPUProgram
from xpu.kunlun2.simulator import Kl2CdnnSimulator
from xpu.ctypes_util import to_array
from xpu.ctypes_util import reinterpret_cast
from xpu.ctypes_util import to_bytes

class InstructionTestCase(unittest.TestCase):
    """ Define Test Cases """
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_dma(self):
        """ test basic dma instruction """
        sim = Kl2CdnnSimulator()
        #perf_sim = Kl2CdnnSimulator(1, 1) #mode 0 : func, 1 : perf
        # mov data from gm 0 ~ 4096 to sram 0 ~ 4096
        gm_base_addr = 0x0
        # r1 unit-id, r2 lock priority, r3 src type and dst type 
        # r4 header, r5 src pos, r6 dst pos, r7 src high addr,
        # r8 dst high addr, r9 dst low addr, r10 src low addr
        # r11 data length
        asm = """
        .MODEL KUNLUN2
        addi r1, r1, 5
        addi r2, r2, 0
        addi r3, r3, 0
        addi r4, r4, 1
        addi r5, r5, 0
        addi r6, r6, 2
        addi r7, r7, {}
        addi r8, r8, 0
        addi r9, r9, 0
        addi r10, r10, {}
        slli r10, r10, 24
        addi r14, r14, {}
        slli r14, r14, 12
        addi r15, r15, {}
        or r10, r10, r14
        or r10, r10, r15
        addi r11, r11, 4096
        lock r2, r1, r0
        dma_cfg r3, r3, r4, 0
        dma_cfg r6, r5, r0, 1
        dma_cfg r8, r7, r0, 2
        dma_cfg r9, r10, r11, 3
        dma_run r0, r0, r0
        unlock r0, r1, r0
        xfence r0, r0, r0
        exit
        """
        asm = asm.format((gm_base_addr)>>32, (gm_base_addr >> 24) & 0xff, (gm_base_addr >> 12) & 0xfff, gm_base_addr & 0xfff)
        sim.set_asm_source(code=asm)
        sim.run()

    def test_ew(self):
        """ test basic ew instruction """
        sim = Kl2CdnnSimulator()
        #perf_sim = Kl2CdnnSimulator(1, 1) #mode 0 : func, 1 : perf
        # add data and ldsd, just test can running successfully 
        # r1 unit-id, r2 lock priority, r3 coeff0 r4 coeff2
        # r5 src_1_addr, r6 src_2 addr r7 res_addr r8 stream size
        # r9 reg_addr #r10 lut_id
        asm = """
        .MODEL KUNLUN2
        addi r1, r0, 3
        addi r2, r0, 0
        addi r3, r0, 1
        addi r4, r0, 2
        addi r5, r0, 64
        addi r6, r0, 128
        addi r7, r0, 256
        addi r8, r0, 10
        addi r9, r0, 1
        addi r10, r0, 2
        lock r2, r1, r0
        ew_cfg r0, r0, r0, 1
        ew_cfg r0, r1, r0, 2
        ew_cfg r8, r0, r0, 3
        ew_cfg r0, r0, r10, 11
        ewcoeff_cfg r3, r4, r4, 0
        dsmadd r7, r5, r6, 0
        ld_sd r20, r1, r9
        unlock r0, r1, r0
        xfence r0, r0, r0
        exit
        """
        sim.set_asm_source(code=asm)
        sim.run()
        print(sim.get_scalar_register(20))
        assert sim.get_scalar_register(20) == 2 # lut id returned by ld_sd instruction

if __name__ == '__main__':
    unittest.main()
