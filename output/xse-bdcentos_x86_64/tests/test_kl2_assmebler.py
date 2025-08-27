"""
Test basic function of kunlun2 assembler
"""
import ctypes
import random
import math
import unittest
import os
 
from xpu.base import XpuException
from xpu.kunlun2.simulator import Kl2ClusterSimulator
from xpu.assembler import XPUProgram

class Kl2AssemblerTestCase(unittest.TestCase):
    """ Define Test Cases """
    def setUp(self):
        pass
    
    def tearDown(self):
        pass

    def fill_instr(self, xpu_pg, code):
        if code is not None:
            if not isinstance(code, (list, tuple)):
                code = code.split('\n')
            xpu_pg.add_lines(code)
    
    def test_sd_instruction(self):
        """ test addi instruction """
        asm = """
        .MODEL KUNLUN2
        lock r5, r2, r0
        unlock r0, r5, r0
        xfence r0, r0, r0
        ld_sd r0, r11, r0
        xsignal r0, r5, r0, 1
        xwait r0, r5, r0, 1
        xflush r0, r8, r0
        ds_cfg r1, r2, r3, 0
        shuffle r1, r4, r5, 1
        shuffle_batch r2, r5, r6, 2
        shuffle_coa r3, r5, r7, 1
        w2v_chw r3, r5, r7, 0
        w2v_hwc r3, r7, r9, 0
        rs_cfg r4, r5, r9, 1
        rs_col r1, r2, r3, 4
        rs_row r5, r9, r3, 5
        mm_cfg r2, r3, r4, 0
        mm r2, r3, r4, 5
        mm_acc r5, r4, r3, 11
        ew_cfg r3, r2, r1, 7
        ewcoeff_cfg r3, r2, r1, 2
        ewtable_cfg r2, r1, r20, 0
        dsmadd r2, r2, r31, 0
        dsmul r29, r27, r30, 1
        dscmpnsel r20, r19, r10, 0
        ssreduce r20, r21, r17, 0
        sspooling r14, r15, r19, 2
        sslut r9, r10, r11, 3
        dsselect r11, r12, r13, 5
        dma_cfg r0, r2, r9, 2
        dma_run r11, r13, r17
        dma_flush r14, r16, r19
        """
        xpu_pg = XPUProgram()
        self.fill_instr(xpu_pg, asm)
        hex = xpu_pg.get_hex()
        expect_hex = [
            '0001029d', 
             '0002901d', 
             '0000003d', 
             '0005805d', 
             '0202807d', 
             '0202907d', 
             '0004207d', 
             '00310081', 
             '025200a1', 
             '04629121', 
             '027281c1', 
             '007281e1', 
             '009391e1', 
             '02928211', 
             '083100b1', 
             '0a3492b1', 
             '00418109', 
             '0a418129', 
             '163212a9', 
             '0e11018d', 
             '0411118d', 
             '0140a10d', 
             '01f1012d', 
             '03ed9ead', 
             '00a9aa2d', 
             '011a8a4d', 
             '0537974d', 
             '06b524cd', 
             '0ad605ed', 
             '04910015', 
             '011685b5', 
             '01380755'] 
        assert hex == expect_hex

    def test_vprintf(self):
        """ test vprintf instr """
        asm = """
        .MODEL KUNLUN2
        vprintf r0, r1, r2
        """
        xpu_pg = XPUProgram()
        self.fill_instr(xpu_pg, asm)
        hex = xpu_pg.get_hex()
        expect_hex = ['fe20f07c']
        assert hex == expect_hex

if __name__ == '__main__':
    unittest.main()
