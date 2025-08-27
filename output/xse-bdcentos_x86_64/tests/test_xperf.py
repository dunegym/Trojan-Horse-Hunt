import ctypes
import random
import logging
import unittest
import os

import numpy as np

from xpu.simulator import XPUSimulator

'''
class XPerfTestCase(unittest.TestCase):
    def test_reg_hazard(self):
        sim = XPUSimulator(1, xperf=True)
        asm = """
        add r0, r1, r2;
        add r3, r0, r4;
        add r5, r4, r3;
        add r5, r6, r7;
        ret;
        """
        sim.set_asm_source(code=asm)
        sim.fast_run()
        stat = sim.get_core(0, 0).get_core_stat()
        instr = sim.get_core(0, 0).get_instr_stat()
        self.assertEqual(stat.reg_hazard_cycle, 4)
        self.assertEqual(instr[1].decode_time, instr[0].retire_time + 1)
        self.assertEqual(instr[2].decode_time, instr[1].retire_time + 1)
        self.assertEqual(instr[3].decode_time, instr[2].dispatch_time)

    def test_sfu_hazard(self):
        sim = XPUSimulator(1, xperf=True)
        asm = """
        {}
        {}
        ret;
        """.format(
            '\n'.join(['rset r{};'.format(i) for i in range(32)]),
            '\n'.join(['exp_s r{}, r{};'.format(i + 1, i) for i in range(31)]))
        sim.set_asm_source(code=asm)
        sim.fast_run()
        instr = sim.get_core(0, 0).get_instr_stat()
        stat = sim.get_core(0, 0).get_core_stat()
        # should have 30 hazards
        self.assertEqual(stat.reg_hazard_cnt, 30)
        # stall the pipeline for 480 cycles
        self.assertGreaterEqual(stat.reg_hazard_cycle, 440)
        # total cycles should be about larger than 500, less than 600
        self.assertTrue(instr[-2].retire_time > 500)
        self.assertTrue(instr[-2].retire_time < 550)

    def test_sfu_no_hazard(self):
        sim = XPUSimulator(1, xperf=True)
        asm = """
        {}
        {}
        ret;
        """.format(
            '\n'.join(['rset r{};'.format(i) for i in range(32)]),
            '\n'.join(['exp_s r{}, r{};'.format(i, i) for i in range(31)]))
        sim.set_asm_source(code=asm)
        sim.fast_run()
        instr = sim.get_core(0, 0).get_instr_stat()
        stat = sim.get_core(0, 0).get_core_stat()
        # should have no reg hazard
        self.assertEqual(stat.reg_hazard_cnt, 0)
        self.assertTrue(instr[-2].retire_time > 500)
        self.assertTrue(instr[-2].retire_time < 550)

    def test_sync(self):
        sim = XPUSimulator(1, xperf=True)
        asm = """
        core_id cid;
        rset zero;
        bne cid, zero, branch;
        add zero, zero, zero;
        add zero, zero, zero;
        add zero, zero, zero;
        add zero, zero, zero;
        add zero, zero, zero;
        add zero, zero, zero;
        add zero, zero, zero;
        add zero, zero, zero;
        sync;
        ret;
        branch:
        add zero, zero, zero;
        sync;
        ret;
        """
        # open('code_binary.s', 'w').write(asm)
        sim.set_asm_source(code=asm)
        sim.fast_run()
        stat = sim.get_core(0, 0).get_core_stat()
        core0 = sim.get_core(0, 0).get_instr_stat()
        core1 = sim.get_core(0, 1).get_instr_stat()
        # print stat
        # for i in core0[-5:]:
        #     print i
        # for i in core1[-5:]:
        #     print i
        self.assertEqual(stat.reg_hazard_cnt, 8)
        self.assertLessEqual(abs(core0[-1].retire_time - core1[-1].retire_time), 1)
        self.assertEqual(core0[-2].retire_time, core1[-2].retire_time)

    def test_lm_read(self):
        sim = XPUSimulator(1, xperf=True)
        asm = """
        rset r0;
        addi r1, r0, 2;
        addi r0, r0, 64;
        div r0, r0, r1;
        lw_lm r1, 28(r0);
        ret;
        """
        sim.set_asm_source(code=asm)
        sim.fast_run()
        core0 = sim.get_core(0, 0).get_instr_stat()
        # for i in core0:
        #     print i
        # check if reg_hazard works correct
        self.assertEqual(core0[2].retire_time + 1, core0[3].decode_time)
        self.assertEqual(core0[3].retire_time + 1, core0[4].decode_time)
        self.assertGreaterEqual(core0[-1].retire_time, 19)

    def test_lm_write(self):
        sim = XPUSimulator(1, xperf=True)
        asm = """
        rset r0;
        vvadd_s r0, r0, r0;
        sw_lm r1, 0(r0);
        ret;
        """
        sim.set_asm_source(code=asm)
        sim.fast_run()
        core0 = sim.get_core(0, 0).get_instr_stat()
        stat = sim.get_core(0, 0).get_core_stat()
        self.assertEqual(stat.reg_hazard_cnt, 1)
        self.assertEqual(stat.lm_hazard_cycle, 2)
        # print stat
        # for i in core0[-5:]:
        #     print i
        self.assertLess(core0[-1].dispatch_time, core0[1].dispatch_time)

    def test_simd(self):
        sim = XPUSimulator(1, xperf=True)
        asm = """
        rset r0;
        {}
        ret;
        """.format('vvadd_s r0, r0, r0;\n' * 44)
        sim.set_asm_source(code=asm)
        sim.fast_run()
        core0 = sim.get_core(0, 0).get_instr_stat()
        stat = sim.get_core(0, 0).get_core_stat()
        # print stat
        # for i in core0[-5:]:
        #     print i
        # RTL result: ???
        self.assertTrue(530 <= core0[-1].retire_time <= 540)
        # RTL result: 473
        self.assertTrue(470 <= stat.lm_hazard_cycle <= 480)

    def test_simd_no_pipelined(self):
        sim = XPUSimulator(1, xperf=True)
        asm = """
        rset zero;
        # loop for 128 times
        addi loop_cnt, zero, 128;
        addi i, zero, 0;
        addi x0, zero, 0;
        addi x1, zero, 4096;
        addi x2, zero, 8192;
        begin:
        vvadd_s x0, x1, x2;
        addi x0, x0, 32;
        addi x1, x1, 32;
        addi x2, x2, 32;
        addi i, i, 1;
        blt i, loop_cnt, begin;
        ret;
        """
        sim.set_asm_source(code=asm)
        sim.fast_run()
        stat = sim.get_core(0, 0).get_core_stat()
        # print stat
        self.assertEqual(stat.reg_hazard_cycle, 260)
        self.assertEqual(stat.reg_hazard_cnt, 130)
        self.assertEqual(stat.branch_cnt, 128)
        self.assertEqual(stat.miss_cnt, 2)

    def test_nop(self):
        sim = XPUSimulator(1, xperf=True)
        asm = """
        rset r0;
        vvadd_s r0, r0, r0;
        nop;
        nop;
        nop;
        sw_lm r1, 0(r0);
        nop;
        ret;
        """
        sim.set_asm_source(code=asm)
        sim.fast_run()
        stat = sim.get_core(0, 0).get_core_stat()
        self.assertEqual(stat.lm_hazard_cycle, 0)

    def test_sm2lm(self):
        sim = XPUSimulator(1, xperf=True)
        asm = """
        rset r0;
        addi r1, r0, 64;
        l_sm2lm r1, r0, 0(r0);
        l_sm2lm r1, r0, 0(r0);
        l_sm2lm r1, r0, 0(r0);
        mfence;
        addi r1, r0, 0;
        ret;
        """
        sim.set_asm_source(code=asm)
        sim.fast_run()
        core0 = sim.get_core(0, 0).get_instr_stat()
        core1 = sim.get_core(0, 1).get_instr_stat()
        # stat = sim.get_core(0, 0).get_core_stat()
        # print stat
        # for i in core0:
        #     print i
        # print
        # for i in core1:
        #     print i
        self.assertEqual(core0[2].retire_time + 6, core1[2].retire_time)
        self.assertEqual(core0[3].retire_time + 6, core1[3].retire_time)
        self.assertEqual(core0[4].retire_time + 6, core1[4].retire_time)
        self.assertGreaterEqual(core0[6].retire_time + 6, core0[5].retire_time)

    def test_sm2lm_with_nop(self):
        sim = XPUSimulator(1, xperf=True)
        asm = """
        rset r0;
        addi r1, r0, 64;
        l_sm2lm r1, r0, 0(r0);
        nop;
        l_sm2lm r1, r0, 0(r0);
        nop;
        l_sm2lm r1, r0, 0(r0);
        nop;
        mfence;
        addi r1, r0, 0;
        ret;
        """
        sim.set_asm_source(code=asm)
        sim.fast_run()
        core0 = sim.get_core(0, 0).get_instr_stat()
        core0.sort(key=lambda x: x.index)
        # stat = sim.get_core(0, 0).get_core_stat()
        # print stat
        # for i in core0:
        #     print i
        # NOP retired earlier than memory instr
        self.assertEqual(core0[-4].retire_time + 5, core0[-5].retire_time)
        self.assertEqual(core0[-6].retire_time + 5, core0[-7].retire_time)

    def test_lm2sm(self):
        sim = XPUSimulator(1, xperf=True)
        asm = """
        rset r0;
        addi r1, r0, 64;
        s_lm2sm r1, r0, 0(r0);
        s_lm2sm r1, r0, 0(r0);
        s_lm2sm r1, r0, 0(r0);
        mfence;
        addi r1, r0, 0;
        ret;
        """
        sim.set_asm_source(code=asm)
        sim.fast_run()
        core0 = sim.get_core(0, 0).get_instr_stat()
        core1 = sim.get_core(0, 1).get_instr_stat()
        # stat = sim.get_core(0, 0).get_core_stat()
        # print stat
        # for i in core0:
        #     print i
        # print
        # for i in core1:
        #     print i
        self.assertEqual(core0[2].retire_time + 6, core1[2].retire_time)
        self.assertEqual(core0[3].retire_time + 6, core1[3].retire_time)
        self.assertEqual(core0[4].retire_time + 6, core1[4].retire_time)
        self.assertGreaterEqual(core0[6].retire_time + 6, core0[5].retire_time)

    def test_sm_gm(self):
        sim = XPUSimulator(1, xperf=True)
        asm = """
        rset r0;
        addi r1, r0, 32;
        s_sm2gm r1, r0, 0(r0);
        addi r1, r0, 512;
        s_sm2gm r1, r0, 0(r0);
        l_gm2sm r1, r0, 0(r0);
        mfence;
        addi r1, r0, 0;
        ret;
        """
        sim.set_asm_source(code=asm)
        sim.fast_run()
        core0 = sim.get_core(0, 0).get_instr_stat()
        core0.sort(key=lambda x: x.index)
        self.assertEqual(core0[1].retire_time + 1, core0[2].decode_time)
        self.assertEqual(core0[3].retire_time + 1, core0[4].decode_time)
        # `s_sm2gm` and `l_gm2sm` should retire at same time
        self.assertEqual(core0[5].retire_time, core0[4].retire_time)

    def test_gm2lm(self):
        sim = XPUSimulator(1, xperf=True)
        asm = """
        rset r0;
        addi r1, r0, 32;
        l_gm2lm r1, r0, 0(r0);
        addi r1, r0, 512;
        l_gm2lm r1, r0, 0(r0);
        mfence;
        addi r1, r0, 0;
        ret;
        """
        sim.set_asm_source(code=asm)
        sim.fast_run()
        core0 = sim.get_core(0, 0).get_instr_stat()
        core0.sort(key=lambda x: x.index)
        # for i in core0:
        #     print i
        self.assertGreaterEqual((core0[4].retire_time - core0[4].dispatch_time),
                                15 + (core0[2].retire_time - core0[2].dispatch_time))

    def test_lm2gm(self):
        sim = XPUSimulator(1, xperf=True)
        asm = """
        rset r0;
        addi r1, r0, 128;
        addi r2, r0, 512;
        s_lm2gm r1, r0, 0(r0);
        s_lm2gm r2, r0, 0(r0);
        mfence;
        addi r1, r0, 0;
        ret;
        """
        sim.set_asm_source(code=asm)
        sim.fast_run()
        core0 = sim.get_core(0, 0).get_instr_stat()
        core0.sort(key=lambda x: x.index)
        self.assertGreater((core0[4].retire_time - core0[4].dispatch_time),
                           2 * (core0[2].retire_time - core0[2].dispatch_time))
'''

if __name__ == '__main__':
    os.environ['XPUSIM_GM_BASE_ADDR'] = "0"
    os.environ['XPUSIM_GM_SIZE'] = "0xFFFFFFFF"
    os.environ['XPUSIM_L3_BASE_ADDR'] = "0x100000000"
    unittest.main()
    del os.environ['XPUSIM_GM_BASE_ADDR']
    del os.environ['XPUSIM_GM_SIZE']
    del os.environ['XPUSIM_L3_BASE_ADDR']
