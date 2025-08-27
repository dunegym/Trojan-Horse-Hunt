import ctypes
import random
import logging
import unittest
import os

import math
import numpy as np

from xpu.base import XpuException
from xpu.simulator import CDNNSimulator, XPUSimulator, XPUDebugger
from xpu.ctypes_util import to_array
from xpu.ctypes_util import reinterpret_cast


def to_uint32(v):
    return ctypes.c_uint32(v).value


def to_int16(v):
    return ctypes.c_int16(v).value


def split_uint32(value):
    assert isinstance(value, ctypes.c_uint32)
    ptr = ctypes.cast(ctypes.pointer(value), ctypes.POINTER(ctypes.c_uint16))
    return ptr[1], ptr[0]


class InstructionTestCase(unittest.TestCase):
    def setUp(self):
        os.environ['XPUSIM_GM_BASE_ADDR'] = "0"
        os.environ['XPUSIM_GM_SIZE'] = "0xFFFFFFFF"
        os.environ['XPUSIM_L3_BASE_ADDR'] = "0x100000000"
        os.environ['XPUSIM_DEVICE_MODEL'] = "KUNLUN1"

    def tearDown(self):
        del os.environ['XPUSIM_DEVICE_MODEL']
        del os.environ['XPUSIM_GM_BASE_ADDR']
        del os.environ['XPUSIM_GM_SIZE']
        del os.environ['XPUSIM_L3_BASE_ADDR']

    def test_exception(self):
        with self.assertRaises(XpuException):
            asm = "add r0, r0, r0;"
            simulator = XPUSimulator(1)
            simulator.set_asm_source(code=asm)
            simulator.run()

    def test_sync(self):
        # TODO(huangyan13): need a better case
        asm = """
        core_id id;
        rset zero;
        rset one;
        rset two;
        addi one, one, 1;
        beq id, zero, branch0;
        beq id, one, branch1;
        ret;
        branch0:
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
        branch1:
        add zero, zero, zero;
        sync;
        add zero, zero, zero;
        ret;
        """
        for sim in (XPUSimulator(1), CDNNSimulator()):
            sim.set_asm_source(code=asm)
            sim.run(15)
            self.assertEqual(sim.get_register(0, 0, 'pc') + 16,
                             sim.get_register(0, 1, 'pc'))

    def test_shift(self):
        asm1 = '''
        rset r0
        addi r0, r0, {num}
        rset r1
        addi r1, r1, {shift}
        {instr} r0, r0, r1
        ret
        '''
        asm2 = '''
        rset r0
        addi r0, r0, {num}
        {instr} r0, r0, {shift}
        ret
        '''
        for asm, instr, n, shift, f in [
            (asm1, 'sll', 12, 3, lambda a, b: a << b),
            (asm1, 'sll', -12, 3, lambda a, b: a << b),
            (asm1, 'srl', 36, 2, lambda a, b: (a % 0x100000000) >> b),
            (asm1, 'srl', -36, 2, lambda a, b: (a % 0x100000000) >> b),
            (asm1, 'sra', 48, 3, lambda a, b: a >> b),
            (asm1, 'sra', -48, 3, lambda a, b: a >> b),
            (asm2, 'slli', 12, 3, lambda a, b: a << b),
            (asm2, 'slli', -12, 3, lambda a, b: a << b),
            (asm2, 'srli', 36, 2, lambda a, b: (a % 0x100000000) >> b),
            (asm2, 'srli', -36, 2, lambda a, b: (a % 0x100000000) >> b),
        ]:
            sim = XPUSimulator(1)
            sim.set_asm_source(code=asm.format(num=n, shift=shift, instr=instr))
            sim.run()
            self.assertEqual(sim.get_register(0, 0, 'r0'), to_uint32(f(n, shift)))

    def test_fix2float(self):
        asm1 = '''
        rset r0
        addi r1, r0, 1
        fix2float r1, r1
        sub_s r1, r0, r1
        ret
        '''
        asm2 = '''
        rset r0
        subi r1, r0, 1
        fix2float r1, r1
        ret
        '''
        sim1 = XPUSimulator(1)
        sim2 = XPUSimulator(1)
        sim1.set_asm_source(code=asm1)
        sim2.set_asm_source(code=asm2)
        sim1.run()
        sim2.run()
        self.assertEqual(sim1.get_register(0, 0, 'r1'),
                         sim2.get_register(0, 0, 'r1'))

    def test_float2fix(self):
        for rs, ans in [
            # +nan
            (ctypes.c_uint32(0x7fffffff), 0x7fffffff),
            (ctypes.c_uint32(0x7fe7dffe), 0x7fffffff),
            # -nan
            (ctypes.c_uint32(0xffffffff), 0x80000000),
            # +inf
            (ctypes.c_uint32(0x7f800000), 0x7fffffff),
            # -inf
            (ctypes.c_uint32(0xff800000), 0x80000000),
        ]:
            asm = """
            rset r0
            rset r2
            # load r1
            ori r1, r0, 0x{:04x}
            slli r1, r1, 16
            ori r1, r1, 0x{:04x}
            float2fix r2, r1
            ret
            """.format(*(split_uint32(rs)))
            sim = XPUSimulator(1)
            sim.set_asm_source(code=asm)
            sim.fast_run()
            res = sim.get_register(0, 0, 'r2')
            # print '0x{:08x}'.format(res)
            self.assertEqual(res, ans)

    def test_jal(self):
        asm = """
        jal function
        ret
        function:
        nop
        nop
        nop
        nop
        jr r31
        """
        simulator = XPUSimulator(1)
        simulator.set_asm_source(code=asm)
        simulator.run(1)
        self.assertEqual(simulator.get_register(0, 0, 'r31'), 0x04)
        self.assertEqual(simulator.get_register(0, 0, 'pc'), 0x08)
        simulator.run(4)
        self.assertEqual(simulator.get_register(0, 0, 'pc'), 0x18)
        simulator.run(1)
        self.assertEqual(simulator.get_register(0, 0, 'pc'), 0x04)
        simulator.run(1)
        self.assertTrue(simulator.is_end())

    def test_jalr(self):
        asm = """
        rset r0
        addi r0, r0, function
        jalr r31, r0
        ret
        function:
        nop
        nop
        nop
        nop
        jr r31
        """
        simulator = XPUSimulator(1)
        simulator.set_asm_source(code=asm)
        simulator.run(2)
        self.assertEqual(simulator.get_register(0, 0, 'r0'), 0x10)
        simulator.run(1)
        self.assertEqual(simulator.get_register(0, 0, 'r31'), 0x0c)
        self.assertEqual(simulator.get_register(0, 0, 'pc'), 0x10)
        simulator.run(5)
        self.assertEqual(simulator.get_register(0, 0, 'pc'), 0x0c)
        simulator.run(1)
        self.assertTrue(simulator.is_end())

    def test_simd_vv(self):
        """Test of SIMD instructions"""
        instr2func = {
            'vvmul_s': lambda x, y: x * y,
            'vvadd_s': lambda x, y: x + y,
            'vvsub_s': lambda x, y: x - y,
        }
        for instr, func in instr2func.items():
            x0 = random.randint(-10, 10)
            x1 = random.randint(-10, 10)
            asm = """
            rset r0
            addi r1, r0, 32
            addi r2, r1, 32
            addi r3, r0, {}
            addi r4, r0, {}
            fix2float r3, r3
            fix2float r4, r4
            sw_lm r3, 0(r0)
            sw_lm r3, 4(r0)
            sw_lm r3, 8(r0)
            sw_lm r3, 12(r0)
            sw_lm r3, 16(r0)
            sw_lm r3, 20(r0)
            sw_lm r3, 24(r0)
            sw_lm r3, 28(r0)
            sw_lm r4, 0(r1)
            sw_lm r4, 4(r1)
            sw_lm r4, 8(r1)
            sw_lm r4, 12(r1)
            sw_lm r4, 16(r1)
            sw_lm r4, 20(r1)
            sw_lm r4, 24(r1)
            sw_lm r4, 28(r1)
            {} r2, r0, r1
            ret
            """.format(x0, x1, instr)
            simulator = XPUSimulator(1)
            simulator.set_asm_source(code=asm)
            simulator.run()
            actual = [v for v in to_array(simulator.get_local(0, 0, 64, 32), ctypes.c_float)]
            expected = [func(a, b) for a, b in zip([float(x0)] * 8, [float(x1)] * 8)]
            self.assertEqual(actual, expected)

    def test_simd_sv(self):
        """Test of SIMD instructions"""
        instr2func = {
            'svmul_s': lambda x, y: x * y,
            'svadd_s': lambda x, y: x + y,
            'svsub_s': lambda x, y: x - y,
        }
        for instr, func in instr2func.items():
            x0 = random.randint(-10, 10)
            x1 = random.randint(-10, 10)
            asm = """
            rset r0
            addi r1, r0, 32
            addi r2, r0, {}
            addi r3, r0, {}
            fix2float r2, r2
            fix2float r3, r3
            sw_lm r3, 0(r0)
            sw_lm r3, 4(r0)
            sw_lm r3, 8(r0)
            sw_lm r3, 12(r0)
            sw_lm r3, 16(r0)
            sw_lm r3, 20(r0)
            sw_lm r3, 24(r0)
            sw_lm r3, 28(r0)
            {} r1, r2, r0
            ret
            """.format(x0, x1, instr)
            simulator = XPUSimulator(1)
            simulator.set_asm_source(code=asm)
            simulator.run()
            actual = [v for v in to_array(simulator.get_local(0, 0, 32, 32), ctypes.c_float)]
            expected = [func(x0, a) for a in [float(x1)] * 8]
            self.assertEqual(actual, expected)

    def test_bit_arithmetic(self):
        op2func = {
            'and': lambda x, y: x & y,
            'or': lambda x, y: x | y,
            'xor': lambda x, y: x ^ y,
            'nor': lambda x, y: ~(x | y),
        }
        for op, f in op2func.items():
            a = random.randint(0, 0xFFFF)
            b = random.randint(0, 0xFFFF)
            sim = XPUSimulator(1)
            asm = '''
            rset r0;
            addi r1, r0, {};
            addi r2, r0, {};
            {} r0, r1, r2;
            ret;
            '''.format(a, b, op)
            sim.set_asm_source(code=asm)
            sim.run()
            self.assertEqual(sim.get_register(0, 0, 'r0'),
                             to_uint32(f(to_int16(a), to_int16(b))))

    def test_act_index(self):
        """Test table index"""
        asm1 = '''
        rset r0
        addi r1, r0, 3
        fix2float r1, r1
        act_index r1, r0, r1, 128
        ret
        '''
        asm2 = '''
        rset r0
        subi r1, r0, 5
        addi r2, r0, 2
        fix2float r1, r1
        fix2float r2, r2
        div_s r1, r1, r2
        act_index r1, r0, r1, 128
        ret
        '''
        asm3 = '''
        rset r0
        subi r1, r0, 19
        addi r2, r0, 2
        fix2float r1, r1
        fix2float r2, r2
        div_s r1, r1, r2
        act_index r1, r0, r1, 128
        ret
        '''

        sim1 = XPUSimulator(1)
        sim2 = XPUSimulator(1)
        sim3 = XPUSimulator(1)
        sim1.set_asm_source(code=asm1)
        sim2.set_asm_source(code=asm2)
        sim3.set_asm_source(code=asm3)
        sim1.run()
        sim2.run()
        sim3.run()
        print sim1.get_register(0, 0, 'r1')
        self.assertEqual(sim1.get_register(0, 0, 'r1'), (3 * 128 + 512) * 8)
        self.assertEqual(sim2.get_register(0, 0, 'r1'), (int(-2.5 * 128) + 512) * 8)
        self.assertEqual(sim3.get_register(0, 0, 'r1'), 0)

    def test_act_index2(self):
        """Test table index"""
        for rs, rt, imm, ans in [
            (ctypes.c_uint32(0x17c8933d), ctypes.c_uint32(0x8d09a5ee), 256, 0x8d09b5ee),
            (ctypes.c_uint32(0xa3f7310e), ctypes.c_uint32(0xa3f71186), 4, 0xa3f72186),
            (ctypes.c_uint32(0xc0ae5511), ctypes.c_uint32(0xc0ae5511), 0x400, 0xc0ae5511),
        ]:
            asm = """
            rset r0
            # load r1
            ori r1, r0, 0x{:04x}
            slli r1, r1, 16
            ori r1, r1, 0x{:04x}
            # load r2
            ori r2, r0, 0x{:04x}
            slli r2, r2, 16
            ori r2, r2, 0x{:04x}
            act_index r3, r2, r1, {imm}
            ret
            """.format(*(split_uint32(rs) + split_uint32(rt)), imm=imm)
            sim = XPUSimulator(1)
            sim.set_asm_source(code=asm)
            sim.fast_run()
            res = sim.get_register(0, 0, 'r3')
            # print '0x{:08x}'.format(res)
            self.assertEqual(res, ans)


class DebuggerTestCase(unittest.TestCase):
    def setUp(self):
        self.asm = """
        addi zero, zero, 1;
        addi zero, zero, 2;
        addi zero, zero, 3;
        addi zero, zero, 4;
        addi zero, zero, 5;
        addi zero, zero, 6;
        addi zero, zero, 7;
        addi zero, zero, 8;
        addi zero, zero, 9;
        ret;
        """
        os.environ['XPUSIM_DEVICE_MODEL'] = "KUNLUN1"
        os.environ['XPUSIM_GM_BASE_ADDR'] = "0"
        os.environ['XPUSIM_GM_SIZE'] = "0xFFFFFFFF"
        os.environ['XPUSIM_L3_BASE_ADDR'] = "0x100000000"
        os.environ['XPUSIM_MALLOC_BASE'] = "0"

    def tearDown(self):
        del os.environ['XPUSIM_DEVICE_MODEL']
        del os.environ['XPUSIM_GM_BASE_ADDR']
        del os.environ['XPUSIM_GM_SIZE']
        del os.environ['XPUSIM_L3_BASE_ADDR']
        del os.environ['XPUSIM_MALLOC_BASE']

    def test_print(self):
        sim = XPUSimulator(4)
        sim.set_asm_source(code=self.asm)
        xdb = XPUDebugger(sim)
        xdb.get_output(True)
        xdb._print_impl('0.0.r0')
        xdb._print_impl('0.0.r0@int')
        xdb._print_impl('0.0.r0@uint')
        xdb._print_impl('0.0.r0@byte')
        xdb._print_impl('0.0.r0@float')
        xdb._print_impl('0.0.r0@hex')
        xdb._print_impl('0-3.0-3.r0-3@byte')
        xdb._print_impl('0.0-3.r0-3@byte')
        xdb._print_impl('0-3.0.r0-3@byte')
        xdb._print_impl('0.0.r0-3@byte')
        xdb._print_impl('0-3.0-3.r0-3@byte')
        xdb._print_impl('0.1.local[0x00:0x10]@byte')
        xdb._print_impl('0-2.0-2.local[0x00:0x10]@hex')
        xdb._print_impl('0.shared[0x00:0x10]@int')
        xdb._print_impl('3.shared[0x00:0x10]@byte')
        xdb._print_impl('global[0x00:0x20]@float')
        xdb._print_impl('global[128:192]@int')
        xdb._print_impl('0-2.shared[20:32]@byte')
        with self.assertRaises(XPUDebugger.UnknownCommandException):
            xdb._print_impl('0-2.shared[20:fuck]@byte')

    def test_list(self):
        sim = XPUSimulator()
        sim.set_asm_source(code=self.asm)
        xdb = XPUDebugger(sim)
        xdb.get_output(True)
        xdb._list_impl('0.0')
        self.assertTrue('addi zero, zero, 1' in '\n'.join(xdb.get_output()))
        xdb._list_impl('0.0.9')
        self.assertEqual(len(xdb.get_output()), 6)
        xdb._list_impl('0.0.9 0x10')
        self.assertEqual(len(xdb.get_output()), 10)
        xdb._list_impl('0-2.0')
        xdb._list_impl('0-2.0-2.9')
        xdb._list_impl('0.0-2.9 0x10')
        xdb._list_impl('0-2.0.9 0x10')
        xdb._list_impl('0-2.0.9 0x10')

    def test_break(self):
        sim = XPUSimulator(1)
        sim.set_asm_source(code=self.asm)
        xdb = XPUDebugger(sim)
        # add first break point
        xdb._break_impl('0.0 add 0x0c')
        xdb._break_impl('0.0 list')
        self.assertTrue('0x0000000c' in '\n'.join(xdb.get_output(True)))
        xdb.simulator.run()
        self.assertEqual(sim.get_register(0, 0, 'pc'), 0x10)
        # delete breakpoints
        xdb._break_impl('0.0 del 0x10')
        xdb._break_impl('0.0 del 0x0c')
        xdb._break_impl('0.0 list')
        self.assertEqual(len(xdb.get_output(True)), 0)
        # add second break point
        xdb._break_impl('0.0 add 0x10')
        xdb._break_impl('0.0 list')
        self.assertTrue('0x00000010' in '\n'.join(xdb.get_output(True)))
        xdb.simulator.run()
        self.assertEqual(sim.get_register(0, 0, 'pc'), 0x14)
        # unknown command
        with self.assertRaises(XPUDebugger.UnknownCommandException):
            xdb._break_impl('a b c')


class NumericalTestCase(unittest.TestCase):
    def setUp(self):
        os.environ['XPUSIM_DEVICE_MODEL'] = "KUNLUN1"
        os.environ['XPUSIM_GM_BASE_ADDR'] = "0"
        os.environ['XPUSIM_GM_SIZE'] = "0xFFFFFFFF"
        os.environ['XPUSIM_L3_BASE_ADDR'] = "0x100000000"

    def tearDown(self):
        del os.environ['XPUSIM_DEVICE_MODEL']
        del os.environ['XPUSIM_GM_BASE_ADDR']
        del os.environ['XPUSIM_GM_SIZE']
        del os.environ['XPUSIM_L3_BASE_ADDR']

    zero_pos = ctypes.c_uint32(0x00000000)
    zero_neg = ctypes.c_uint32(0x80000000)
    normal_pos = reinterpret_cast(ctypes.c_float(1), ctypes.c_uint32)
    normal_neg = reinterpret_cast(ctypes.c_float(-1), ctypes.c_uint32)
    normal_2_pos = reinterpret_cast(ctypes.c_float(2), ctypes.c_uint32)
    normal_2_neg = reinterpret_cast(ctypes.c_float(-2), ctypes.c_uint32)
    nan_pos = ctypes.c_uint32(0x7fffffff)
    nan_neg = ctypes.c_uint32(0xffffffff)
    inf_pos = ctypes.c_uint32(0x7f800000)
    inf_neg = ctypes.c_uint32(0xff800000)

    value_list = [
        (zero_pos, '+0', 'NumType::ZERO_POS'),
        (zero_neg, '-0', 'NumType::ZERO_NEG'),
        (normal_pos, '+normal', 'NumType::NORMAL_POS'),
        (normal_neg, '-normal', 'NumType::NORMAL_NEG'),
        (nan_pos, '+NaN', 'NumType::NAN_POS'),
        (nan_neg, '-NaN', 'NumType::NAN_NEG'),
        (inf_pos, '+Inf', 'NumType::INF_POS'),
        (inf_neg, '-Inf', 'NumType::INF_NEG')
    ]

    result_table = {
        'mul_s': [
            [zero_pos, zero_neg, zero_pos, zero_neg, inf_pos, inf_pos, inf_pos, inf_pos],
            [zero_neg, zero_pos, zero_neg, zero_pos, inf_pos, inf_pos, inf_pos, inf_pos],
            [zero_pos, zero_neg, normal_pos, normal_neg, inf_pos, inf_neg, inf_pos, inf_neg],
            [zero_neg, zero_pos, normal_neg, normal_pos, inf_neg, inf_pos, inf_neg, inf_pos],
            [inf_pos, inf_pos, inf_pos, inf_neg, inf_pos, inf_neg, inf_pos, inf_neg],
            [inf_pos, inf_pos, inf_neg, inf_pos, inf_neg, inf_pos, inf_neg, inf_pos],
            [inf_pos, inf_pos, inf_pos, inf_neg, inf_pos, inf_neg, inf_pos, inf_neg],
            [inf_pos, inf_pos, inf_neg, inf_pos, inf_neg, inf_pos, inf_neg, inf_pos],
        ],
        'div_s': [
            [inf_pos, inf_pos, zero_pos, zero_neg, zero_pos, zero_neg, zero_pos, zero_neg],
            [inf_pos, inf_pos, zero_neg, zero_pos, zero_neg, zero_pos, zero_neg, zero_pos],
            [inf_pos, inf_neg, normal_pos, normal_neg, zero_pos, zero_neg, zero_pos, zero_neg],
            [inf_neg, inf_pos, normal_neg, normal_pos, zero_neg, zero_pos, zero_neg, zero_pos],
            [inf_pos, inf_neg, inf_pos, inf_neg, inf_pos, inf_pos, inf_pos, inf_pos],
            [inf_neg, inf_pos, inf_neg, inf_pos, inf_pos, inf_pos, inf_pos, inf_pos],
            [inf_pos, inf_neg, inf_pos, inf_neg, inf_pos, inf_pos, inf_pos, inf_pos],
            [inf_neg, inf_pos, inf_neg, inf_pos, inf_pos, inf_pos, inf_pos, inf_pos],
        ],
        'add_s': [
            [zero_pos, zero_pos, normal_pos, normal_neg, inf_pos, inf_neg, inf_pos, inf_neg],
            [zero_pos, zero_neg, normal_pos, normal_neg, inf_pos, inf_neg, inf_pos, inf_neg],
            [normal_pos, normal_pos, normal_2_pos, zero_pos, inf_pos, inf_neg, inf_pos, inf_neg],
            [normal_neg, normal_neg, zero_pos, normal_2_neg, inf_pos, inf_neg, inf_pos, inf_neg],
            [inf_pos, inf_pos, inf_pos, inf_pos, inf_pos, inf_pos, inf_pos, inf_pos],
            [inf_neg, inf_neg, inf_neg, inf_neg, inf_pos, inf_neg, inf_pos, inf_neg],
            [inf_pos, inf_pos, inf_pos, inf_pos, inf_pos, inf_pos, inf_pos, inf_pos],
            [inf_neg, inf_neg, inf_neg, inf_neg, inf_pos, inf_neg, inf_pos, inf_neg],
        ]
    }

    def test_sfu_binary(self):
        for instr, f in [('mul_s', lambda x, y: x * y),
                         ('pow_s', math.pow)]:
            lhs = 3.1415926
            rhs = 2.7182818
            asm = """
            rset r0
            # load r1
            ori r1, r0, 0x{:04x}
            slli r1, r1, 16
            ori r1, r1, 0x{:04x}
            # load r2
            ori r2, r0, 0x{:04x}
            slli r2, r2, 16
            ori r2, r2, 0x{:04x}
            {instr} r3, r1, r2
            ret
            """.format(instr=instr,
                       *(split_uint32(reinterpret_cast(ctypes.c_float(lhs), ctypes.c_uint32))) +
                         split_uint32(reinterpret_cast(ctypes.c_float(rhs), ctypes.c_uint32)))
            sim = XPUSimulator(1)
            sim.set_asm_source(code=asm)
            sim.fast_run()
            ans = f(lhs, rhs)
            res = reinterpret_cast(ctypes.c_uint32(sim.get_register(0, 0, 'r3')), ctypes.c_float).value
            self.assertAlmostEqual(ans, res, places=4)

    def test_sfu_unary(self):
        for instr, f in [('exp_s', math.exp),
                         ('log_s', math.log),
                         ('sqrt_s', math.sqrt)]:
            val = 3.1415926
            asm = """
            rset r0
            # load r1
            ori r1, r0, 0x{:04x}
            slli r1, r1, 16
            ori r1, r1, 0x{:04x}
            {instr} r2, r1
            ret
            """
            sim = XPUSimulator(1)
            sim.set_asm_source(code=asm.format(instr=instr, *split_uint32(
                reinterpret_cast(ctypes.c_float(val), ctypes.c_uint32))))
            sim.fast_run()
            ans = f(val)
            res = reinterpret_cast(ctypes.c_uint32(sim.get_register(0, 0, 'r2')), ctypes.c_float).value
            self.assertAlmostEqual(ans, res, places=4)

    def test_abnormal(self):
        asm = """
        rset r0
        # load r1
        ori r1, r0, 0x{:04x}
        slli r1, r1, 16
        ori r1, r1, 0x{:04x}
        # load r2
        ori r2, r0, 0x{:04x}
        slli r2, r2, 16
        ori r2, r2, 0x{:04x}
        {instr} r3, r1, r2
        ret
        """

        def get_ans_name(result_value):
            for val, name, _ in self.value_list:
                if val.value == result_value:
                    return name
            raise RuntimeError

        for instr in self.result_table.keys():
            for il, lhs in enumerate(self.value_list):
                for ir, rhs in enumerate(self.value_list):
                    ans = self.result_table[instr][il][ir]
                    sim = XPUSimulator(1)
                    sim.set_asm_source(code=asm.format(instr=instr,
                                                       *(split_uint32(lhs[0]) +
                                                         split_uint32(rhs[0]))))
                    sim.fast_run()
                    res = sim.get_register(0, 0, 'r3')
                    if res != ans.value:
                        # print '{}({}, {}) shoule be {} (actual 0x{:08x})'.format(instr, lhs[1], rhs[1], get_ans_name(ans.value), res)
                        # print 'insert(map, {}, {}, {});'.format(lhs[2], rhs[2], get_ans_name(ans.value))
                        self.assertEqual(sim.get_register(0, 0, 'r0'), ans.value)

    def test_denormal(self):
        """
        Denormal float input should be treated as zero.
        """
        asm = """
        rset r0
        # load r1
        ori r1, r0, 0x{:04x}
        slli r1, r1, 16
        ori r1, r1, 0x{:04x}
        # load r2
        ori r2, r0, 0x{:04x}
        slli r2, r2, 16
        ori r2, r2, 0x{:04x}
        {instr} r3, r1, r2
        ret
        """
        for instr, lhs, rhs, ans in [
            ('add_s', self.inf_pos, ctypes.c_uint32(0x2d40), self.inf_pos),
            ('add_s', self.inf_neg, ctypes.c_uint32(0x2d40), self.inf_neg),
            ('mul_s', ctypes.c_uint32(0x5271a95), ctypes.c_uint32(0x2f061c1c), self.zero_pos),
            ('mul_s', ctypes.c_uint32(0x5271a95), ctypes.c_uint32(0xaf061c1c), self.zero_neg),
            ('sub_s', ctypes.c_uint32(0xf3d7c4b6), ctypes.c_uint32(0xff800000), ctypes.c_uint32(0x7f800000)),
            ('sub_s', ctypes.c_uint32(0xff800000), ctypes.c_uint32(0xff800000), ctypes.c_uint32(0x7f800000)),
            ('sub_s', ctypes.c_uint32(0xff4b4503), ctypes.c_uint32(0x7f800000), ctypes.c_uint32(0xff800000)),
            ('max_s', ctypes.c_uint32(0x787207dd), ctypes.c_uint32(0x7fc0d942), ctypes.c_uint32(0x7fc0d942)),
            ('min_s', ctypes.c_uint32(0x787207dd), ctypes.c_uint32(0x7fc0d942), ctypes.c_uint32(0x787207dd)),
        ]:
            sim = XPUSimulator(1)
            sim.set_asm_source(code=asm.format(instr=instr,
                                               *(split_uint32(lhs) +
                                                 split_uint32(rhs))))
            sim.fast_run()
            res = sim.get_register(0, 0, 'r3')
            # print '0x{:08x}'.format(res)
            self.assertEqual(res, ans.value)


if __name__ == '__main__':
    os.environ['XPUSIM_GM_BASE_ADDR'] = "0"
    os.environ['XPUSIM_GM_SIZE'] = "0xFFFFFFFF"
    os.environ['XPUSIM_L3_BASE_ADDR'] = "0x100000000"
    unittest.main()
    del os.environ['XPUSIM_GM_BASE_ADDR']
    del os.environ['XPUSIM_GM_SIZE']
    del os.environ['XPUSIM_L3_BASE_ADDR']
