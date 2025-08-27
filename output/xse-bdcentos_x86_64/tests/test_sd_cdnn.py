import ctypes
import random
import logging
import unittest
import math
import struct
import sys
import os
import numpy as np

np.set_printoptions(threshold=np.inf)

from xpu.simulator import CDNNSimulator
from xpu.math_op import get_sdcdnn_tanh_table
from xpu.math_op import get_sdcdnn_nexp_table

def to_uint32(v):
    return ctypes.c_uint32(v).value


def to_int16(v):
    return ctypes.c_int16(v).value


# EW_BANK_NUM = 64
EW_BANK_NUM = 16


def float_to_hex(f):
    return hex(struct.unpack('<I', struct.pack('<f', f))[0])


def addr2bankoff(addr, bank_bits=4, align_bits=1):
    align_mask = (1 << align_bits) - 1
    bank = ((addr >> align_bits) & ((1 << bank_bits) - 1))
    offset = ((addr >> (bank_bits + align_bits)) << align_bits) | (addr & align_mask)
    return bank, offset

def addr2bankoff_8bit(addr, bank_bits=4, align_bits=1):
    pos = addr & 0x1
    addr = addr & ~0x1
    align_mask = (1 << align_bits) - 1
    bank = ((addr >> align_bits) & ((1 << bank_bits) - 1))
    offset = addr2bankoffbase_8bit(addr, bank_bits, align_bits)
    local_offset = (addr >> (bank_bits + align_bits)) & (1 << 4 - 1)
    if pos == 1:
        offset += 16
    offset += local_offset
    return bank, offset

def addr2bankoffbase_8bit(addr, bank_bits=4, align_bits=1):
    offset_base = ((addr >> (4 + bank_bits + align_bits)) << (4 + align_bits))
    return offset_base

def bankoff2addr(bank, offset, bank_bits=6, align_bits=1):
    align_mask = (1 << align_bits) - 1
    off_low = offset & align_mask
    off_hi = offset >> align_bits
    addr = (off_hi << (bank_bits + align_bits)) | (bank << align_bits) | off_low
    return addr


def null_func(x):
    return x


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def relu(x):
    return np.maximum(0, x)


def tanh(x):
    return math.tanh(x)

def align(v, target):
    return (v + target - 1) / target * target

def caffe_resize(src, src_height, src_width,
                dst, dst_height, dst_width):
    scale_w = src_width / dst_width
    scale_h = src_height / dst_height
    dst_data = dst
    src_data = src
    for dst_h in range(0, dst_height):
        fh = dst_h * scale_h
        src_h = math.floor(fh)
        fh -= src_h
        w_h0 = abs(1.0 - fh)
        w_h1 = abs(fh)
        dst_offset_1 = dst_h * dst_width
        src_offset_1 = src_h * src_width
        for dst_w in range(0, dst_width):
            fw = dst_w * scale_w
            src_w = math.floor(fw)
            fw -= src_w
            w_w0 = abs(1.0 - fw)
            w_w1 = abs(fw)

            dst_value = 0
            src_idx = int(src_offset_1 + src_w)
            dst_value += (w_h0 * w_w0 * src_data[src_idx])
            flag = 0
            if src_w + 1 < src_width:
                dst_value += (w_h0 * w_w1 * src_data[src_idx + 1])
                flag += 1
            if src_h + 1 < src_height:
                dst_value += (w_h1 * w_w0 * src_data[src_idx + src_width])
                flag += 1
            if flag > 1:
                dst_value += (w_h1 * w_w1 * src_data[src_idx + src_width + 1])
            dst_data[dst_offset_1] = dst_value
            dst_offset_1 += 1

def resize_func(src_addr, dst_addr,
                factor_w, factor_h,
                res_w, res_h,
                src_w, src_h,
                res_offset_w, res_offset_h,
                src_offset_w, src_offset_h):
    for i in range(0, res_h):
        h_cnt = res_offset_h + i
        h_cnt_fp = h_cnt
        src_y = math.floor(h_cnt * factor_h) - src_offset_h
        alpha = h_cnt_fp * factor_h - src_offset_h - src_y
        weight_h0 = 1 - alpha
        weight_h1 = alpha
        for j in range(0, res_w):
            w_cnt = res_offset_w + j
            w_cnt_fp = w_cnt
            src_x = math.floor(w_cnt_fp * factor_w) - src_offset_w
            belta = w_cnt_fp * factor_w - src_offset_w - src_x
            weight_w0 = 1 - belta
            weight_w1 = belta
            resize_read_addr_offset = int(src_y * src_w + src_x)
            if src_x < src_w and src_y < src_h:
                res_tmp0 = (weight_h0 * weight_w0) * src_addr[resize_read_addr_offset]
            else:
                res_tmp0 = 0
            if src_x + 1 < src_w and src_y < src_h:
                res_tmp1 = (weight_h0 * weight_w1) * src_addr[resize_read_addr_offset + 1]
            else:
                res_tmp1 = 0
            if src_x < src_w and src_y + 1 < src_h:
                res_tmp2 = (weight_h1 * weight_w0) * src_addr[resize_read_addr_offset + src_w]
            else:
                res_tmp2 = 0
            if src_x + 1 < src_w and src_y + 1 < src_h:
                res_tmp3 = (weight_h1 * weight_w1) * src_addr[resize_read_addr_offset + src_w + 1]
            else:
                res_tmp3 = 0
            res_tmp0 += res_tmp1
            res_tmp2 += res_tmp3
            reg = res_tmp0 + res_tmp2
            dst_addr[i * res_w + j] = reg

def pooling_func(src_addr_base, dst_addr_base,
                    dst_index_addr_base,
                    pooling_type, pooling_pad_type,
                    input_w, input_h,
                    padding_left, padding_right,
                    padding_top, padding_bottom,
                    filter_w, filter_h,
                    stride_w, stride_h,
                    output_w, output_h):

    for i in range(0, output_h * output_w):
        if pooling_type == 0:
            dst_addr_base[i] = -sys.float_info.max
            dst_index_addr_base[i] = -1
        elif pooling_type == 1:
            dst_addr_base[i] = 0.0

    #construct input with pad
    input_with_pad_w = input_w + padding_left + padding_right
    input_with_pad_h = input_h + padding_top + padding_bottom
    input_range_w = (output_w - 1) * stride_w + filter_w
    input_range_h = (output_h - 1) * stride_h + filter_h
    if input_range_w > input_with_pad_w:
        input_with_pad_w = input_range_w
    if input_range_h > input_with_pad_h:
        input_with_pad_h = input_range_h

    input_with_pad = np.zeros((input_with_pad_w * input_with_pad_h), dtype=np.float32)
    for i in range(0, input_h):
        for j in range(0, input_w):
            input_with_pad[(padding_top + i) * input_with_pad_w + padding_left + j] = src_addr_base[i * input_w + j]

    #pooling for input_with_pad
    for ph in range(0, output_h):
        for pw in range(0, output_w):
            hstart = ph * stride_h
            wstart = pw * stride_w
            hend = min(hstart + filter_h, input_with_pad_h)
            wend = min(wstart + filter_w, input_with_pad_w)
            hstart = max(hstart, 0)
            wstart = max(wstart, 0)
            output_index = ph * output_w + pw

            pooled_size = 0
            for h in range(hstart, hend):
                for w in range(wstart, wend):
                    index = h * input_with_pad_w + w
                    if pooling_type == 0:
                        if input_with_pad[index] > dst_addr_base[output_index]:
                            dst_addr_base[output_index] = input_with_pad[index]
                            dst_index_addr_base[output_index] = (h - hstart) * filter_w + (w - wstart)
                    elif pooling_type == 1:
                        dst_addr_base[output_index] += input_with_pad[index]
                        if pooling_pad_type == 1:
                            if h >= padding_top and h < padding_top + input_h and w >= padding_left and w < padding_left + input_w:
                                pooled_size = pooled_size + 1
                        elif pooling_pad_type == 0:
                            if h < padding_top + input_h + padding_bottom and w < padding_left + input_w + padding_right:
                                pooled_size = pooled_size + 1

            if pooling_type == 1:
                if pooled_size == 0:
                    dst_addr_base[output_index] = 0
                else:
                    dst_addr_base[output_index] /= pooled_size


def vec_reform_int8(vec, banks=16):
    np_dt=np.int8
    m = vec.shape[0]
    k = vec.shape[1]

    if (k % 16) != 0:
        kk = k / 16 + 1
        rfm = np.hstack((vec.astype(np_dt), np.zeros((m, (16 - k % 16)), dtype=np_dt))).reshape(m, kk, 16)
    else:
        kk = k / 16
        rfm = vec.reshape(m, kk, 16).astype(np_dt)
    # print "vec aligned to 16"
    # print rfm
    # print rfm.shape

    empty_beat = np.zeros((1, 16), dtype=np_dt)
    rfm_array = []
    out_m = m / 32 * 16 + m % 32
    # print "out_m =", out_m
    for idx in range(out_m):
        bid = idx / 16
        midx = idx % 16 + bid * 32
        reform_row = []
        for kidx in range(kk):
            if midx < m:
                reform_row.append(rfm[midx][kidx])
            else:
                reform_row.append(empty_beat[0])
            if midx + banks < m:
                reform_row.append(rfm[midx + banks][kidx])
            else:
                reform_row.append(empty_beat[0])
        reform_row = np.hstack(reform_row)
        rfm_array.append(reform_row)
    rfm = np.vstack(rfm_array)
    # print "rfm shape and value"
    # print rfm.shape
    return rfm

def wrap_int4(vec):
    m = vec.shape[0]
    k = vec.shape[1]
    assert(k % 2 ==0)
    tmp = vec.reshape(m * k / 2, 2)
    out = np.ndarray(m * k / 2).astype(np.int8)
    for i in range(tmp.shape[0]):
        out[i] = (tmp[i][0] & 0xF) | ((tmp[i][1] & 0xF) << 4)
    out = out.reshape(m, k / 2)
    return out

def is_int8_mac(data_type):
    if data_type == 0b10000 or data_type == 0b10010 or data_type == 0b10001 or data_type == 0b10011:
        return True
    return False

def conv_win2vec_numpy(image, c, h, w, winh, winw, stride, pad_l, pad_r, pad_t, pad_b, nptype):
    def roundup(n, k):
        return (n + k - 1) / k * k
    #format of image is CHW
    image_with_pad = np.zeros((c, h + pad_l + pad_r, w + pad_t + pad_b)).astype(nptype)
    image_with_pad[:, pad_t:(h + pad_t), pad_l:(w + pad_l)] = image
    vec_id = 0
    outputh = (h + pad_t + pad_b - winh) / stride + 1
    outputw = (w + pad_l + pad_r - winw) / stride + 1
    result = np.zeros((outputh * outputw, c * winh * winw)).astype(nptype)
    for h_start in range(0, h + pad_t + pad_b - winh + 1, stride):
        for w_start in range(0, w + pad_l + pad_b - winw + 1, stride):
            vec = image_with_pad[0, h_start: (h_start + winh),
                    w_start: (w_start + winw)].reshape(-1)
            for cc in range(1, c):
                vec = np.append(vec, image_with_pad[cc, h_start: (h_start + winh),
                        w_start: (w_start + winw)].reshape(-1))
            result[vec_id] = vec
            vec_id = vec_id + 1
    return result


def extract_result_from_L1(sim, cluster_id, sram_id, dim, active_column_list, dst_addr, nptype):
    if (nptype == np.int16):
        dst_bank_id, dst_bank_offset = addr2bankoff(dst_addr, bank_bits = 4, align_bits=1)
        roundup_dim = (dim + 15) / 16 * 16  # round up to times of 16
        size_per_bank = len(active_column_list) * roundup_dim * 2 # 2 is sizeof(int16)
        tmp = []
        for i in range(16):
            ans = sim.get_sram_data_bybank(cluster_id, sram_id, i, dst_bank_offset, size_per_bank)
            tmp = tmp + [np.frombuffer(ans, dtype = np.int16)]
        result = np.zeros((sum(active_column_list), dim)).astype(np.int16)
        result_row_idx = 0
        bank_offset = 0
        for col in active_column_list:
            for bank_id in range(col):
                result[result_row_idx] = tmp[bank_id][bank_offset: (bank_offset + dim)]
                result_row_idx = result_row_idx + 1
            bank_offset = bank_offset + roundup_dim
        return result
    if (nptype == np.int8):
        dst_bank_id, dst_bank_offset = addr2bankoff_8bit(dst_addr, bank_bits = 4, align_bits = 1)
        dst_offset_base = addr2bankoffbase_8bit(dst_addr, bank_bits = 4, align_bits = 1)
        dst_local_offset = (dst_addr >> (4 + 1)) & 0xf
        pos = dst_addr & 0x1

        size_per_bank = len(active_column_list) * dim * 2 # each bank has 2 column

        result = np.zeros((sum(active_column_list), dim)).astype(np.int8)
        dst_offset_index = 0
        used_len = 0
        temp_len = 0
        if pos == 1:
            dst_offset_index = dst_offset_base + 16
        used_len = min(16 - dst_local_offset, dim)
        for i in range(sum(active_column_list)):
            ans = sim.get_sram_data_bybank(cluster_id, sram_id, i, dst_offset_index + dst_local_offset, used_len)
            result[i][0: used_len] = np.frombuffer(ans, dtype = np.int8).reshape(-1)
        dst_offset_index += 32

        while used_len < dim:
            temp_len = min(dim - used_len, 16)
            for i in range(sum(active_column_list)):
                ans = sim.get_sram_data_bybank(cluster_id, sram_id, i, dst_offset_index, temp_len)
                result[i][used_len: used_len + temp_len] = np.frombuffer(ans, dtype = np.int8).reshape(-1)
            used_len += temp_len
            dst_offset_index += 32

        return result

# type - type of the lookup table
# 0 - tanh + sigmoid
# 1 - negative side of exp
def load_act_table(sim, type=0):
    # ew_tab_offset = src1_offset + 4 * stream_len
    ew_tab_offset = 0
    tabk_addr = bankoff2addr(0, ew_tab_offset, 4, 2)
    tabb_addr = bankoff2addr(0, ew_tab_offset+4*512, 4, 2)
    cluster_id = 0

    #  act_mode - activation mode
    #  0 - tanh + sigmoid
    #  1 - centrosymmetric, input could be positive and negative
    #  2 - mirro symmetric, input could be positive and negative
    #  3 - negative side, input could only be negative
    #  4 - both side, input could only be negative
    if type == 0:
        act_mode = 0
        ew_table = get_sdcdnn_tanh_table()
    elif type == 1:
        act_mode = 3
        ew_table = get_sdcdnn_nexp_table()

    loading_asm = '''
    rset r0
    addi lock, r0, 3  // lock ew
    lock r0, lock, r0, 0
    core_id r1
    bne r1, r0, end
    addi r1, r0, {act_mode}// tanh+sigmoid mode
    addi r2, r0, 1 // interval 1.0 / 64
    ew_cfg r1, r2, r0, 9 // activ_mode
    ori r4, r0, {tabk_addr_hi}
    slli r1, r4, 16
    ori r1, r1, {tabk_addr_lo}
    ewtable_cfg r0, r1, r0, 0 // load k table
    ori r4, r0, {tabb_addr_hi}
    slli r1, r4, 16
    ori r1, r1, {tabb_addr_lo}
    ewtable_cfg r0, r1, r0, 1 // load b table
    unlock r0, lock, r0, 0
    end:
    ret
    '''.format(tabk_addr_hi=(tabk_addr >> 16), tabk_addr_lo=(tabk_addr & 0xFFFF),
            tabb_addr_hi=(tabb_addr >> 16), tabb_addr_lo=(tabb_addr & 0xFFFF),
            act_mode=act_mode)

    for bank in range(0, 16):
        sim.set_sram_data_bybank(cluster_id, sim.L1_SRAM_E, bank, ew_tab_offset, ew_table.tobytes())
    sim.set_asm_source(code=loading_asm)
    sim.run()
    return


def reset_sram(sim, cluster_id = 0):
    assert isinstance(sim, CDNNSimulator)
    for sram_id in [sim.L2_SRAM_D, sim.L2_SRAM_W,
                    sim.L1_SRAM_D, sim.L1_SRAM_W,
                    sim.L1_SRAM_E, sim.L2_SRAM_E,
                    sim.L2_SRAM_R]:
        sram = sim.get_sram(cluster_id, sram_id)
        sram.set_data(0, '\x00' * sram.size)


def bf16_to_fp32(bf16):
    import ctypes as ct
    z = bf16 << 16
    return ct.cast(ct.pointer(ct.c_uint32(z)), ct.POINTER(ct.c_float)).contents.value

def fp32_to_bf16(f):
    import ctypes as ct
    if math.isnan(f):
        return ct.c_uint16(0x7FC0).value
    z = ct.cast(ct.pointer(ct.c_float(f)), ct.POINTER(ct.c_uint32)).contents.value
    z = z >> 16
    return ct.c_uint16(z).value

def stat_diff(a, b):
    '''print diff status of two numpy array'''
    diff = abs(a - b)
    #print "diff:"
    #print diff
    max = diff.max()
    print "max diff = ", max
    idx = np.argmax(diff)
    idx = np.unravel_index(idx, diff.shape)
    print "idx of the max diff = ",idx
    print "value in the max diff place: "
    print a[idx], b[idx]

class InstructionTestCase(unittest.TestCase):
    def setUp(self):
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

    def test_ew_resize(self):
        src_addr = 0
        src_bank_id, src_offset = addr2bankoff(src_addr, bank_bits=4, align_bits=1)
        dst_addr = 0
        dst_bank_id, dst_offset = addr2bankoff(dst_addr, bank_bits=4, align_bits=1)
        resize_shamt = 0

        #resize params
        vld_core_num = 16
        bool_findmax = 1
        old_max = 0
        factor_w = 2
        factor_h = 2

        factor_w_u32 = ctypes.cast(ctypes.pointer(ctypes.c_float(factor_w)), ctypes.POINTER(ctypes.c_uint32))[0]
        factor_h_u32 = ctypes.cast(ctypes.pointer(ctypes.c_float(factor_h)), ctypes.POINTER(ctypes.c_uint32))[0]
        factor_w_u32_hi = factor_w_u32 >> 16
        factor_w_u32_lo = factor_w_u32 & 0xFFFF
        factor_h_u32_hi = factor_h_u32 >> 16
        factor_h_u32_lo = factor_h_u32 & 0xFFFF

        resize_output_w = 3
        resize_output_h = 3
        resize_src_w = 6
        resize_src_h = 6
        resize_output_offset_w = 0
        resize_output_offset_h = 0
        resize_input_offset_w = 0
        resize_input_offset_h = 0

        # l1E -> l2E
        print "test_ew_ssresize l1E -> l2E"
        sim = None
        sim = CDNNSimulator()
        reset_sram(sim)
        cluster_id = 0

        result_golden = np.zeros((vld_core_num, resize_output_w * resize_output_h), dtype=np.float32)
        result_caffe_golden = np.zeros((vld_core_num, resize_output_w * resize_output_h), dtype=np.float32)
        for i in range(0, vld_core_num):
            data = np.random.randint(0, 128, size=(resize_src_w * resize_src_h)).astype(np.float32)
            #data = np.arange(0, resize_src_w * resize_src_h).astype(np.float32)
            sim.set_sram_data_bybank(cluster_id, sim.L1_SRAM_E, i, src_offset, data.astype(np.float32).tobytes())
            resize_func(data, result_golden[i], factor_w, factor_h, resize_output_w, resize_output_h, resize_src_w, resize_src_h, resize_output_offset_w, resize_output_offset_h, resize_input_offset_w, resize_input_offset_h)
            caffe_resize(data, resize_src_h, resize_src_w, result_caffe_golden[i], resize_output_h, resize_output_w)
            #print "data is ", data
            #print "i is ",i
            #print "result_golden",result_golden[i]
            #print "result_caffe_golden",result_caffe_golden[i]

        asm = """
        rset r0
        rset r9 // null

        addi lock, r0, 3  // lock ew
        lock r0, lock, r0, 0

        addi r2, r0, {} // find_max
        addi r3, r0, {} // old_max
        ew_cfg r2, r3, r9, 0

        addi r1, r0, {} // vld_core_num
        ew_cfg r1, r9, r9, 1

        ori r4, r0, {}
        slli r2, r4, 16
        ori r2, r2, {}

        ori r5, r0, {}
        slli r3, r5, 16
        ori r3, r3, {}
        ew_cfg r2, r3, r9, 11

        addi r2, r0, {} // output_w
        addi r3, r0, {} // output_h
        ew_cfg r2, r3, r9, 12

        addi r2, r0, {} // input_w
        addi r3, r0, {} // input_h
        ew_cfg r2, r3, r9, 13

        addi r2, r0, {} // output_offset_w
        addi r3, r0, {} // output_offset_h
        ew_cfg r2, r3, r9, 14

        addi r2, r0, {} // input_offset_w
        addi r3, r0, {} // input_offset_h
        ew_cfg r2, r3, r9, 15

        addi r4, r0, {} // src_addr
        addi r5, r0, {} // dst_addr
        ssresize r5, r4, r9, {}

        rset r8
        addi r2, r0, 0 // offset 0
        ld_sd r8, r2, r9, 3 // load EW module data max into r8
        unlock r0, lock, r0, 0

        ret
        """.format(bool_findmax, old_max,
                    vld_core_num,
                    factor_w_u32_hi, factor_w_u32_lo,
                    factor_h_u32_hi, factor_h_u32_lo,
                    resize_output_w, resize_output_h,
                    resize_src_w, resize_src_h,
                    resize_output_offset_w, resize_output_offset_h,
                    resize_input_offset_w, resize_input_offset_h,
                    src_addr, dst_addr,
                    resize_shamt)
        #print asm
        sim.set_asm_source(code=asm)
        sim.run()

        result_len = resize_output_w * resize_output_h
        for i in range(0, vld_core_num):
            ans = sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_E, i,
                dst_offset, result_len * ctypes.sizeof(ctypes.c_float))
            y_ = np.frombuffer(ans, dtype=np.float32).reshape(result_len)
            #print "xpu result:          ",y_
            #print "python result:       ",result_golden[i]
            #print "python caffe result: ",result_caffe_golden[i]
            np.testing.assert_allclose(result_golden[i], y_, 1e-3, 1e-3)

        result_golden_datamax = abs(result_golden).max()
        result_sim_datamax = ctypes.cast(ctypes.pointer(ctypes.c_uint32(sim.get_register(0, 0, 'r8'))),
            ctypes.POINTER(ctypes.c_float))[0]
        #print "result_golden_datamax is",result_golden_datamax
        #print "result_sim_datamax is",result_sim_datamax
        np.testing.assert_allclose(result_golden_datamax, result_sim_datamax, 1e-3, 1e-3)

    def test_ew_pooling(self):
        src_addr = 0
        dst_data_addr = 0
        src_offset = 0
        dst_offset = 0
        saddr_hi = src_addr >> 16
        saddr_lo = src_addr & 0xFFFF
        d_data_addr_hi = dst_data_addr >> 16
        d_data_addr_lo = dst_data_addr & 0xFFFF

        input_w = 4
        input_h = 5
        dst_index_offset = 4 * input_w * input_h
        dst_index_addr = bankoff2addr(0, dst_index_offset, 4, 2)
        d_index_addr_hi = dst_index_addr >> 16
        d_index_addr_lo = dst_index_addr & 0xFFFF

        pad_left = 1
        pad_right = 1
        pad_top = 1
        pad_bottom = 1

        filter_w = 3
        filter_h = 4
        stride_w = 1
        stride_h = 1

        output_w = ((input_w + pad_left + pad_right - filter_w) / stride_w) + 1
        output_h = ((input_h + pad_top + pad_bottom - filter_h) / stride_h) + 1

        vld_core_num = 16

        ave_with_pad = 0
        ave_without_pad = 1
        ave_pooling_type = ave_without_pad

        max_pooling = 0
        ave_pooling = 1
        pooling_type = ave_pooling

        enable_max_index = 1
        disable_max_index = 0
        max_index_enble = enable_max_index

        dst_l2e_sram_id = 0 #data:l2E index:l1E
        dst_l1e_sram_id = 1 #data:l1E index:l2E
        dst_sram_id = dst_l2e_sram_id

        sspooling_shamt = (ave_pooling_type << 4) | \
                            (pooling_type << 3) | \
                            (max_index_enble << 2) | \
                            (dst_sram_id)

        bypass = 0
        findmax = 1
        bool_findmax = findmax
        old_max = 0

        #debug
        '''
        print "sspooling_shamt is ", sspooling_shamt
        print "input_w=",input_w
        print "input_h=",input_h
        print "pad_left=",pad_left
        print "pad_right=",pad_right
        print "pad_top=",pad_top
        print "pad_bottom=",pad_bottom

        print "filter_w=",filter_w
        print "filter_h=",filter_h
        print "stride_w=",stride_w
        print "stride_h=",stride_h

        print "output_w=",output_w
        print "output_h=",output_h

        print "bool_findmax=",bool_findmax
        print "old_max=",old_max
        '''

        # l1E -> l2E
        print "test_ew_pooling l1E -> l2E"
        sim = None; sim = CDNNSimulator()
        reset_sram(sim)
        cluster_id = 0

        result_array_golden = np.zeros((EW_BANK_NUM, output_h * output_w), dtype=np.float32)
        result_index_array_golden = np.zeros((EW_BANK_NUM, output_h * output_w), dtype=np.float32)
        for i in range(0, EW_BANK_NUM):
            data = np.random.randint(0, 128, size=(input_w * input_h)).astype(np.float32)
            '''
            print data
            for k in range(0, input_h):
                for j in range(0, input_w):
                    print data[k * input_w + j],
                print ""
            '''
            sim.set_sram_data_bybank(cluster_id, sim.L1_SRAM_E, i, src_offset, data.astype(np.float32).tobytes())

            pooling_func(data, result_array_golden[i],
                    result_index_array_golden[i],
                    pooling_type, ave_pooling_type,
                    input_w, input_h,
                    pad_left, pad_right,
                    pad_top, pad_bottom,
                    filter_w, filter_h,
                    stride_w, stride_h,
                    output_w, output_h)
            '''
            print "after pooling process result_array_golden[i]:"
            for k in range(0, output_h):
                for j in range(0, output_w):
                    print result_array_golden[i][k * output_w + j],
                print ""

            print "after pooling process result_index_array_golden[i]:"
            for k in range(0, output_h):
                for j in range(0, output_w):
                    print result_index_array_golden[i][k * output_w + j],
                print ""
            '''
        asm = """
        rset r0
        rset r9 // null
        rset r7 // dst_index_addr
        addi lock, r0, 3  // lock ew
        lock r0, lock, r0, 0
        addi r1, r0, {} // vld_core_num
        ew_cfg r1, r9, r9, 1 // set vld_core_num

        addi r2, r0, {} // input_w
        addi r3, r0, {} // input_h
        ew_cfg r2, r3, r9, 4 // set input_w and input_h

        addi r2, r0, {} // pad_left
        addi r3, r0, {} // padding_right
        ew_cfg r2, r3, r9, 7 // set pad_left and pad_right

        addi r2, r0, {} // pad_top
        addi r3, r0, {} // pad_bottom
        ew_cfg r2, r3, r9, 8 // set pad_top and pad_bottom

        addi r2, r0, {} // filter_w
        addi r3, r0, {} // filter_h
        ew_cfg r2, r3, r9, 5 // set filter_w and filter_h

        addi r2, r0, {} // stride_w
        addi r3, r0, {} // stride_h
        ew_cfg r2, r3, r9, 6 // set stride_w and stride_h

        addi r2, r0, {} // output_w
        addi r3, r0, {} // output_h
        ew_cfg r2, r3, r9, 3 // set output_w and output_h

        addi r2, r0, {} // find_max
        addi r3, r0, {} // old_max
        ew_cfg r2, r3, r9, 0 // set find_max and old_max

        ori r4, r0, {}
        slli r2, r4, 16
        ori r2, r2, {}  // src_addr
        ori r4, r0, {}
        slli r3, r4, 16
        ori r3, r3, {}  // dst_addr dst_data_addr
        ori r4, r0, {}
        slli r7, r4, 16
        ori r7, r7, {}  // dst_addr dst_index_addr
        sspooling r3, r2, r7, {} // set src_addr and dst_addr

        rset r8
        addi r2, r0, 0 // offset 0
        ld_sd r8, r2, r9, 3 // load EW module data max into r8
        unlock r0, lock, r0, 0

        ret
        """.format(vld_core_num,
                    input_w, input_h,
                    pad_left, pad_right,
                    pad_top, pad_bottom,
                    filter_w, filter_h,
                    stride_w, stride_h,
                    output_w, output_h,
                    bool_findmax, old_max,
                    saddr_hi, saddr_lo,
                    d_data_addr_hi, d_data_addr_lo,
                    d_index_addr_hi, d_index_addr_lo,
                    sspooling_shamt)
        print asm
        sim.set_asm_source(code=asm)
        sim.run()

        result_len = output_h * output_w
        #for i in range(0, EW_BANK_NUM):
        for i in range(0, vld_core_num):
            ans = sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_E, i,
                dst_offset, result_len * ctypes.sizeof(ctypes.c_float))
            y_ = np.frombuffer(ans, dtype=np.float32).reshape(result_len)
            index_ans = sim.get_sram_data_bybank(cluster_id, sim.L1_SRAM_E, i,
                dst_index_offset, result_len * ctypes.sizeof(ctypes.c_int32))
            index_y_ = np.frombuffer(index_ans, dtype=np.int32).reshape(result_len)

            '''
            print "x_:",result_array_golden[i]
            print "y_:", y_
            for k in range(0, output_h):
                for j in range(0, output_w):
                    print y_[k * output_w + j],
                print ""
            print "index_x_:",result_index_array_golden[i]
            print "index_y:", index_y_
            for k in range(0, output_h):
                for j in range(0, output_w):
                    print index_y_[k * output_w + j],
                print ""
            '''

            np.testing.assert_allclose(result_array_golden[i], y_, 1e-3, 1e-3)
            np.testing.assert_allclose(result_index_array_golden[i], index_y_, 1e-3, 1e-3)

        result_array_golden = result_array_golden[0:vld_core_num]
        result_golden_datamax = abs(result_array_golden).max()
        result_sim_datamax = ctypes.cast(ctypes.pointer(ctypes.c_uint32(sim.get_register(0, 0, 'r8'))),
            ctypes.POINTER(ctypes.c_float))[0]
        #np.testing.assert_almost_equal(result_golden_datamax, result_sim_datamax, decimal=5)
        np.testing.assert_allclose(result_golden_datamax, result_sim_datamax, 1e-3, 1e-3)

    def test_rs_col(self):
        print "test_rs_col...."
        CORE0_RS_BANK_NUM = 16
        depth = 40
        cluster_id = 0
        sim = None; sim = CDNNSimulator()
        reset_sram(sim)
        data = np.arange(0, (depth * CORE0_RS_BANK_NUM)).astype(np.float32)
        data = data.reshape(depth, CORE0_RS_BANK_NUM)
        #print "data"
        #print data
        trans = data.T
        #print "trans"
        #print trans

        rs_src_addr = 0
        rs_dst_addr = 0
        for i in range(0, CORE0_RS_BANK_NUM):
            row = trans[i]
            sim.set_sram_data_bybank(cluster_id, sim.L2_SRAM_E, i, rs_src_addr, row.astype(np.float32).tobytes())

        saddr_hi = rs_src_addr >> 16
        saddr_lo = rs_src_addr & 0xFFFF
        daddr_hi = rs_dst_addr >> 16
        daddr_lo = rs_dst_addr & 0xFFFF
        dst_bank, dst_off = addr2bankoff(rs_dst_addr, bank_bits = 4, align_bits=2)
        src_bank, src_off = addr2bankoff(rs_src_addr, bank_bits = 4, align_bits=2)

        dst_col_stride = 2
        dst_row_stride = 2
        src_row_stride = 3
        length = 11
        loop = 5

        asm = '''
        rset r0
        addi lock, r0, 4  // rs ew
        lock r0, lock, r0, 0
        addi r1, r0, {}
        addi r2, r0, {}
        addi r3, r0, {}
        rs_cfg r1, r2, r3, 2
        '''.format(dst_col_stride, src_row_stride, dst_row_stride)
        asm += '''
        addi r7, r0, {}
        rs_cfg r1, r7, r3, 4
        '''.format(loop)
        asm += '''
        ori r8, r0, {}
        slli r4, r8, 16
        ori r4, r4, {}
        ori r8, r0, {}
        slli r5, r8, 16
        ori r5, r5, {}
        addi r6, r0, {}
        rs_col r6, r4, r5, 1
        unlock r0, lock, r0, 0
        ret
        '''.format(saddr_hi, saddr_lo, daddr_hi, daddr_lo, length)

        sim.set_asm_source(code=asm)
        sim.run()

        l2_sram_r_bank_size = 1024
        l2_sram_r_bank_num = 16

        golden = np.zeros(l2_sram_r_bank_num * l2_sram_r_bank_size).reshape(l2_sram_r_bank_size, l2_sram_r_bank_num)
        for i in range(loop):
            for j in range(length):
                golden[dst_off + j * dst_row_stride][(dst_bank + i * dst_col_stride) % 64] = data[src_off + i * src_row_stride][src_bank + j]

        result_depth = dst_off + length * dst_row_stride - 1
        #print "golden (only piror {} rows)".format(result_depth)
        #print golden[:result_depth]

        res_list = []
        for bank_id in range(0, l2_sram_r_bank_num):
            ans = sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_R, bank_id, 0, l2_sram_r_bank_size * ctypes.sizeof(ctypes.c_float))
            y = np.frombuffer(ans, dtype=np.float32)
            res_list.append(y)

        res = np.array(res_list).reshape(l2_sram_r_bank_num, l2_sram_r_bank_size).T
        #print res.shape
        #print res[:result_depth]

        np.testing.assert_array_equal(golden[: result_depth], res[: result_depth])

    def test_rs_row(self):
        # print "test_rs_row "
        # np.set_printoptions(threshold=np.nan)
        src_addr = 0
        CORE0_RS_BANK_NUM = 16
        depth = 40
        cluster_id = 0
        sim = None; sim = CDNNSimulator()
        reset_sram(sim)
        data = np.arange(0, (depth * CORE0_RS_BANK_NUM)).astype(np.float32)
        data = data.reshape(depth, CORE0_RS_BANK_NUM)
        # print data
        trans = data.T
        # print data
        for i in range(0, CORE0_RS_BANK_NUM):
            row = trans[i]
            # print i,
            # print row
            sim.set_sram_data_bybank(cluster_id, sim.L2_SRAM_E, i, src_addr, row.astype(np.float32).tobytes())

        dst_col_stride = 0
        dst_row_stride = 1
        src_row_stride = 1
        # rs_src_addr = 108288
        # rs_dst_addr = 192832
        rs_src_addr = 0
        rs_dst_addr = 0
        dst_bank, dst_off = addr2bankoff(rs_dst_addr, bank_bits=4, align_bits=2)
        src_bank, src_off = addr2bankoff(rs_src_addr, bank_bits=4, align_bits=2)
        saddr_hi = rs_src_addr >> 16
        saddr_lo = rs_src_addr & 0xFFFF
        daddr_hi = rs_dst_addr >> 16
        daddr_lo = rs_dst_addr & 0xFFFF
        length = 11
        loop = 4
        asm = '''
        rset r0
        addi lock, r0, 4  // lock rs
        lock r0, lock, r0, 0
        addi r1, r0, {}
        addi r2, r0, {}
        addi r3, r0, {}
        rs_cfg r1, r2, r3, 1
        '''.format(dst_col_stride, src_row_stride, dst_row_stride)
        asm += '''
        addi r7, r0, {}
        rs_cfg r1, r7, r3, 3
        '''.format(loop)
        asm += '''
        ori r4, r0, {}
        slli r1, r4, 16
        ori r1, r1, {}
        ori r4, r0, {}
        slli r2, r4, 16
        ori r2, r2, {}
        addi r6, r0, {}
        rs_row r6, r1, r2, 16
        unlock r0, lock, r0, 0
        ret
        '''.format(saddr_hi, saddr_lo, daddr_hi, daddr_lo, length)

        sim.set_asm_source(code=asm)
        sim.run()

        l2_sram_r_bank_size = 1024
        l2_sram_r_bank_num = 16

        golden = np.zeros(l2_sram_r_bank_num * l2_sram_r_bank_size).reshape(l2_sram_r_bank_size, l2_sram_r_bank_num)
        for i in range(loop):
            for j in range(length):
                golden[dst_off + i * dst_row_stride][(dst_bank + j + i * dst_col_stride) % 64] = data[src_off + i * src_row_stride][src_bank + j]

        # print "golden (only piror {} rows)".format(depth)
        # print golden[:depth - 1]

        res_list = []
        for bank_id in range(0, l2_sram_r_bank_num):
            #ans = sim.get_sram_data_bybank(cluster_id,
                    #sim.L2_SRAM_R, bank_id, 0, l2_sram_r_bank_size * ctypes.sizeof(ctypes.c_float))
            #y = np.frombuffer(ans, dtype=np.float32).reshape(depth)
            ans = sim.get_sram_data_bybank(cluster_id,
                    sim.L2_SRAM_R, bank_id, 0, l2_sram_r_bank_size * ctypes.sizeof(ctypes.c_float))
            #print bank_id,
            y = np.frombuffer(ans, dtype=np.float32)
            # print y
            res_list.append(y)

        cc_vec = tuple(res_list)
        res = np.concatenate(cc_vec).reshape(l2_sram_r_bank_num, l2_sram_r_bank_size).T
        # print "result (only piror {} rows)".format(depth)
        # print res.shape
        # print res[:depth - 1]

        np.testing.assert_array_equal(golden[: depth], res[: depth])

    def test_lock(self):
        asm = '''
        rset r0
        addi r1, r0, 2
        lock r0, r1, r0, 0
        unlock r0, r1, r0, 0
        addi r1, r0, 3
        lock r0, r1, r0, 0
        unlock r0, r1, r0, 0
        ret
        '''
        sim = None; sim = CDNNSimulator()
        sim.set_asm_source(code=asm)
        sim.run()

    def test_win2vec_asymmetric_pad_wb(self):
        channel = 3
        data_type = 1 # 0:int8 1:int16
        win_x = 3
        win_y = 3
        blk_x = 40
        blk_y = 7
        np_dt = np.int16
        ct_dt = ctypes.c_int16
        if data_type == 0:
            np_dt = np.int8
            ct_dt = ctypes.c_int8
        block_distance = blk_x * blk_y * ctypes.sizeof(ct_dt)

        dilation = 1
        stride = 2
        start_addr = 0
        dst_addr = channel * block_distance
        dst_addr = 10000
        win2vec_len = 30
        pad_left = 1
        pad_right = 0
        pad_top = 1
        pad_bottom = 0
        l2_wb_stride = channel * win_x * win_y
        asm = '''
        rset r0
        addi lock, r0, 0  // lock ew
        lock r0, lock, r0, 0
        addi r1, r0, {}
        ds_cfg r1, r0, r2, 2 // conv_datatype, rs, rt is not necessary in this case
        addi r1, r0, {}
        ds_cfg r1, r0, r2, 3 // conv_channel, rs, rt is not necessary in this case
        addi r1, r0, {} // win_x
        addi r2, r0, {} // win_y
        ds_cfg r1, r2, r0, 4 // conv_win_size, rt is not necessary in this case
        addi r1, r0, {} // blk_x
        addi r2, r0, {} // blk_y
        ds_cfg r1, r2, r0, 5 // conv_block_size, rt is not necessary in this case
        addi r1, r0, {} // blok_distance
        ds_cfg r1, r2, r0, 6 // conv_block_distance, rs, rt is not necessary in this case
        addi r1, r0, {} // dilation
        ds_cfg r1, r2, r0, 7 // conv_dilation, rs, rt is not necessary in this case
        addi r1, r0, {} // stride
        ds_cfg r1, r2, r0, 8 // conv_stride, rs, rt is not necessary in this case
        addi r1, r0, {} // pad_left
        addi r2, r0, {} // pad_right
        ds_cfg r1, r2, r0, 9 // conv_pad_horizontal
        addi r1, r0, {} // pad_top
        addi r2, r0, {} // pad_bottom
        ds_cfg r1, r2, r0, 10 // conv_pad_vertical
        addi r3, r0, {} // l2_wb_stride
        ds_cfg r3, r2, r1, 12 // l2_wb_stride
        addi r1, r0, {} // start_addr
        addi r2, r0, {} // win2vec_len
        addi r3, r0, {} // dst_addr
        win2vec r1, r2, r3, 8
        unlock r0, lock, r0, 0
        ret
        '''.format(data_type, channel, win_x, win_y, blk_x, blk_y, block_distance, dilation, stride,
                pad_left, pad_right, pad_top, pad_bottom, l2_wb_stride, start_addr, win2vec_len, dst_addr)

        image = (np.arange(channel * blk_y * blk_x) + 1).reshape((channel, blk_y, blk_x)).astype(np_dt)
        block = image[:, 0: win_y, 0: win2vec_len]
        expect = conv_win2vec_numpy(block, channel, win_y, win2vec_len, win_y, win_x, stride,
                pad_left, pad_right, pad_top, pad_bottom, np_dt)

        cluster_id = 0
        sim = None
        sim = CDNNSimulator()
        bank_id = 0
        sim.set_sram_data_bybank(cluster_id, sim.L2_SRAM_W, bank_id, 0, image.astype(np_dt).tobytes())
        sim.set_asm_source(code=asm)
        #print asm
        sim.run()

        outputh = (win_y + pad_top + pad_bottom - win_y) / stride + 1
        outputw = (win2vec_len + pad_left + pad_right - win_x) / stride + 1

        result = np.zeros((outputw, channel * win_x * win_y)).astype(np_dt)
        for i in range(outputw):
            ans = sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_W, bank_id, dst_addr + i * l2_wb_stride * ctypes.sizeof(ct_dt), \
                channel * win_x * win_y * ctypes.sizeof(ct_dt))
            result[i][:] = np.frombuffer(ans, dtype=np_dt).reshape(-1)

        np.testing.assert_array_equal(expect, result)

    def test_win2vec_asymmetric_pad(self):
        channel = 3
        data_type = 0 # int8(0) or int16(1)
        win_x = 3
        win_y = 3
        blk_x = 40
        blk_y = 7
        block_distance = blk_x * blk_y * 2
        np_dt = np.int16
        ct_dt = ctypes.c_int16
        if data_type == 0:
            np_dt = np.int8
            ct_dt = ctypes.c_int8
            block_distance = blk_x * blk_y

        dilation = 1
        stride = 2
        start_addr = 0
        dst_addr = 0
        win2vec_len = 30
        pad_left = 1
        pad_right = 0
        pad_top = 1
        pad_bottom = 0
        asm = '''
        rset r0
        addi lock, r0, 0  // lock ds-w
        lock r0, lock, r0, 0
        addi r1, r0, {}
        ds_cfg r1, r0, r2, 2 // conv_datatype, rs, rt is not necessary in this case
        addi r1, r0, {} 
        ds_cfg r1, r0, r2, 3 // conv_channel, rs, rt is not necessary in this case
        addi r1, r0, {} // win_x
        addi r2, r0, {} // win_y
        ds_cfg r1, r2, r0, 4 // conv_win_size, rt is not necessary in this case
        addi r1, r0, {} // blk_x
        addi r2, r0, {} // blk_y
        ds_cfg r1, r2, r0, 5 // conv_block_size, rt is not necessary in this case
        addi r1, r0, {} // blok_distance
        ds_cfg r1, r2, r0, 6 // conv_block_distance, rs, rt is not necessary in this case
        addi r1, r0, {} // dilation
        ds_cfg r1, r2, r0, 7 // conv_dilation, rs, rt is not necessary in this case
        addi r1, r0, {} // stride
        ds_cfg r1, r2, r0, 8 // conv_stride, rs, rt is not necessary in this case
        addi r1, r0, {} // pad_left
        addi r2, r0, {} // pad_right
        ds_cfg r1, r2, r0, 9 // conv_pad_horizontal
        addi r1, r0, {} // pad_top
        addi r2, r0, {} // pad_bottom
        ds_cfg r1, r2, r0, 10 // conv_pad_vertical
        addi r1, r0, {} // start_addr
        addi r2, r0, {} // win2vec_len
        addi r3, r0, {} // dst_addr
        addi r9, r0, 0
        win2vec r1, r2, r3, 2
        unlock r0, lock, r0, 0
        addi lock, r0, 1  // lock ds-d
        lock r0, lock, r0, 0
        win2vec r1, r2, r3, 1
        unlock r0, lock, r0, 0
        ret
        '''.format(data_type, channel, win_x, win_y, blk_x, blk_y, block_distance, dilation, stride,
                pad_left, pad_right, pad_top, pad_bottom, start_addr, win2vec_len, dst_addr)

        image = (np.arange(channel * blk_y * blk_x) + 1).reshape((channel, blk_y, blk_x)).astype(np_dt)
        block = image[:, 0: win_y, 0: win2vec_len]
        expect = conv_win2vec_numpy(block, channel, win_y, win2vec_len, win_y, win_x, stride,
                pad_left, pad_right, pad_top, pad_bottom, np_dt)

        cluster_id = 0
        sim = None; sim = CDNNSimulator()
        bank_id = 0
        # x = np.arange(1000).astype(np_dt)
        #sim.set_sram_data(sim.L2_SRAM_D, 0, x.astype(np_dt).tobytes())
        sim.set_sram_data_bybank(cluster_id, sim.L2_SRAM_D, bank_id, 0, image.astype(np_dt).tobytes())
        #sim.set_sram_data(sim.L2_SRAM_W, 0, x.astype(np_dt).tobytes())
        sim.set_sram_data_bybank(cluster_id, sim.L2_SRAM_W, bank_id, 0, image.astype(np_dt).tobytes())
        sim.set_asm_source(code=asm)
        sim.run()
        # print "win2vec reslut of L1_SRAM_D"
        # for i in range(0, expect.shape[0]):
        #     print "bank " + str(i)
        #     ans = sim.get_sram_data_bybank(sim.L1_SRAM_D, i, 0, 3 * (channel * win_x * win_y) *
        #             ctypes.sizeof(ct_dt))
        #     y = np.frombuffer(ans, dtype=np_dt)
        #     print y
        # np.testing.assert_array_equal(expect[i], y)
        # print "win2vec reslut of L1_SRAM_W"
        # for i in range(0, expect.shape[0]):
        #     print "bank " + str(i)
        #     ans = sim.get_sram_data_bybank(sim.L1_SRAM_W, i, 0, (channel * win_x * win_y) *
        #             ctypes.sizeof(ct_dt))
        #     y = np.frombuffer(ans, dtype=np_dt)
        #     print y
        #     np.testing.assert_array_equal(expect[i], y)

        # extract real useful data
        roundup_dim = (channel * win_y * win_y + 15) / 16 * 16
        outputh = (win_y + pad_top + pad_bottom - win_y) / stride + 1
        outputw = (win2vec_len + pad_left + pad_right - win_x) / stride + 1
        active_column_list = []
        for i in range(outputh):
            j = 0
            while(j < outputw):
                active_column_list = active_column_list + [min(32, outputw - j)]
                j = j + 32
        actual = extract_result_from_L1(sim, cluster_id, sim.L1_SRAM_D, channel * win_y * win_x, active_column_list, dst_addr, np_dt)
        np.testing.assert_array_equal(expect, actual)
        actual = extract_result_from_L1(sim, cluster_id, sim.L1_SRAM_W, channel * win_y * win_x, active_column_list, dst_addr, np_dt)
        np.testing.assert_array_equal(expect, actual)

    def test_shuffle_int8(self):
        src_addr = 0
        src_bank_id, src_offset = addr2bankoff(src_addr, bank_bits = 1, align_bits=1)

        dst_addr = 96
        dst_bank_id, dst_offset = addr2bankoff_8bit(dst_addr, bank_bits = 4, align_bits=1)
        dst_offset_base = addr2bankoffbase_8bit(dst_addr, bank_bits = 4, align_bits=1)
        dst_local_offset = (dst_addr >> (4 + 1)) & 0xf
        bank_data_pos = dst_addr & 0x1

        # int8
        data = np.random.randint(0, 128, size=(35)).astype(np.int8)
        src_len = len(data)

        cluster_id = 0
        # int8 l2sramw -> l1sramw
        print "test_shuffle_int8 l2sramw -> l1sramw"
        sim = None; sim = CDNNSimulator()
        reset_sram(sim)
        sim.set_sram_data_bybank(cluster_id, sim.L2_SRAM_W, src_bank_id, src_offset, data.astype(np.int8).tobytes())
        asm = """
        rset r0
        addi lock, r0, 0  // lock ds-w
        lock r0, lock, r0, 0
        addi r4, r0, 0 // int8
        addi r1, r0, {}
        addi r2, r0, {}
        addi r0, r0, {}
        ds_cfg r4, r0, r1, 0 // cfg shuffle data type
        rset r9
        shuffle r0, r1, r2, 2
        unlock r0, lock, r0, 0
        ret
        """.format(src_addr, dst_addr, len(data))
        print asm
        sim.set_asm_source(code=asm)
        sim.run()

        result_len = align(len(data) + dst_local_offset, 16) * 2
        ans = sim.get_sram_data_bybank(cluster_id, sim.L1_SRAM_W, dst_bank_id, dst_offset_base, result_len * ctypes.sizeof(ctypes.c_int8))
        result = [0] * result_len
        y_ = np.frombuffer(ans, dtype=np.int8).reshape(len(result))

        result_golden = [0] * result_len
        result_golden_index = 0
        src_data_index = 0
        used_len = 0

        if bank_data_pos == 1:
            result_golden_index = result_golden_index + 16
        used_len = min(src_len, 16 - dst_local_offset)
        for i in range(0, used_len):
            result_golden[result_golden_index + dst_local_offset + i] = data[src_data_index + i]
        result_golden_index += 32
        src_data_index += used_len
        while used_len < src_len:
            temp_len = min(16, src_len - used_len)
            for j in range(0, temp_len):
                result_golden[result_golden_index + j] = data[src_data_index + j]
            used_len += temp_len
            result_golden_index += 32
            src_data_index += temp_len

        #debug
        '''
        print "data:",data
        print "src_bank_id is ", src_bank_id
        print "src_offset is ", src_offset
        print "bank_data_pos is ", bank_data_pos
        print "dst_bank_id is ", dst_bank_id
        print "dst_offset_base is ",dst_offset_base
        print "dst_offset is ", dst_offset
        print "dst_local_offset is ", dst_local_offset
        print "result_len is ",result_len
        print "xpu result:", y_
        print "python result:", result_golden
        '''

        np.testing.assert_array_equal(result_golden, y_)

    def test_shuffle_int16(self):
        src_addr = 0
        dst_addr = 0
        bank_id = 0

        # int16
        data = np.random.randint(0, 128, size=(64)).astype(np.int16)

        cluster_id = 0
        # int16 l2sramw -> l1sramw
        print "test_shuffle_int16 l2sramw -> l1sramw"
        sim = None; sim = CDNNSimulator()
        sim.set_sram_data_bybank(cluster_id, sim.L2_SRAM_W, bank_id, src_addr, data.astype(np.int16).tobytes())
        asm = """
        rset r0
        addi lock, r0, 0  // lock ds-w
        lock r0, lock, r0, 0
        addi r4, r0, 1 // int16
        addi r1, r0, {}
        addi r2, r0, {}
        addi r0, r0, {}
        ds_cfg r4, r0, r1, 0 // cfg shuffle data type
        rset r9
        shuffle r0, r1, r2, 2
        unlock r0, lock, r0, 0
        ret
        """.format(src_addr, dst_addr, len(data))
        print asm
        sim.set_asm_source(code=asm)
        sim.run()
        ans = sim.get_sram_data(cluster_id, sim.L1_SRAM_W, dst_addr, len(data) * ctypes.sizeof(ctypes.c_int16))
        y_ = np.frombuffer(ans, dtype=np.int16).reshape(data.shape)
        np.testing.assert_array_equal(data, y_)

        # int16 l2sramd -> l1sramd
        print "int16 l2sramd -> l1sramd"
        sim = None; sim = CDNNSimulator()
        sim.set_sram_data_bybank(cluster_id, sim.L2_SRAM_D, bank_id, src_addr, data.astype(np.int16).tobytes())
        asm = """
        rset r0
        addi lock, r0, 1  // lock ds-d
        lock r0, lock, r0, 0
        addi r4, r0, 1 // int16
        addi r1, r0, {}
        addi r2, r0, {}
        addi r0, r0, {}
        ds_cfg r4, r0, r1, 0 // cfg shuffle data type, rs, rt is not necessary in this case
        rset r9
        shuffle r0, r1, r2, 1
        unlock r0, lock, r0, 0
        ret
        """.format(src_addr, dst_addr, len(data))
        print asm
        sim.set_asm_source(code=asm)
        sim.run()
        ans = sim.get_sram_data(cluster_id, sim.L1_SRAM_D, dst_addr, len(data) * ctypes.sizeof(ctypes.c_int16))
        y_ = np.frombuffer(ans, dtype=np.int16).reshape(data.shape)
        np.testing.assert_array_equal(data, y_)

    def test_shuffle_float32(self):
        src_addr = 0
        dst_addr = 0
        bank_id = 0

        # float32
        data = np.random.randint(0, 128, size=(64)).astype(np.float32)

        cluster_id = 0
        # float32 l2sramw -> l1srame
        print "test_shuffle_float32 l2sramw -> l1srame"
        sim = None; sim = CDNNSimulator()
        sim.set_sram_data_bybank(cluster_id, sim.L2_SRAM_W, bank_id, src_addr, data.astype(np.float32).tobytes())
        asm = """
        rset r0
        addi lock, r0, 0  // lock ds-w
        lock r0, lock, r0, 0
        addi r4, r0, 2 //float32
        addi r1, r0, {}
        addi r2, r0, {}
        addi r0, r0, {}
        ds_cfg r4, r0, r1, 0 // cfg shuffle data type
        rset r9
        shuffle r0, r1, r2, 4
        unlock r0, lock, r0, 0
        ret
        """.format(src_addr, dst_addr, len(data))
        print asm
        sim.set_asm_source(code=asm)
        sim.run()
        ans = sim.get_sram_data(cluster_id, sim.L1_SRAM_E, dst_addr, len(data) * ctypes.sizeof(ctypes.c_float))
        y_ = np.frombuffer(ans, dtype=np.float32).reshape(data.shape)
        np.testing.assert_array_equal(data, y_)

        # float32 l2sramd -> l1srame
        print "test_shuffle_float l2sramd -> l1srame"
        sim = None; sim = CDNNSimulator()
        sim.set_sram_data_bybank(cluster_id, sim.L2_SRAM_D, bank_id, src_addr, data.astype(np.float32).tobytes())
        asm = """
        rset r0
        addi lock, r0, 1  // lock ds-w
        lock r0, lock, r0, 0
        addi r4, r0, 2 //float32
        addi r1, r0, {}
        addi r2, r0, {}
        addi r0, r0, {}
        ds_cfg r4, r0, r1, 0 // cfg shuffle data type, rs, rt is not necessary in this case
        rset r9
        shuffle r0, r1, r2, 4
        unlock r0, lock, r0, 0
        ret
        """.format(src_addr, dst_addr, len(data))
        print asm
        sim.set_asm_source(code=asm)
        sim.run()
        ans = sim.get_sram_data(cluster_id, sim.L1_SRAM_E, dst_addr, len(data) * ctypes.sizeof(ctypes.c_float))
        y_ = np.frombuffer(ans, dtype=np.float32).reshape(data.shape)
        np.testing.assert_array_equal(data, y_)

    def test_shuffle_batch_int8(self):
        src_addr = 0
        dst_addr = 0
        src_bank_id, src_offset = addr2bankoff(src_addr, bank_bits = 1, align_bits=1)
        dst_bank_id, dst_offset = addr2bankoff_8bit(dst_addr, bank_bits = 4, align_bits=1)
        dst_offset_base = addr2bankoffbase_8bit(dst_addr, bank_bits = 4, align_bits=1)
        dst_local_offset = (dst_addr >> (4 + 1)) & 0xf

        len = 60
        shuffle_batch_blockx = 64
        output_bank_num = 6

        bank_data_pos = dst_addr & 0x1

        # int8
        data = np.random.randint(0, 128, size=(shuffle_batch_blockx * 16)).astype(np.int8)
        #print data

        cluster_id = 0
        # int8 l2sramw -> l1sramw
        print "test_shuffle_batch_int8 l2sramw -> l1sramw"
        sim = None; sim = CDNNSimulator()
        reset_sram(sim)
        sim.set_sram_data_bybank(cluster_id, sim.L2_SRAM_W, src_bank_id, src_addr, data.astype(np.int8).tobytes())
        asm = """
        rset r0
        addi lock, r0, 0  // lock ds-w
        lock r0, lock, r0, 0
        addi r4, r0, 0 // int8
        addi r5, r0, {}
        addi r6, r0, {}
        addi r1, r0, {}
        addi r2, r0, {}
        addi r0, r0, {}
        ds_cfg r4, r0, r1, 0 // cfg shuffle data type
        ds_cfg r5, r0, r1, 1 // cfg shuffle shuffle_batch_blockx
        ds_cfg r6, r0, r1, 11 // cfg shuffle output bank num
        rset r9
        shuffle_batch r0, r1, r2, 2
        unlock r0, lock, r0, 0
        ret
        """.format(shuffle_batch_blockx, output_bank_num, src_addr, dst_addr, len)
        print asm
        sim.set_asm_source(code=asm)
        sim.run()

        result_len = align(len + dst_local_offset, 16) * 2
        for bank_idx in range(output_bank_num):
            ans = sim.get_sram_data_bybank(cluster_id, sim.L1_SRAM_W, bank_idx, dst_offset_base, result_len * ctypes.sizeof(ctypes.c_int8))
            y_ = np.frombuffer(ans, dtype=np.int8).reshape(-1)
            result_golden = [0] * result_len
            result_golden_index = 0
            src_data_index = 0
            used_len = 0
            if bank_data_pos == 1:
                result_golden_index = result_golden_index + 16
            used_len = min(len, 16 - dst_local_offset)
            for i in range(0, used_len):
                result_golden[result_golden_index + dst_local_offset + i] = data[bank_idx * shuffle_batch_blockx + src_data_index + i]
            result_golden_index += 32
            src_data_index += used_len
            while used_len < len:
                temp_len = min(16, len - used_len)
                for j in range(0, temp_len):
                    result_golden[result_golden_index + j] = data[bank_idx * shuffle_batch_blockx + src_data_index + j]
                used_len += temp_len
                result_golden_index += 32
                src_data_index += temp_len
            #print "xpu_result:", y_
            #print "result_golden:", result_golden
            np.testing.assert_array_equal(result_golden, y_)

    def test_shuffle_batch_int16(self):
        src_addr = 0
        dst_addr = 0
        bank_id = 0
        offset = 0
        len = 60
        shuffle_batch_blockx = 64
        output_bank_num = 6

        # int16
        data = np.random.randint(0, 128, size=(shuffle_batch_blockx * 16)).astype(np.int16)

        cluster_id = 0
        # int16 l2sramw -> l1sramw
        print "test_shuffle_batch_int16 l2sramw -> l1sramw"
        sim = None; sim = CDNNSimulator()
        sim.set_sram_data_bybank(cluster_id, sim.L2_SRAM_W, bank_id, src_addr, data.astype(np.int16).tobytes())
        asm = """
        rset r0
        addi lock, r0, 0  // lock ds-w
        lock r0, lock, r0, 0
        addi r4, r0, 1 // int16
        addi r5, r0, {} // shuffle_batch_blockx
        addi r6, r0, {} // shuffle output bank num
        addi r1, r0, {}
        addi r2, r0, {}
        addi r0, r0, {}
        ds_cfg r4, r0, r1, 0 // cfg shuffle data type
        ds_cfg r5, r0, r1, 1 // cfg shuffle shuffle_batch_blockx
        ds_cfg r6, r0, r1, 11 // cfg huffle output bank num
        rset r9
        shuffle_batch r0, r1, r2, 2
        unlock r0, lock, r0, 0
        ret
        """.format(shuffle_batch_blockx, output_bank_num, src_addr, dst_addr, len)
        print asm
        sim.set_asm_source(code=asm)
        sim.run()
        for bank_index in range(output_bank_num):
            ans = sim.get_sram_data_bybank(cluster_id, sim.L1_SRAM_W, bank_index, offset, len * ctypes.sizeof(ctypes.c_int16))
            y_ = np.frombuffer(ans, dtype=np.int16).reshape(-1)
            np.testing.assert_array_equal(data[bank_index * shuffle_batch_blockx : bank_index * shuffle_batch_blockx + len], y_)

    def test_shuffle_batch_float32(self):
        src_addr = 0
        dst_addr = 0
        bank_id = 0
        offset = 0
        len = 60
        shuffle_batch_blockx = 64
        output_bank_num = 6

        # float32
        data = np.random.randint(0, 128, size=(shuffle_batch_blockx * 16)).astype(np.float32)

        cluster_id = 0
        # float32 l2sramw -> l1sramw
        print "test_shuffle_batch_float32 l2sramw -> l1sramw"
        sim = None; sim = CDNNSimulator()
        sim.set_sram_data_bybank(cluster_id, sim.L2_SRAM_W, bank_id, src_addr, data.astype(np.float32).tobytes())
        asm = """
        rset r0
        addi lock, r0, 0  // lock ds-w
        lock r0, lock, r0, 0
        addi r4, r0, 2 // float32
        addi r5, r0, {} // shuffle_batch_blockx
        addi r6, r0, {} // shuffle output bank num
        addi r1, r0, {}
        addi r2, r0, {}
        addi r0, r0, {}
        ds_cfg r4, r0, r1, 0 // cfg shuffle data type
        ds_cfg r5, r0, r1, 1 // cfg shuffle shuffle_batch_blockx
        ds_cfg r6, r0, r1, 11 // cfg huffle output bank num
        rset r9
        shuffle_batch r0, r1, r2, 4
        unlock r0, lock, r0, 0
        ret
        """.format(shuffle_batch_blockx, output_bank_num, src_addr, dst_addr, len)
        print asm
        sim.set_asm_source(code=asm)
        sim.run()
        for bank_index in range(output_bank_num):
            ans = sim.get_sram_data_bybank(cluster_id, sim.L1_SRAM_E, bank_index, offset, len * ctypes.sizeof(ctypes.c_float))
            y_ = np.frombuffer(ans, dtype=np.float32).reshape(-1)
            np.testing.assert_array_equal(data[bank_index * shuffle_batch_blockx : bank_index * shuffle_batch_blockx + len], y_)

    def test_shuffle_batch_wb(self):
        data_type = 1 #0:int8 1:int16 2:fp32
        np_dt = np.int16
        ct_dt = ctypes.c_int16
        if data_type == 0:
            np_dt = np.int8
            ct_dt = ctypes.c_int8
        if data_type == 2:
            np_dt = np.float32
            ct_dt = ctypes.c_float

        src_addr = 0
        bank_id = 0
        len = 20
        shuffle_batch_blockx = 24
        output_bank_num = 16
        dst_addr = output_bank_num * shuffle_batch_blockx * ctypes.sizeof(ct_dt)
        l2_wb_stride = shuffle_batch_blockx + 10

        data = np.random.randint(0, 128, size=(shuffle_batch_blockx * output_bank_num)).astype(np_dt)
        #print data

        cluster_id = 0
        sim = None
        sim = CDNNSimulator()
        sim.set_sram_data_bybank(cluster_id, sim.L2_SRAM_W, bank_id, src_addr, data.astype(np_dt).tobytes())
        asm = """
        rset r0
        addi lock, r0, 0  // lock ds-w
        lock r0, lock, r0, 0
        addi r4, r0, {} // data type
        addi r5, r0, {} // shuffle_batch_blockx
        addi r6, r0, {} // shuffle output bank num
        addi r7, r0, {} // shuffle l2 write back stride
        addi r1, r0, {}
        addi r2, r0, {}
        addi r3, r0, {}
        ds_cfg r4, r0, r1, 0 // cfg shuffle data type
        ds_cfg r5, r0, r1, 1 // cfg shuffle shuffle_batch_blockx
        ds_cfg r6, r0, r1, 11 // cfg shuffle output bank num
        ds_cfg r7, r0, r1, 12 // cfg shuffle l2 write back stride
        shuffle_batch r3, r1, r2, 8
        unlock r0, lock, r0, 0
        ret
        """.format(data_type, shuffle_batch_blockx, output_bank_num, l2_wb_stride, src_addr, dst_addr, len)
        #print asm
        sim.set_asm_source(code=asm)
        sim.run()
        for bank_index in range(output_bank_num):
            ans = sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_W, bank_id, dst_addr + bank_index * l2_wb_stride * ctypes.sizeof(ct_dt), \
                len * ctypes.sizeof(ct_dt))
            y_ = np.frombuffer(ans, dtype=np_dt).reshape(-1)
            #print "bank_index is",bank_index
            #print "xpu result is"
            #print y_
            #print "python result is"
            #print data[bank_index * shuffle_batch_blockx : bank_index * shuffle_batch_blockx + len]
            np.testing.assert_array_equal(data[bank_index * shuffle_batch_blockx: bank_index * shuffle_batch_blockx + len], y_)

    def test_shuffle_wb(self):
        data_type = 1 #0:int8 1:int16 2:fp32
        np_dt = np.int16
        ct_dt = ctypes.c_int16
        if data_type == 0:
            np_dt = np.int8
            ct_dt = ctypes.c_int8
        if data_type == 2:
            np_dt = np.float32
            ct_dt = ctypes.c_float

        cluster_id = 0
        src_addr = 0
        bank_idx = 0
        len = 200
        dst_addr = src_addr + len * ctypes.sizeof(ct_dt) + 10
        data = np.arange(len).astype(np_dt)

        sim = None
        sim = CDNNSimulator()
        reset_sram(sim)

        sim.set_sram_data_bybank(cluster_id, sim.L2_SRAM_D, bank_idx, src_addr, data.astype(np_dt).tobytes())

        asm = """
        rset r0
        addi lock, r0, 1  // lock ds-d
        lock r0, lock, r0, 0
        addi r1, r0, {} // data type
        ds_cfg r1, r0, r2, 0 // cfg shuffle data type, rs, rt is not necessary in this case
        addi r2, r0, {}
        addi r3, r0, {}
        addi r4, r0, {}
        shuffle r2, r3, r4, 8
        unlock r0, lock, r0, 0
        ret
        """.format(data_type, len, src_addr, dst_addr)
        #print asm
        sim.set_asm_source(code=asm)
        sim.run()

        output_size = len
        ans = sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_D, bank_idx, dst_addr, output_size * ctypes.sizeof(ct_dt))
        result_xpu = np.frombuffer(ans, dtype=np_dt).reshape(output_size)
        #print "xpu result is: "
        #print result_xpu
        #print "data is"
        #print data
        np.testing.assert_array_equal(data, result_xpu)

    def test_shuffle_coa_wb(self):
        data_type = 1 #0:int8 1:int16 2:fp32
        np_dt = np.int16
        ct_dt = ctypes.c_int16
        if data_type == 0:
            np_dt = np.int8
            ct_dt = ctypes.c_int8
        if data_type == 2:
            np_dt = np.float32
            ct_dt = ctypes.c_float

        cluster_id = 0
        src_addr = 0
        bank_idx = 0
        len = 20
        shuffle_coa_output_bank_num = 16
        shuffle_coa_blockx = 28
        l2_wb_stride = 20
        dst_addr = src_addr + len * shuffle_coa_blockx * ctypes.sizeof(ct_dt) + 10
        assert(shuffle_coa_blockx >= shuffle_coa_output_bank_num)
        assert(dst_addr > len * shuffle_coa_blockx)
        data = np.arange(len * shuffle_coa_blockx).astype(np_dt)

        sim = None; sim = CDNNSimulator()
        reset_sram(sim)

        sim.set_sram_data_bybank(cluster_id, sim.L2_SRAM_D, bank_idx, src_addr, data.astype(np_dt).tobytes())

        asm = """
        rset r0
        addi lock, r0, 1  // lock ds-d
        lock r0, lock, r0, 0
        addi r1, r0, {} // data type
        ds_cfg r1, r0, r2, 0 // cfg shuffle data type, rs, rt is not necessary in this case
        addi r1, r0, {}
        ds_cfg r1, r0, r2, 1 // cfg shuffle coa blockx, rs, rt is not necessary in this case
        addi r1, r0, {}
        ds_cfg r1, r0, r2, 11 // cfg shuffle_coa_output_bank_num
        addi r1, r0, {}
        ds_cfg r1, r0, r2, 12 // cfg l2_wb_stride
        addi r2, r0, {}
        addi r3, r0, {}
        addi r4, r0, {}
        shuffle_coa r2, r3, r4, 8
        unlock r0, lock, r0, 0
        ret
        """.format(data_type, shuffle_coa_blockx, shuffle_coa_output_bank_num, l2_wb_stride, len, src_addr, dst_addr)
        #print asm
        sim.set_asm_source(code=asm)
        sim.run()

        output_size = shuffle_coa_output_bank_num * l2_wb_stride
        ans = sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_D, bank_idx, dst_addr, output_size * ctypes.sizeof(ct_dt))

        result_xpu = np.frombuffer(ans, dtype=np_dt).reshape(output_size)
        result_golden = np.zeros(output_size, dtype=np_dt)
        for i in range(0, len):
            src_index_base = i * shuffle_coa_blockx
            dst_index_base = i
            for j in range(0, shuffle_coa_output_bank_num):
                result_golden[dst_index_base + j * l2_wb_stride] = data[src_index_base + j]
        #print "xpu result is: "
        #print result_xpu
        #print "result_golden is"
        #print result_golden
        np.testing.assert_array_equal(result_golden, result_xpu)

    def test_shuffle_coa_int8(self):
        src_addr = 0
        dst_addr = 96
        src_bank_id, src_offset = addr2bankoff(src_addr, bank_bits = 1, align_bits=1)
        dst_bank_id, dst_offset = addr2bankoff_8bit(dst_addr, bank_bits = 4, align_bits=1)
        dst_offset_base = addr2bankoffbase_8bit(dst_addr, bank_bits = 4, align_bits=1)
        dst_local_offset = (dst_addr >> (4 + 1)) & 0xf

        data_src_row = 35
        data_src_col = 8

        focus_data_src_row = 33
        focus_data_src_col = 5

        dst_data_col = focus_data_src_row
        dst_data_row = focus_data_src_col
        shuffle_coa_output_bank_num = focus_data_src_col
        length = focus_data_src_row

        blkx_addr_offset = data_src_col

        # int8
        data = np.random.randint(0, 128, size=(data_src_row * data_src_col)).astype(np.int8).reshape(data_src_row, data_src_col)
        focus_data = np.zeros((focus_data_src_row, focus_data_src_col), dtype=np.int8).reshape(focus_data_src_row, focus_data_src_col)
        focus_data = data[0:focus_data_src_row, 0:focus_data_src_col]

        result_row_len = (align(dst_data_col + dst_local_offset, 16) * 2)
        result_matrix_golden = np.zeros((result_row_len, shuffle_coa_output_bank_num), dtype=np.int8)

        bank_data_pos = dst_addr & 0x1
        src_data_row_base_index = 0
        result_array_bank_base = 0
        if bank_data_pos == 1:
            result_array_bank_base += 16

        used_len = min(length, 16 - dst_local_offset)
        for i in range(0, used_len):
            result_array_bank_base_index = result_array_bank_base + dst_local_offset + i
            for j in range(0, shuffle_coa_output_bank_num):
                result_matrix_golden[result_array_bank_base_index][j] \
                    = data[src_data_row_base_index + i][j]
        result_array_bank_base += 32
        src_data_row_base_index += used_len
        while (used_len < length):
            temp_len = min(16, length - used_len)
            for i in range(0, temp_len):
                result_array_bank_base_index = result_array_bank_base + i
                for j in range(0, shuffle_coa_output_bank_num):
                    result_matrix_golden[result_array_bank_base_index][j] \
                        = data[src_data_row_base_index + i][j]
            result_array_bank_base += 32
            used_len += temp_len
            src_data_row_base_index += temp_len

        cluster_id = 0
        # int8 l2sramd -> l1sramd
        print "test_shuffle_coa int8 l2sramd -> l1sramd"
        sim = None; sim = CDNNSimulator()
        reset_sram(sim)
        sim.set_sram_data_bybank(cluster_id, sim.L2_SRAM_D, src_bank_id, src_addr, data.astype(np.int8).tobytes())
        asm = """
        rset r0
        addi lock, r0, 1  // lock ds-d
        lock r0, lock, r0, 0
        addi r1, r0, 0 // int8
        ds_cfg r1, r0, r2, 0 // cfg shuffle data type, rs, rt is not necessary in this case
        addi r1, r0, {}
        ds_cfg r1, r0, r2, 1 // cfg shuffle coa blockx, rs, rt is not necessary in this case
        addi r1, r0, {}
        ds_cfg r1, r0, r2, 11
        addi r2, r0, {}
        addi r3, r0, {}
        addi r4, r0, {}
        shuffle_coa r2, r3, r4, 1
        unlock r0, lock, r0, 0
        ret
        """.format(blkx_addr_offset, shuffle_coa_output_bank_num, length, src_addr, dst_addr)
        sim.set_asm_source(code=asm)
        sim.run()

        result_d = np.zeros((result_row_len, shuffle_coa_output_bank_num), dtype=np.int8)
        for bank_id in range(0, shuffle_coa_output_bank_num):
            ans = sim.get_sram_data_bybank(cluster_id, sim.L1_SRAM_D, bank_id, dst_offset_base, result_row_len * ctypes.sizeof(ctypes.c_int8))
            y = np.frombuffer(ans, dtype=np.int8).reshape(result_row_len)
            result_d[:, bank_id] = y

        #debug
        '''
        print "src_bank_id is ", src_bank_id
        print "src_offset is ", src_offset
        print "dst_bank_id is ", dst_bank_id
        print "dst_offset is ", dst_offset
        print "bank_data_pos is ", bank_data_pos
        print asm
        #print "data is:"
        #print data
        print "focus_data is:"
        print focus_data
        print "result_matrix_golden is"
        print result_matrix_golden
        print "xpu result_d"
        print result_d
        '''

        np.testing.assert_array_equal(result_matrix_golden, result_d)

    def test_shuffle_coa_int16(self):
        src_addr = 0
        dst_addr = 0
        bank_idx = 0
        bank_id = 0
        offset = 0
        data_src_row = 4
        data_src_col = 18

        focus_data_src_row = 3
        focus_data_src_col = 4
        dst_data_col = focus_data_src_row
        dst_data_row = focus_data_src_col
        shuffle_coa_output_bank_num = focus_data_src_col
        length = dst_data_col

        blkx_addr_offset = data_src_col

        # int16
        data = np.random.randint(0, 128, size=(data_src_row * data_src_col)).astype(np.int16)
        '''
        print data
        print "src data matrix:" , data_src_row, "*", data_src_col
        for i in range(0, data_src_row):
            for j in range(0, data_src_col):
                print data[i * data_src_col + j],
            print ""

        print "src data focus matrix:", focus_data_src_row, "*", focus_data_src_col
        for i in range(0, focus_data_src_row):
            for j in range(0, focus_data_src_col):
                print data[i * data_src_col+ j],
            print ""

        result_golden = [0] * (dst_data_row * dst_data_col)
        print "dst data focus matrix:", dst_data_row, "*", dst_data_col
        for i in range(0, dst_data_row):
            for j in range(0, dst_data_col):
                result_golden[i * dst_data_col + j] = data[j * data_src_col + i]
                print data[j * data_src_col + i],
            print ""
        print "result_golden is"
        print result_golden
        '''
        result_golden = [0] * (dst_data_row * dst_data_col)
        for i in range(0, dst_data_row):
            for j in range(0, dst_data_col):
                result_golden[i * dst_data_col + j] = data[j * data_src_col + i]

        cluster_id = 0
        # int16 l2sramd -> l1sramd
        print "test_shuffle_coa int16 l2sramd -> l1sramd"
        sim = None; sim = CDNNSimulator()
        #sim.set_sram_data(sim.L2_SRAM_D, src_addr, data.astype(np.int16).tobytes())
        sim.set_sram_data_bybank(cluster_id, sim.L2_SRAM_D, bank_idx, src_addr, data.astype(np.int16).tobytes())
        asm = """
        rset r0
        addi lock, r0, 1  // lock ds-d
        lock r0, lock, r0, 0
        addi r1, r0, 1 // int16
        ds_cfg r1, r0, r2, 0 // cfg shuffle data type, rs, rt is not necessary in this case
        addi r1, r0, {}
        ds_cfg r1, r0, r2, 1 // cfg shuffle coa blockx, rs, rt is not necessary in this case
        addi r1, r0, {}
        ds_cfg r1, r0, r2, 11
        addi r2, r0, {}
        addi r3, r0, {}
        addi r4, r0, {}
        shuffle_coa r2, r3, r4, 1
        unlock r0, lock, r0, 0
        ret
        """.format(blkx_addr_offset, shuffle_coa_output_bank_num, length, src_addr, dst_addr)
        print asm
        sim.set_asm_source(code=asm)
        sim.run()

        result_d = [0] * (dst_data_col * dst_data_row)
        for bank_id in range(0, dst_data_row):
            ans = sim.get_sram_data_bybank(cluster_id, sim.L1_SRAM_D, bank_id, offset, length * ctypes.sizeof(ctypes.c_int16))
            y = np.frombuffer(ans, dtype=np.int16).reshape(length)
            for y_index in range(0, length):
                result_d[bank_id * dst_data_col + y_index] = y[y_index]
            #print y
        #print "xpu result_d"
        #print result_d
        np.testing.assert_array_equal(result_golden, result_d)

        # int16 l2sramw -> l1sramw
        print "test_shuffle_coa int16 l2sramw -> l1sramw"
        sim = None; sim = CDNNSimulator()
        #sim.set_sram_data(sim.L2_SRAM_W, src_addr, data.astype(np.int16).tobytes())
        sim.set_sram_data_bybank(cluster_id, sim.L2_SRAM_W, bank_idx, src_addr, data.astype(np.int16).tobytes())
        asm = """
        rset r0
        addi lock, r0, 0  // lock ds-w
        lock r0, lock, r0, 0
        addi r1, r0, 1 // int16
        ds_cfg r1, r0, r2, 0 // cfg shuffle data type, rs, rt is not necessary in this case
        addi r1, r0, {}
        ds_cfg r1, r0, r2, 1 // cfg shuffle coa blockx, rs, rt is not necessary in this case
        addi r1, r0, {}
        ds_cfg r1, r0, r2, 11
        addi r2, r0, {}
        addi r3, r0, {}
        addi r4, r0, {}
        shuffle_coa r2, r3, r4, 2
        unlock r0, lock, r0, 0
        ret
        """.format(blkx_addr_offset, shuffle_coa_output_bank_num, length, src_addr, dst_addr)
        print asm
        sim.set_asm_source(code=asm)
        sim.run()

        result_w = [0] * (dst_data_col * dst_data_row)
        for bank_id in range(0, dst_data_row):
            ans = sim.get_sram_data_bybank(cluster_id, sim.L1_SRAM_W, bank_id,
                    offset, length * ctypes.sizeof(ctypes.c_int16))
            y = np.frombuffer(ans, dtype=np.int16).reshape(length)
            for y_index in range(0, length):
                result_w[bank_id * dst_data_col + y_index] = y[y_index]
            #print y
        #print "xpu result_w"
        #print result_w
        np.testing.assert_array_equal(result_golden, result_w)

    def test_shuffle_coa_float32(self):
        src_addr = 0
        dst_addr = 0
        bank_idx = 0
        bank_id = 0
        offset = 0
        data_src_row = 4
        data_src_col = 18

        focus_data_src_row = 3
        focus_data_src_col = 4
        dst_data_col = focus_data_src_row
        dst_data_row = focus_data_src_col
        shuffle_coa_output_bank_num = focus_data_src_col
        length = dst_data_col

        blkx_addr_offset = data_src_col

        # float32
        data = np.random.randint(0, 128, size=(data_src_row * data_src_col)).astype(np.float32)
        '''
        print data
        print "src data matrix:" , data_src_row, "*", data_src_col
        for i in range(0, data_src_row):
            for j in range(0, data_src_col):
                print data[i * data_src_col + j],
            print ""

        print "src data focus matrix:", focus_data_src_row, "*", focus_data_src_col
        for i in range(0, focus_data_src_row):
            for j in range(0, focus_data_src_col):
                print data[i * data_src_col+ j],
            print ""

        result_golden = [0] * (dst_data_row * dst_data_col)
        print "dst data focus matrix:", dst_data_row, "*", dst_data_col
        for i in range(0, dst_data_row):
            for j in range(0, dst_data_col):
                result_golden[i * dst_data_col + j] = data[j * data_src_col + i]
                print data[j * data_src_col + i],
            print ""
        print "result_golden is"
        print result_golden
        '''
        result_golden = [0] * (dst_data_row * dst_data_col)
        for i in range(0, dst_data_row):
            for j in range(0, dst_data_col):
                result_golden[i * dst_data_col + j] = data[j * data_src_col + i]

        cluster_id = 0
        # float32 l2sramd -> l1srame
        print "test_shuffle_coa float32 l2sramd -> l1srame"
        sim = None
        sim = CDNNSimulator()
        sim.set_sram_data_bybank(cluster_id, sim.L2_SRAM_D, bank_idx, src_addr, data.astype(np.float32).tobytes())
        asm = """
        rset r0
        addi lock, r0, 1  // lock ds-d
        lock r0, lock, r0, 0
        addi r1, r0, 2 // float32
        ds_cfg r1, r0, r2, 0 // cfg shuffle data type, rs, rt is not necessary in this case
        addi r1, r0, {}
        ds_cfg r1, r0, r2, 1 // cfg shuffle coa blockx, rs, rt is not necessary in this case
        addi r1, r0, {}
        ds_cfg r1, r0, r2, 11
        addi r2, r0, {}
        addi r3, r0, {}
        addi r4, r0, {}
        shuffle_coa r2, r3, r4, 4
        unlock r0, lock, r0, 0
        ret
        """.format(blkx_addr_offset, shuffle_coa_output_bank_num, length, src_addr, dst_addr)
        print asm
        sim.set_asm_source(code=asm)
        sim.run()

        result_d = [0] * (dst_data_col * dst_data_row)
        for bank_id in range(0, dst_data_row):
            ans = sim.get_sram_data_bybank(cluster_id, sim.L1_SRAM_E, bank_id, offset, length * ctypes.sizeof(ctypes.c_float))
            y = np.frombuffer(ans, dtype=np.float32).reshape(length)
            for y_index in range(0, length):
                result_d[bank_id * dst_data_col + y_index] = y[y_index]
            #print y
        #print "xpu result_d"
        #print result_d
        np.testing.assert_array_equal(result_golden, result_d)

        # float32 l2sramw -> l1srame
        print "test_shuffle_coa float32 l2sramw -> l1srame"
        sim = None
        sim = CDNNSimulator()
        sim.set_sram_data_bybank(cluster_id, sim.L2_SRAM_W, bank_idx, src_addr, data.astype(np.float32).tobytes())
        asm = """
        rset r0
        addi lock, r0, 0  // lock ds-w
        lock r0, lock, r0, 0
        addi r1, r0, 2 // float32
        ds_cfg r1, r0, r2, 0 // cfg shuffle data type, rs, rt is not necessary in this case
        addi r1, r0, {}
        ds_cfg r1, r0, r2, 1 // cfg shuffle coa blockx, rs, rt is not necessary in this case
        addi r1, r0, {}
        ds_cfg r1, r0, r2, 11
        addi r2, r0, {}
        addi r3, r0, {}
        addi r4, r0, {}
        shuffle_coa r2, r3, r4, 4
        unlock r0, lock, r0, 0
        ret
        """.format(blkx_addr_offset, shuffle_coa_output_bank_num, length, src_addr, dst_addr)
        print asm
        sim.set_asm_source(code=asm)
        sim.run()

        result_w = [0] * (dst_data_col * dst_data_row)
        for bank_id in range(0, dst_data_row):
            ans = sim.get_sram_data_bybank(cluster_id, sim.L1_SRAM_E, bank_id,
                    offset, length * ctypes.sizeof(ctypes.c_float))
            y = np.frombuffer(ans, dtype=np.float32).reshape(length)
            for y_index in range(0, length):
                result_w[bank_id * dst_data_col + y_index] = y[y_index]
            #print y
        #print "xpu result_w"
        #print result_w
        np.testing.assert_array_equal(result_golden, result_w)

    def test_l1(self):
        # print "test l1 sram start"
        # x = np.random.randint(-128, 128, size=(1, 30)).astype(np.int32)
        x = "Hello"
        cluster_id = 0
        simulator = CDNNSimulator()
        # simulator.set_global(0, x.astype(np.int8).tobytes())
        # ans = simulator.get_global(5, ctypes.sizeof(ctypes.c_int32))
        simulator.set_sram_data(cluster_id, simulator.L1_SRAM_D, 0, x)
        ans = simulator.get_sram_data(cluster_id, simulator.L1_SRAM_D, 0, 6 * ctypes.sizeof(ctypes.c_int8))
        print "read from l1: " + ans
        x = "Hello World"
        simulator.set_sram_data(cluster_id, simulator.L1_SRAM_D, 0x1CE9, x)
        ans = simulator.get_sram_data(cluster_id, simulator.L1_SRAM_D, 0x1CE9, 20 * ctypes.sizeof(ctypes.c_int8))
        print "read from l1: " + ans

    def test_user_activation(self):
        coeff0 = 1
        coeff1 = 0
        coeff2 = 0
        stream_len = 64
        vld_core_num = 16
        findmax = 1
        oldmax = 0

        src_addr = 0
        src0_offset = 0
        src1_offset = 4 * stream_len
        src1_addr = bankoff2addr(0, src1_offset, 4, 2)
        res_addr = 0

        active = 4 # 0 - null, 1 - relu, 2 - sigmoid, 3 - tanh
        writeto = 0 # 0 - l2E, 1 - l1E
        dual_stream_mode = writeto
        cluster_id = 0
        asm = '''
        rset r0
        addi lock, r0, 3  // lock ew
        lock r0, lock, r0, 0
        addi r4, r0, {coeff0}
        fix2float r1, r4
        addi r4, r0, {coeff1}
        fix2float r2, r4
        addi r4, r0, {coeff2}
        fix2float r3, r4
        ewcoeff_cfg r1, r2, r3, 0 // scalar mode
        addi r1, r0, {stream_len}
        ew_cfg r1, r2, r3, 2 // stream_size
        addi r1, r0, {vld_core_num}
        ew_cfg r1, r2, r3, 1 // vld_core_num
        addi r1, r0, {findmax}
        addi r2, r0, {oldmax}
        ew_cfg r1, r2, r3, 0 // findmax
        addi r1, r0, {active}
        ew_cfg r1, r2, r3, 10 // active
        addi r1, r0, {res_addr}
        addi r2, r0, {src_addr}
        addi r3, r0, {src1_addr}
        dsmadd r1, r2, r3, {dual_stream_mode}
        rset r8
        addi r2, r0, 0 // offset 0
        ld_sd r8, r2, r9, 3 // load EW module data max into r8
        unlock r0, lock, r0, 0
        ret
        '''.format(coeff0=coeff0, coeff1=coeff1, coeff2=coeff2, stream_len=stream_len,
                vld_core_num=vld_core_num, findmax=findmax, oldmax=oldmax, active=active,
                res_addr=res_addr, src_addr=src_addr, src1_addr=src1_addr,
                dual_stream_mode=dual_stream_mode)

        sim = None
        sim = CDNNSimulator()

        # print "prepare memory data"
        depth = stream_len
        load_act_table(sim, 1)
        sim.clear_exception()
        sim.reset()

        # print "src0 data in L1E"
        # x = np.arange(depth * EW_BANK_NUM).astype(np.float32)
        # src0 = x.reshape(EW_BANK_NUM, depth)
        # src0 = src0 / -100.0
        src0 = (np.random.rand(EW_BANK_NUM, depth) - 1.0) * 10
        for i in range(0, EW_BANK_NUM):
            row = src0[i: i + 1]
            # print i,
            # print row
            sim.set_sram_data_bybank(cluster_id, sim.L1_SRAM_E,
                    i, src0_offset, row.astype(np.float32).tobytes())

        # print "src1 data in L1E"
        # src1 = x.reshape(depth, EW_BANK_NUM).T
        src1 = (np.random.rand(EW_BANK_NUM, depth) - 1.0) * 10
        for i in range(0, EW_BANK_NUM):
            row = src1[i: i + 1]
            # print i,
            # print row
            sim.set_sram_data_bybank(cluster_id, sim.L1_SRAM_E,
                    i, src1_offset, row.astype(np.float32).tobytes())

        # print 'set data'
        sim.set_asm_source(code=asm)
        sim.run()

        np_res = src0 * coeff0 + src1 * coeff1 + coeff2

        f = np.vectorize(lambda x: math.exp(x))
        np_res = f(np_res)
        # generated exp negative side table has the last value set to 0 for the input that is out of
        # range, following hack is necessary to make the test pass
        f = np.vectorize(lambda x: 0.0 if (x < 0.0003422) else x)
        np_res = f(np_res)

        # print "ew reslut"
        # f = np.vectorize(lambda x: math.fabs(x))
        for i in range(0, vld_core_num):
            # print "bank " + str(i)
            ans = sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_E, i, 0, stream_len *
                    ctypes.sizeof(ctypes.c_int32))
            y = np.frombuffer(ans, dtype=np.float32)
            row = np.asarray(src0[i: i + 1])
            # index = np.rint(f(row) / (1.0 / 64))
            # print "in:", row
            # print "idx:", index
            # print "xpu:",y
            # print "np:",np_res[i]
            np.testing.assert_allclose(np_res[i], y, 1e-3, 1e-3)
        np_res = np_res[0:vld_core_num]
        result_golden_datamax = abs(np_res).max()
        result_sim_datamax = ctypes.cast(ctypes.pointer(ctypes.c_uint32(
            sim.get_register(0, 0, 'r8'))), ctypes.POINTER(ctypes.c_float))[0]
        np.testing.assert_allclose(result_golden_datamax, result_sim_datamax, 1e-3, 1e-3)

    def test_dsdiv(self):
        coeff0 = 4
        stream_len = 10
        vld_core_num = 16
        findmax = 1
        oldmax = 0

        src_addr = 0
        src0_offset = 0
        src1_offset = 4 * stream_len
        src1_addr = bankoff2addr(0, src1_offset, 4, 2)
        res_addr = 0

        active = 2 # 0 - null, 1 - relu, 2 - sigmoid, 3 - tanh
        writeto = 0 # 0 - l2E, 1 - l1E
        mode = 0; #0:scalar/vector 1:vector/scalar 2:vector/vector
        div_shamt = (mode << 1) | writeto
        cluster_id = 0
        asm = '''
        rset r0
        addi lock, r0, 3  // lock ew
        lock r0, lock, r0, 0
        addi r4, r0, {}
        fix2float r1, r4
        ewcoeff_cfg r1, r2, r3, 0 // scalar mode
        addi r1, r0, {}
        ew_cfg r1, r2, r3, 2 // stream_size
        addi r1, r0, {}
        ew_cfg r1, r2, r3, 1 // vld_core_num
        addi r1, r0, {}
        addi r2, r0, {}
        ew_cfg r1, r2, r3, 0 // findmax
        addi r1, r0, {}
        ew_cfg r1, r2, r3, 10 // active
        addi r1, r0, {}
        addi r2, r0, {}
        addi r3, r0, {}
        dsdiv r1, r2, r3, {}
        rset r8
        addi r2, r0, 0 // offset 0
        ld_sd r8, r2, r9, 3 // load EW module data max into r8
        unlock r0, lock, r0, 0
        ret
        '''.format(coeff0, stream_len, vld_core_num, findmax,
                oldmax, active, res_addr, src_addr, src1_addr, div_shamt)

        sim = None
        sim = CDNNSimulator()

        load_act_table(sim)
        sim.clear_exception()
        sim.reset()

        depth = stream_len
        # print "src0 data in L1E"
        # src0 = x.reshape(EW_BANK_NUM, depth)
        src0 = np.random.rand(EW_BANK_NUM, depth)
        for i in range(0, EW_BANK_NUM):
            row = src0[i: i + 1]
            # print i,
            # print row
            sim.set_sram_data_bybank(cluster_id, sim.L1_SRAM_E, i, src0_offset, row.astype(np.float32).tobytes())

        # print "src1 data in L1E"
        # src1 = x.reshape(depth, EW_BANK_NUM).T
        src1 = np.random.rand(EW_BANK_NUM, depth)
        for i in range(0, EW_BANK_NUM):
            row = src1[i: i + 1]
            # print i,
            # print row
            sim.set_sram_data_bybank(cluster_id, sim.L1_SRAM_E, i, src1_offset, row.astype(np.float32).tobytes())

        # print 'set data'
        sim.set_asm_source(code=asm)
        # print asm
        sim.run()

        if mode == 0:
            np_res = coeff0 / src0
        elif mode == 1:
            np_res = src0 / coeff0
        elif mode == 2:
            np_res = src0 / src1

        f = np.vectorize(null_func)
        if active == 1:
            f = np.vectorize(relu)
        elif active == 2:
            f = np.vectorize(sigmoid)
        elif active == 3:
            f = np.vectorize(tanh)
        np_res = f(np_res)

        # print "ew reslut"
        for i in range(0, vld_core_num):
            # print "bank " + str(i)
            ans = sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_E, i, 0, stream_len *
                    ctypes.sizeof(ctypes.c_int32))
            y = np.frombuffer(ans, dtype=np.float32)
            # print "xpu:",y
            # print "np:",np_res[i]
            # y = [s & 0xFFFFFFFF for s in y]
            np.testing.assert_allclose(y, np_res[i], 1e-3, 1e-3)
            # np.savetxt("bank-{}.txt".format(i), y, fmt="%08x", delimiter='\n')
            # out_lst = ('{:08x}'.format(ele) for ele in y)
            # out_lst = (hex(ele & 0xFFFFFFFF) for ele in y)
            # open("bank-{}.txt".format(i), 'w').write('\n'.join(out_lst))
        np_res = np_res[0:vld_core_num]
        result_golden_datamax = abs(np_res).max()
        result_sim_datamax = ctypes.cast(ctypes.pointer(ctypes.c_uint32(sim.get_register(0, 0, 'r8'))),
            ctypes.POINTER(ctypes.c_float))[0]
        # print "result_golden_datamax is ",result_golden_datamax
        # print "result_sim_datamax is", result_sim_datamax
        #np.testing.assert_almost_equal(result_golden_datamax, result_sim_datamax, decimal=4)
        np.testing.assert_allclose(result_golden_datamax, result_sim_datamax, 1e-3, 1e-3)

    def test_simple_dsmadd(self):
        coeff0 = 4
        coeff1 = 3
        coeff2 = 2
        stream_len = 30
        vld_core_num = 16
        findmax = 1
        oldmax = 0

        src_addr = 0
        src0_offset = 0
        src1_offset = 4 * stream_len
        src1_addr = bankoff2addr(0, src1_offset, 4, 2)
        res_addr = 0

        active = 3 # 0 - null, 1 - relu, 2 - sigmoid, 3 - tanh
        writeto = 0 # 0 - l2E, 1 - l1E
        dual_stream_mode = writeto
        cluster_id = 0
        asm = '''
        rset r0
        addi lock, r0, 3  // lock ew
        lock r0, lock, r0, 0
        core_id r1
        bne r1, r0, end
        addi r4, r0, {coeff0}
        fix2float r1, r4
        addi r4, r0, {coeff1}
        fix2float r2, r4
        addi r4, r0, {coeff2}
        fix2float r3, r4
        ewcoeff_cfg r1, r2, r3, 0 // scalar mode
        addi r1, r0, {stream_len}
        ew_cfg r1, r2, r3, 2 // stream_size
        addi r1, r0, {vld_core_num}
        ew_cfg r1, r2, r3, 1 // vld_core_num
        addi r1, r0, {findmax}
        addi r2, r0, {oldmax}
        ew_cfg r1, r2, r3, 0 // findmax
        addi r1, r0, {active}
        ew_cfg r1, r2, r3, 10 // active
        addi r1, r0, {res_addr}
        addi r2, r0, {src_addr}
        addi r3, r0, {src1_addr}
        dsmadd r1, r2, r3, {dual_stream_mode}
        rset r8
        addi r2, r0, 0 // offset 0
        ld_sd r8, r2, r9, 3 // load EW module data max into r8
        unlock r0, lock, r0, 0
        end:
        ret
        '''.format(coeff0=coeff0, coeff1=coeff1, coeff2=coeff2, stream_len=stream_len,
                vld_core_num=vld_core_num, findmax=findmax, oldmax=oldmax, active=active,
                res_addr=res_addr, src_addr=src_addr, src1_addr=src1_addr,
                dual_stream_mode=dual_stream_mode)

        sim = None
        sim = CDNNSimulator()

        # print "prepare memory data"
        depth = stream_len
        # x = np.arange(depth * EW_BANK_NUM).astype(np.float32)
        load_act_table(sim)
        sim.clear_exception()
        sim.reset()

        # print "src0 data in L1E"
        # src0 = x.reshape(EW_BANK_NUM, depth)
        src0 = np.random.rand(EW_BANK_NUM, depth)
        for i in range(0, EW_BANK_NUM):
            row = src0[i: i + 1]
            # print i,
            # print row
            sim.set_sram_data_bybank(cluster_id, sim.L1_SRAM_E, i, src0_offset, row.astype(np.float32).tobytes())

        # print "src1 data in L1E"
        # src1 = x.reshape(depth, EW_BANK_NUM).T
        src1 = np.random.rand(EW_BANK_NUM, depth)
        for i in range(0, EW_BANK_NUM):
            row = src1[i: i + 1]
            # print i,
            # print row
            sim.set_sram_data_bybank(cluster_id, sim.L1_SRAM_E, i, src1_offset, row.astype(np.float32).tobytes())

        # print 'set data'
        sim.set_asm_source(code=asm)
        sim.run()

        np_res = src0 * coeff0 + src1 * coeff1 + coeff2

        f = np.vectorize(null_func)
        if active == 1:
            f = np.vectorize(relu)
        elif active == 2:
            f = np.vectorize(sigmoid)
        elif active == 3:
            f = np.vectorize(tanh)
        np_res = f(np_res)

        # print "ew reslut"
        for i in range(0, vld_core_num):
            # print "bank " + str(i)
            ans = sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_E, i, 0, stream_len *
                    ctypes.sizeof(ctypes.c_int32))
            y = np.frombuffer(ans, dtype=np.float32)
            # print "xpu:",y
            # print "np:",np_res[i]
            # y = [s & 0xFFFFFFFF for s in y]
            np.testing.assert_allclose(y, np_res[i], 1e-3, 1e-3)
            # np.savetxt("bank-{}.txt".format(i), y, fmt="%08x", delimiter='\n')
            # out_lst = ('{:08x}'.format(ele) for ele in y)
            # out_lst = (hex(ele & 0xFFFFFFFF) for ele in y)
            # open("bank-{}.txt".format(i), 'w').write('\n'.join(out_lst))
        np_res = np_res[0:vld_core_num]
        result_golden_datamax = abs(np_res).max()
        result_sim_datamax = ctypes.cast(ctypes.pointer(ctypes.c_uint32(sim.get_register(0, 0, 'r8'))),
            ctypes.POINTER(ctypes.c_float))[0]
        #np.testing.assert_almost_equal(result_golden_datamax, result_sim_datamax, decimal=5)
        np.testing.assert_allclose(result_golden_datamax, result_sim_datamax, 1e-3, 1e-3)

    # coeff2 in vector
    def test_cvec_dsmadd(self):
        coeff0 = 2
        coeff1 = 3
        coeff2 = 4
        stream_len = 40

        vld_core_num = 16
        findmax = 1
        oldmax = 1

        src_addr = 0
        src0_offset = 0
        src1_offset = 4 * stream_len
        src1_addr = bankoff2addr(0, src1_offset, 4, 2)
        res_addr = 0

        coeff_base_offset = src0_offset + 4 * stream_len + 4 * stream_len
        coeff0_offset = coeff_base_offset + 4
        coeff0_vec_addr = bankoff2addr(0, coeff0_offset, 4, 2)
        coeff1_offset = coeff_base_offset + 8
        coeff1_vec_addr = bankoff2addr(0, coeff1_offset, 4, 2)
        coeff2_offset = coeff_base_offset + 12
        coeff2_vec_addr = bankoff2addr(0, coeff2_offset, 4, 2)

        active = 2 # 0 - null, 1 - relu, 2 - sigmoid, 3 - tanh
        writeto = 0 # 0 - l2E, 1 - l1E
        dual_stream_mode = writeto
        asm = '''
        rset r0
        addi lock, r0, 3  // lock ew
        lock r0, lock, r0, 0
        addi r1, r0, {}
        addi r2, r0, {}
        addi r3, r0, {}
        ewcoeff_cfg r1, r2, r3, 7 // vector mode
        addi r1, r0, {}
        ew_cfg r1, r2, r3, 2 // stream_size
        addi r1, r0, {}
        ew_cfg r1, r2, r3, 1 // vld_core_num
        addi r1, r0, {}
        addi r2, r0, {}
        ew_cfg r1, r2, r3, 0 // findmax
        addi r1, r0, {}
        ew_cfg r1, r2, r3, 10 // active
        addi r1, r0, {}
        addi r2, r0, {}
        addi r3, r0, {}
        dsmadd r1, r2, r3, {}
        rset r8
        addi r2, r0, 0 // offset 0
        ld_sd r8, r2, r9, 3 // load EW module data max into r8
        unlock r0, lock, r0, 0
        ret
        '''.format(coeff0_vec_addr, coeff1_vec_addr, coeff2_vec_addr, stream_len, vld_core_num, findmax,
                oldmax, active, res_addr, src_addr, src1_addr, dual_stream_mode)

        cluster_id = 0
        sim = None
        sim = CDNNSimulator()

        load_act_table(sim)
        sim.clear_exception()
        sim.reset()

        # print "prepare memory data"
        depth = stream_len
        # x = np.arange(depth * EW_BANK_NUM).astype(np.float32)

        # src0 = x.reshape(EW_BANK_NUM, depth)
        src0 = np.random.rand(EW_BANK_NUM, depth)
        for i in range(0, EW_BANK_NUM):
            row = src0[i: i + 1]
            # print i,
            # print row
            sim.set_sram_data_bybank(cluster_id, sim.L1_SRAM_E, i, src0_offset, row.astype(np.float32).tobytes())

        # src1 = x.reshape(depth, EW_BANK_NUM).T
        src1 = np.random.rand(EW_BANK_NUM, depth)
        for i in range(0, EW_BANK_NUM):
            row = src1[i: i + 1]
            # print i,
            # print row
            sim.set_sram_data_bybank(cluster_id, sim.L1_SRAM_E, i, src1_offset, row.astype(np.float32).tobytes())

        coeff0 = np.random.rand(EW_BANK_NUM, 1)
        coeff1 = np.random.rand(EW_BANK_NUM, 1)
        coeff2 = np.random.rand(EW_BANK_NUM, 1)
        for i in range(0, EW_BANK_NUM):
            row0 = coeff0[i: i + 1]
            row1 = coeff1[i: i + 1]
            row2 = coeff2[i: i + 1]
            # print i,
            # print row0,
            # print row1,
            # print row2
            sim.set_sram_data_bybank(cluster_id, sim.L1_SRAM_E, i, coeff0_offset,
                    row0.astype(np.float32).tobytes())
            sim.set_sram_data_bybank(cluster_id, sim.L1_SRAM_E, i, coeff1_offset,
                    row1.astype(np.float32).tobytes())
            sim.set_sram_data_bybank(cluster_id, sim.L1_SRAM_E, i, coeff2_offset,
                    row2.astype(np.float32).tobytes())

        # print 'set data'
        sim.set_asm_source(code=asm)
        sim.run()

        np_res = np.zeros((EW_BANK_NUM, stream_len), dtype=np.float32)
        for i in range(0, EW_BANK_NUM):
            for j in range(0, stream_len):
                np_res[i][j] = coeff0[i][0] * src0[i][j] + coeff1[i][0] * src1[i][j] + coeff2[i][0]

        f = np.vectorize(null_func)
        if active == 1:
            f = np.vectorize(relu)
        elif active == 2:
            f = np.vectorize(sigmoid)
        elif active == 3:
            f = np.vectorize(tanh)

        np_res = f(np_res)

        # print "ew reslut"
        for i in range(0, vld_core_num):
            # print "bank " + str(i)
            ans = sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_E, i, 0, stream_len *
                    ctypes.sizeof(ctypes.c_int32))
            y = np.frombuffer(ans, dtype=np.float32)
            # print "xpu:",y
            # print "np:",np_res[i]
            # y = [s & 0xFFFFFFFF for s in y]
            np.testing.assert_allclose(y, np_res[i], 1e-3, 1e-3)
            # np.savetxt("bank-{}.txt".format(i), y, fmt="%08x", delimiter='\n')
            # out_lst = ('{:08x}'.format(ele) for ele in y)
            # out_lst = (hex(ele & 0xFFFFFFFF) for ele in y)
            # open("bank-{}.txt".format(i), 'w').write('\n'.join(out_lst))
        np_res = np_res[0:vld_core_num]
        result_golden_datamax = abs(np_res).max()
        result_sim_datamax = ctypes.cast(ctypes.pointer(ctypes.c_uint32(sim.get_register(0, 0, 'r8'))),
            ctypes.POINTER(ctypes.c_float))[0]
        #np.testing.assert_almost_equal(result_golden_datamax, result_sim_datamax, decimal=5)
        np.testing.assert_allclose(result_golden_datamax, result_sim_datamax, 1e-3, 1e-3)

    def test_dsmul(self):
        coeff0 = 4
        coeff1 = 3
        coeff2 = 2
        stream_len = 40
        vld_core_num = 16
        findmax = 1
        oldmax = 0

        src_addr = 0
        src0_offset = 0
        src1_offset = 4 * stream_len
        src1_addr = bankoff2addr(0, src1_offset, 4, 2)
        res_addr = 0

        active = 3 # 0 - null, 1 - relu, 2 - sigmoid, 3 - tanh
        writeto = 0 # 0 - l2E, 1 - l1E
        dual_stream_mode = writeto
        asm = '''
        rset r0
        addi lock, r0, 3  // lock ew
        lock r0, lock, r0, 0
        addi r4, r0, {}
        fix2float r1, r4
        addi r4, r0, {}
        fix2float r2, r4
        addi r4, r0, {}
        fix2float r3, r4
        ewcoeff_cfg r1, r2, r3, 0 // scalar mode
        addi r1, r0, {}
        ew_cfg r1, r2, r3, 2 // stream_size
        addi r1, r0, {}
        ew_cfg r1, r2, r3, 1 // vld_core_num
        addi r1, r0, {}
        addi r2, r0, {}
        ew_cfg r1, r2, r3, 0 // findmax
        addi r1, r0, {}
        ew_cfg r1, r2, r3, 10 // active
        addi r1, r0, {}
        addi r2, r0, {}
        addi r3, r0, {}
        dsmul r1, r2, r3, {}
        rset r8
        addi r2, r0, 0 // offset 0
        ld_sd r8, r2, r9, 3 // load EW module data max into r8
        unlock r0, lock, r0, 0
        ret
        '''.format(coeff0, coeff1, coeff2, stream_len, vld_core_num, findmax, oldmax, active, res_addr,
                src_addr, src1_addr, dual_stream_mode)

        cluster_id = 0
        sim = None
        sim = CDNNSimulator()

        load_act_table(sim)
        sim.clear_exception()
        sim.reset()

        # print "prepare memory data"
        depth = stream_len
        # x = np.arange(depth * EW_BANK_NUM).astype(np.float32)

        # src0 = x.reshape(EW_BANK_NUM, depth)
        src0 = np.random.rand(EW_BANK_NUM, depth)
        for i in range(0, EW_BANK_NUM):
            row = src0[i: i + 1]
            # print i,
            # print row
            sim.set_sram_data_bybank(cluster_id, sim.L1_SRAM_E, i, src0_offset, row.astype(np.float32).tobytes())

        # print "data in L1-C"
        # src1 = x.reshape(depth, EW_BANK_NUM).T
        src1 = np.random.rand(EW_BANK_NUM, depth)
        for i in range(0, EW_BANK_NUM):
            row = src1[i: i + 1]
            # print i,
            # print row
            sim.set_sram_data_bybank(cluster_id, sim.L1_SRAM_E, i, src1_offset, row.astype(np.float32).tobytes())

        sim.set_asm_source(code=asm)
        sim.run()

        np_res = (coeff0 + src0) * (coeff1 + src1) * coeff2

        f = np.vectorize(null_func)
        if active == 1:
            f = np.vectorize(relu)
        elif active == 2:
            f = np.vectorize(sigmoid)
        elif active == 3:
            f = np.vectorize(tanh)

        np_res = f(np_res)
        # print "dsmul reslut"
        for i in range(0, vld_core_num):
            # print "bank " + str(i)
            ans = sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_E, i, 0, stream_len *
                    ctypes.sizeof(ctypes.c_int32))
            y = np.frombuffer(ans, dtype=np.float32)
            # print "xpu:",y
            # print "np:",np_res[i]
            # y = [s & 0xFFFFFFFF for s in y]
            np.testing.assert_allclose(y, np_res[i], 1e-3, 1e-3)
            # np.savetxt("bank-{}.txt".format(i), y, fmt="%08x", delimiter='\n')
            # out_lst = ('{:08x}'.format(ele) for ele in y)
            # out_lst = (hex(ele & 0xFFFFFFFF) for ele in y)
            # open("bank-{}.txt".format(i), 'w').write('\n'.join(out_lst))
        np_res = np_res[0:vld_core_num]
        result_golden_datamax = abs(np_res).max()
        result_sim_datamax = ctypes.cast(ctypes.pointer(ctypes.c_uint32(sim.get_register(0, 0, 'r8'))),
            ctypes.POINTER(ctypes.c_float))[0]
        #np.testing.assert_almost_equal(result_golden_datamax, result_sim_datamax, decimal=5)
        np.testing.assert_allclose(result_golden_datamax, result_sim_datamax, 1e-3, 1e-3)

    def test_dscmpnsel(self):
        stream_len = 40
        vld_core_num = 16
        findmax = 1
        oldmax = 0

        src_addr = 0
        src0_offset = 0
        src1_offset = 4 * stream_len
        src1_addr = bankoff2addr(0, src1_offset, 4, 2)
        res_addr = 0

        compare = 1 # 0 - max, 1 - min
        active = 2 # 0 - null, 1 - relu, 2 - sigmoid, 3 - tanh
        writeto = 0 # 0 - l2E, 1 - l1E
        dual_stream_mode = (compare << 2) | writeto
        asm = '''
        rset r0
        addi lock, r0, 3  // lock ew
        lock r0, lock, r0, 0
        addi r1, r0, {}
        ew_cfg r1, r2, r3, 2 // stream_size
        addi r1, r0, {}
        ew_cfg r1, r2, r3, 1 // vld_core_num
        addi r1, r0, {}
        addi r2, r0, {}
        ew_cfg r1, r2, r3, 0 // findmax
        addi r1, r0, {}
        ew_cfg r1, r2, r3, 10 // active
        addi r1, r0, {}
        addi r2, r0, {}
        addi r3, r0, {}
        dscmpnsel r1, r2, r3, {}
        rset r8
        addi r2, r0, 0 // offset 0
        ld_sd r8, r2, r9, 3 // load EW module data max into r8
        unlock r0, lock, r0, 0
        ret
        '''.format(stream_len, vld_core_num, findmax,
                oldmax, active, res_addr, src_addr, src1_addr, dual_stream_mode)

        cluster_id = 0
        sim = None
        sim = CDNNSimulator()

        load_act_table(sim)
        sim.clear_exception()
        sim.reset()

        # print "prepare memory data"
        depth = stream_len
        #x = np.arange(depth * EW_BANK_NUM).astype(np.float32)

        # src0 = x.reshape(EW_BANK_NUM, depth)
        src0 = np.random.rand(EW_BANK_NUM, depth)
        for i in range(0, EW_BANK_NUM):
            row = src0[i: i + 1]
            # print i,
            # print row
            sim.set_sram_data_bybank(cluster_id, sim.L1_SRAM_E, i, src0_offset, row.astype(np.float32).tobytes())

        # print "data in L1-C"
        # src1 = x.reshape(depth, EW_BANK_NUM).T
        src1 = np.random.rand(EW_BANK_NUM, depth)
        for i in range(0, EW_BANK_NUM):
            row = src1[i: i + 1]
            # print i,
            # print row
            sim.set_sram_data_bybank(cluster_id, sim.L1_SRAM_E, i, src1_offset, row.astype(np.float32).tobytes())

        sim.set_asm_source(code=asm)
        sim.run()

        if compare == 0:
            np_res = np.maximum(src0, src1)
        elif compare == 1:
            np_res = np.minimum(src0, src1)

        f = np.vectorize(null_func)
        if active == 1:
            f = np.vectorize(relu)
        elif active == 2:
            f = np.vectorize(sigmoid)
        elif active == 3:
            f = np.vectorize(tanh)

        np_res = f(np_res)
        # print "dscmpnsel reslut"
        for i in range(0, vld_core_num):
            # print "bank " + str(i)
            ans = sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_E, i, 0, stream_len *
                    ctypes.sizeof(ctypes.c_int32))
            y = np.frombuffer(ans, dtype=np.float32)
            # print "xpu:",y
            # print "np:",np_res[i]
            np.testing.assert_allclose(y, np_res[i], 1e-3, 1e-3)
            # y = [s & 0xFFFFFFFF for s in y]
            # np.savetxt("bank-{}.txt".format(i), y, fmt="%08x", delimiter='\n')
            # out_lst = ('{:08x}'.format(ele) for ele in y)
            # out_lst = (hex(ele & 0xFFFFFFFF) for ele in y)
            # open("bank-{}.txt".format(i), 'w').write('\n'.join(out_lst))
        np_res = np_res[0:vld_core_num]
        result_golden_datamax = abs(np_res).max()
        result_sim_datamax = ctypes.cast(ctypes.pointer(ctypes.c_uint32(sim.get_register(0, 0, 'r8'))),
            ctypes.POINTER(ctypes.c_float))[0]
        #np.testing.assert_almost_equal(result_golden_datamax, result_sim_datamax, decimal=5)
        np.testing.assert_allclose(result_golden_datamax, result_sim_datamax, 1e-3, 1e-3)

    def test_ssvsum(self):
        stream_len = 40
        vld_core_num = 16
        findmax = 1
        oldmax = 0

        src_addr = 0
        src0_offset = 0
        res_addr = 0

        active = 3 # 0 - null, 1 - relu, 2 - sigmoid, 3 - tanh
        writeto = 0 # 0 - l2E, 1 - l1E
        dual_stream_mode = writeto
        asm = '''
        rset r0
        addi lock, r0, 3  // lock ew
        lock r0, lock, r0, 0
        addi r1, r0, {}
        ew_cfg r1, r2, r3, 2 // stream_size
        addi r1, r0, {}
        ew_cfg r1, r2, r3, 1 // vld_core_num
        addi r1, r0, {}
        addi r2, r0, {}
        ew_cfg r1, r2, r3, 0 // findmax
        addi r1, r0, {}
        ew_cfg r1, r2, r3, 10 // active
        addi r1, r0, {}
        addi r2, r0, {}
        ssvsum r1, r2, r3, {}
        rset r8
        addi r2, r0, 0 // offset 0
        ld_sd r8, r2, r9, 3 // load EW module data max into r8
        unlock r0, lock, r0, 0
        ret
        '''.format(stream_len, vld_core_num, findmax,
                oldmax, active, res_addr, src_addr, dual_stream_mode)

        cluster_id = 0
        sim = None
        sim = CDNNSimulator()

        load_act_table(sim)
        sim.clear_exception()
        sim.reset()

        depth = stream_len

        src0 = np.random.rand(EW_BANK_NUM, depth)
        for i in range(0, EW_BANK_NUM):
            row = src0[i: i + 1]
            # print i,
            # print row
            sim.set_sram_data_bybank(cluster_id, sim.L1_SRAM_E, i, src0_offset, row.astype(np.float32).tobytes())

        sim.set_asm_source(code=asm)
        sim.run()

        np_res = np.random.rand(EW_BANK_NUM, 1)
        for i in range(0, EW_BANK_NUM):
            np_res[i] = np.sum(src0[i: i + 1])

        f = np.vectorize(null_func)
        if active == 1:
            f = np.vectorize(relu)
        elif active == 2:
            f = np.vectorize(sigmoid)
        elif active == 3:
            f = np.vectorize(tanh)

        np_res = f(np_res)
        for i in range(0, vld_core_num):
            ans = sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_E, i, 0, ctypes.sizeof(ctypes.c_float))
            y = np.frombuffer(ans, dtype=np.float32)
            # print "xpu:",y
            # print "np:",np_res[i]
            np.testing.assert_allclose(y, np_res[i], 1e-3, 1e-3)
        np_res = np_res[0:vld_core_num]
        result_golden_datamax = abs(np_res).max()
        result_sim_datamax = ctypes.cast(ctypes.pointer(ctypes.c_uint32(sim.get_register(0, 0, 'r8'))),
            ctypes.POINTER(ctypes.c_float))[0]
        #np.testing.assert_almost_equal(result_golden_datamax, result_sim_datamax, decimal=5)
        np.testing.assert_allclose(result_golden_datamax, result_sim_datamax, 1e-3, 1e-3)

    def test_mac(self):
        data_type = 0b10000
        # m = 4
        m = 17
        # k = 17
        k = 354
        # k = 64
        if is_int8_mac(data_type):
            m = 73
            k = 334
        int8_off = m + 2
        n = 16
        dqt_scale = 1

        addr_a = bankoff2addr(0, 2, bank_bits=4)
        addr_b = bankoff2addr(0, 32, bank_bits=4)
        addr_d = bankoff2addr(0, 35, bank_bits=4, align_bits=2)
        asm = '''
        rset r0
        addi lock, r0, 2  // lock mac
        lock r0, lock, r0, 0
        core_id id
        bne id, r0, end
        addi r1, r0, {}
        addi r2, r0, {}
        addi r3, r0, {}
        mm_cfg r3, r1, r2, 0 // k in rs, m in rt
        addi r4, r0, {}
        fix2float r1, r4
        mm_cfg r0, r1, r0, 1 // dequant scale in rs
        addi r1, r0, {}
        addi r2, r0, {}
        addi r3, r0, {}
        addi stride, r0, 1
        mm_cfg r0, stride, r0, 2
        mm r3, r1, r2, {} // addr_d in rd, addr_a in rs addr_b in rt, shamt for data_type 0 - int16, 1 - int8
        unlock r0, lock, r0, 0
        end:
        ret
        '''.format(k, m, int8_off, dqt_scale, addr_a, addr_b, addr_d, data_type)

        a_bank, a_off = addr2bankoff(addr_a, bank_bits=4)
        b_bank, b_off = addr2bankoff(addr_b, bank_bits=4)
        d_bank, d_off = addr2bankoff(addr_d, bank_bits=4, align_bits=2)
        # print "addr_a is " + str(addr_a)
        # print "addr_b is " + str(addr_b)
        # print "addr_d is " + str(addr_d)
        # print "a_off is " + str(a_off)
        # print "b_off is " + str(b_off)
        # print "d_off is " + str(d_off)


        np_dt = np.int16
        ct_dt = ctypes.c_int16
        accuracy = 1e-3
        depth = m
        if is_int8_mac(data_type):
            np_dt = np.int8
            ct_dt = ctypes.c_int8
            accuracy = 1e-1
            n = 32
            depth = m + (int8_off - m) + m
            # print "depth = ", depth
            # print "m = ", m
            # print "int8_off = ", int8_off

        cluster_id = 0
        sim = None; sim = CDNNSimulator()
        reset_sram(sim)

        # print "prepare memory data"
        x = np.arange(m * k).astype(np_dt)

        vec_a = x.reshape(m, k)

        rfm_a = vec_a
        if is_int8_mac(data_type):
            if data_type == 0b10001 or data_type == 0b10011:
                vec_a = vec_a >> 4
                rfm_a = wrap_int4(vec_a)

            rfm_a = vec_reform_int8(rfm_a)

        # print "vec_a"
        # print vec_a.shape
        # print vec_a
        # print "rfm_a"
        # print rfm_a.shape
        # print rfm_a
        roundup_k = (k + 16 - 1) / 16 * 16
        if data_type == 0b10001 or data_type == 0b10011:
            roundup_k = ((k + 1) / 2 + 16 - 1) / 16 * 16
        for i in range(0, rfm_a.shape[0]):
            row = rfm_a[i: i + 1]
            # print i,
            # print row
            mbid = i / 16
            # mboff = mbid * roundup_k * ctypes.sizeof(ct_dt)
            mboff = mbid * roundup_k * 2
            sim.set_sram_data_bybank(cluster_id,
                    sim.L1_SRAM_D, (i % 16), a_off + mboff, row.astype(np_dt).tobytes())

        # for i in range(0, 16):
        #     data2 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L1_SRAM_D, i,
        #         a_off, 4 * roundup_k * ctypes.sizeof(ct_dt)), dtype=np_dt)
        #     print "data2"
        #     print data2

        vec_b = np.arange(k * n).astype(np_dt).reshape(n, k)
        # vec_b = np.random.rand(k, n)
        rfm_b = vec_b
        if is_int8_mac(data_type):
            if data_type == 0b10010 or data_type == 0b10011:
                vec_b = vec_b >> 4
                rfm_b = wrap_int4(vec_b)
            rfm_b = vec_reform_int8(rfm_b, 16)

        # print "rfm_b"
        for i in range(0, rfm_b.shape[0]):
            row = rfm_b[i: i + 1]
            # print i,
            # print row
            sim.set_sram_data_bybank(cluster_id,
                    sim.L1_SRAM_W, i, b_off, row.astype(np_dt).tobytes())

        sim.set_asm_source(code=asm)
        sim.run()

        vec_a = vec_a.astype(float)
        vec_b = vec_b.astype(float)
        np_res = np.dot(vec_a, vec_b.T).T
        # print "vec_a"
        # print vec_a
        # print "vec_b"
        # print vec_b
        # print "result of vec_a * vec_b"
        # print np_res.shape
        # print np_res

        if is_int8_mac(data_type):
            # print "first half result:"
            res0 = np_res[0:16]
            # print "second half result:"
            res1 = np_res[16:32]
            off_res = np.zeros((int8_off - m) * 16).reshape(16, (int8_off - m))
            # print off_res.shape
            # print res0.shape
            np_res = np.hstack((res0, off_res, res1))
            # print "full result:"
            # print np_res
            # print np_res.shape

        # print "mac reslut"
        for i in range(0, 16):
            # print "bank " + str(i)
            ans = sim.get_sram_data_bybank(cluster_id,
                    sim.L1_SRAM_E, i, d_off, depth * ctypes.sizeof(ctypes.c_float))
            y = np.frombuffer(ans, dtype=np.float32)
            # print y
            # y = [s & 0xFFFFFFFF for s in y]
            np.testing.assert_allclose(y, np_res[i], accuracy, accuracy)
            # np.savetxt("bank-{}.txt".format(i), y, fmt="%08x", delimiter='\n')
            # out_lst = ('{:08x}'.format(ele) for ele in y)
            # out_lst = (hex(ele & 0xFFFFFFFF) for ele in y)
            # open("bank-{}.txt".format(i), 'w').write('\n'.join(out_lst))

    def test_mac_stride(self):
        data_type = 0b00000
        stride = 2
        # m = 4
        m = 16
        # k = 17
        k = 359
        int8_off = m + 2
        n = 16
        addr_a = bankoff2addr(0, 2, bank_bits=4)
        addr_b = bankoff2addr(0, 32, bank_bits=4)
        addr_d = bankoff2addr(0, 35, bank_bits=4, align_bits=2)
        dqt_scale = 1
        asm = '''
        rset r0
        addi lock, r0, 2  // lock mac
        lock r0, lock, r0, 0
        addi r1, r0, {}
        addi r2, r0, {}
        addi r3, r0, {}
        mm_cfg r3, r1, r2, 0 // k in rs, m in rt
        addi r4, r0, {}
        fix2float r1, r4
        mm_cfg r0, r1, r2, 1 // Vector A max in rs, Vector B max in rt
        addi r1, r0, {}
        addi r2, r0, {}
        addi r3, r0, {}
        addi stride, r0, {}
        mm_cfg r0, stride, r0, 2
        mm r3, r1, r2, {} // addr_d in rd, addr_a in rs addr_b in rt, shamt for data_type 0 - int16, 1 - int8
        unlock r0, lock, r0, 0
        ret
        '''.format(k, m, int8_off, dqt_scale, addr_a, addr_b, addr_d, stride, data_type)

        a_bank, a_off = addr2bankoff(addr_a, bank_bits=4)
        b_bank, b_off = addr2bankoff(addr_b, bank_bits=4)
        d_bank, d_off = addr2bankoff(addr_d, bank_bits=4, align_bits=2)

        np_dt = np.int16
        ct_dt = ctypes.c_int16
        accuracy = 1e-3
        depth = m * stride

        cluster_id = 0
        sim = None
        sim = CDNNSimulator()
        reset_sram(sim)

        # print "prepare memory data"
        x = np.arange(m * k).astype(np_dt)

        # print "data in bypass"
        vec_a = x.reshape(m, k)

        rfm_a = vec_a

        # print "rfm_a"
        for i in range(0, rfm_a.shape[0]):
            row = rfm_a[i: i + 1]
            # print i,
            # print row
            sim.set_sram_data_bybank(cluster_id,
                    sim.L1_SRAM_D, i, a_off, row.astype(np_dt).tobytes())

        vec_b = np.arange(k * n).astype(np_dt).reshape(n, k)
        # vec_b = np.random.rand(k, n)
        rfm_b = vec_b

        # print "rfm_b"
        for i in range(0, rfm_b.shape[0]):
            row = rfm_b[i: i + 1]
            # print i,
            # print row
            sim.set_sram_data_bybank(cluster_id,
                    sim.L1_SRAM_W, i, b_off, row.astype(np_dt).tobytes())

        sim.set_asm_source(code=asm)
        sim.run()

        vec_a = vec_a.astype(float)
        vec_b = vec_b.astype(float)
        np_res = np.dot(vec_a, vec_b.T).T
        # print "vec_a"
        # print vec_a
        # print "vec_b"
        # print vec_b
        # print "result of vec_a * vec_b"
        # print np_res.shape
        # print np_res

        for i in range(0, 16):
            ans = sim.get_sram_data_bybank(cluster_id,
                    sim.L1_SRAM_E, i, d_off, depth * ctypes.sizeof(ctypes.c_float))
            y = np.frombuffer(ans, dtype=np.float32)
            # print y
            golden = np.zeros(depth).astype(np.float32)
            golden = [np_res[i][j / 2] if j % 2 == 0 else 0 for j in range(depth)]
            np.testing.assert_allclose(y, golden, accuracy, accuracy)


    def test_readL2R(self):
        type2info = {
            'float32': ('000', 4),
            'int8': ('100', 1),
            'int16': ('010', 2),
            'float16': ('001', 2),
            'bfloat16': ('011', 2),
            'int31': ('101', 2),
        }
        for t in ('float32', 'int8', 'int16', 'float16', 'bfloat16', 'int31'):
            max_val = 16383
            info = type2info[t]
            int31_l2_hi = 16384 * info[1]
            asm = '''
            rset zero
            core_id id
            bne id, zero, end
            addi header, zero, {header}
            fix2float header, header
            # L2R => HBM
            ori l2_hi, zero, {l2_hi}
            slli l2_hi, l2_hi, 12
            addi dst_desc, zero, 0b000{type}0000
            or dst_desc, dst_desc, l2_hi
            addi src_desc, zero, 0b0100000000
            addi lk, zero, 7
            lock zero, lk, zero, 0
            l2_mov_cfg dst_desc, src_desc, header
            addi len, zero, 4096;           # 4KB per bank
            muli len, len, 16;              # 16 banks per core
            addi src, zero, 0;              # logical address on L2R
            addi dst, zero, 1024;           # compute absolute offset on GM
            muli dst, dst, 16
            muli dst, dst, {data_size}
            mul dst, dst, id
            l2_mov dst, src, len, 0
            unlock zero, lk, zero, 0
            # L2R => L2D
            addi dst_desc, zero, 0b110{type}0000
            or dst_desc, dst_desc, l2_hi
            lock zero, lk, zero, 0
            l2_mov_cfg dst_desc, src_desc, header
            l2_mov zero, src, len, 0
            unlock zero, lk, zero, 0
            # L2R => L2W
            addi dst_desc, zero, 0b100{type}0000
            or dst_desc, dst_desc, l2_hi
            lock zero, lk, zero, 0
            l2_mov_cfg dst_desc, src_desc, header
            l2_mov zero, src, len, 0
            unlock zero, lk, zero, 0
            end:
            ret
            '''.format(header=max_val, type=info[0], data_size=info[1], l2_hi=int31_l2_hi)
            sim = None; sim = CDNNSimulator()
            sim.set_asm_source(code=asm)
            cluster_id = 0
            # generate data in special format, to read "0...16383" back from L2R
            for i in range(16):
                start = i % 16
                stop = start + 16 * 1024
                data = np.arange(start, stop, 16).astype(np.float32).tobytes()
                sim.set_sram_data_bybank(cluster_id, sim.L2_SRAM_R, i, 0, data)
            sim.run()
            size = 16 * 1024 * info[1]
            if t == 'float32':
                data1 = np.frombuffer(sim.get_global(0, size), dtype=np.float32)
                data2 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_D, 0, 0,
                    size), dtype=np.float32)
                data3 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_W, 0, 0,
                    size), dtype=np.float32)
                ans = np.arange(len(data1)).astype(np.float32)
                np.testing.assert_array_equal(ans, data1)
                np.testing.assert_array_equal(ans, data2)
                np.testing.assert_array_equal(ans, data3)
            elif t == 'int8':
                data1 = np.frombuffer(sim.get_global(0, size), dtype=np.int8)
                data2 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_D, 0, 0,
                    size), dtype=np.int8)
                data3 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_W, 0, 0,
                    size), dtype=np.int8)
                ans = np.arange(len(data1)).astype(np.float32) * (127. / float(max_val))
                ans = np.rint(ans).astype(np.int8)

                np.testing.assert_array_equal(ans, data1)
                np.testing.assert_array_equal(ans, data2)
                np.testing.assert_array_equal(ans, data3)
            elif t == 'int16':
                data1 = np.frombuffer(sim.get_global(0, size), dtype=np.int16)
                data2 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_D, 0, 0,
                    size), dtype=np.int16)
                data3 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_W, 0, 0,
                    size), dtype=np.int16)
                ans = np.arange(len(data1)).astype(np.float32) * (32767. / float(max_val))
                ans = np.rint(ans).astype(np.int16)
                np.testing.assert_array_equal(ans, data1)
                np.testing.assert_array_equal(ans, data2)
                np.testing.assert_array_equal(ans, data3)
            elif t == 'float16':
                data1 = np.frombuffer(sim.get_global(0, size), dtype=np.float16)
                data2 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_D, 0, 0,
                    size), dtype=np.float16)
                data3 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_W, 0, 0,
                    size), dtype=np.float16)
                ans = np.arange(len(data1)).astype(np.float16)
                np.testing.assert_allclose(ans, data1, rtol=1e-3, atol=1e-3)
                np.testing.assert_allclose(ans, data2, rtol=1e-3, atol=1e-3)
                np.testing.assert_allclose(ans, data3, rtol=1e-3, atol=1e-3)
            elif t == 'bfloat16':
                data1 = np.frombuffer(sim.get_global(0, size), dtype=np.int16)
                # x = data1[1]
                # print x, bf16_to_fp32(x)
                data1 = [bf16_to_fp32(x) for x in data1]
                data2 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_D, 0, 0,
                    size), dtype=np.int16)
                data2 = [bf16_to_fp32(x) for x in data2]
                data3 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_W, 0, 0,
                    size), dtype=np.int16)
                data3 = [bf16_to_fp32(x) for x in data3]
                ans = np.arange(len(data1)).astype(np.float32)
                np.testing.assert_allclose(ans, data1, rtol=1e-2, atol=1e-3)
                np.testing.assert_allclose(ans, data2, rtol=1e-2, atol=1e-3)
                np.testing.assert_allclose(ans, data3, rtol=1e-2, atol=1e-3)
            elif t == 'int31':
                data1_lo = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_D, 0, 0,
                    size), dtype=np.int16)
                data1_hi = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_D, 0,
                    int31_l2_hi, size), dtype=np.int16)
                data1 = (data1_hi.astype(np.int32) << 15) + data1_lo.astype(np.int32)

                data2_lo = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_W, 0, 0,
                    size), dtype=np.int16)
                data2_hi = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_W, 0,
                    int31_l2_hi, size), dtype=np.int16)
                data2 = (data2_hi.astype(np.int32) << 15) + data2_lo.astype(np.int32)

                data3_lo = np.frombuffer(sim.get_global(0, size), dtype=np.int16)
                data3_hi = np.frombuffer(sim.get_global(int31_l2_hi, size), dtype=np.int16)
                data3 = (data3_hi.astype(np.int32) << 15) + data3_lo.astype(np.int32)

                src_data = np.arange(len(data1_lo)).astype(np.float32)
                ans = src_data / np.float32(max_val) * np.float32(1073741823)
                ans = np.rint(ans)
                ans = ans.astype(np.int32)
                ans_hi = ans >> 15
                ans_lo = ans & 0x7FFF
                print "diff of hi"
                stat_diff(data1_hi, ans_hi)
                print "diff of lo"
                stat_diff(data1_lo, ans_lo)
                print "diff of all"
                stat_diff(data1, ans)
                np.testing.assert_allclose(ans, data3, 1e-5, 1e-5)
                np.testing.assert_allclose(ans, data2, 1e-5, 1e-5)
                np.testing.assert_allclose(ans, data1, 1e-5, 1e-5)

    def test_readL2R_2(self):
        max_val = 16383
        asm = '''
        rset zero
        core_id id
        bne id, zero, end
        addi header, zero, {header}
        fix2float header, header
        # L2R => HBM
        addi dst_desc, zero, 0b000{type}0000
        addi src_desc, zero, 0b0100000000
        addi lk, zero, 7
        lock zero, lk, zero, 0
        l2_mov_cfg dst_desc, src_desc, header
        addi len, zero, 4096;           # 4KB per bank
        muli len, len, 16;              # 16 banks per core
        subi len, len, 64;              # skip first row
        addi src, zero, {l2r_offset};   # logical address on L2R
        mul dst, dst, id
        l2_mov dst, src, len, 0
        unlock zero, lk, zero, 0
        end:
        ret
        '''.format(header=max_val, type='000', l2r_offset=1 * 16 * 4)
        cluster_id = 0
        sim = None; sim = CDNNSimulator()
        sim.set_asm_source(code=asm)
        for i in range(16):
            start = i % 16
            stop = start + 16 * 1024
            data = np.arange(start, stop, 16).astype(np.float32).tobytes()
            sim.set_sram_data_bybank(cluster_id, sim.L2_SRAM_R, i, 0, data)
        sim.run()
        size = 16 * 1024 * 4 - 1 * 16 * 4
        data1 = np.frombuffer(sim.get_global(0, size), dtype=np.float32)
        ans = np.arange(16, len(data1) + 16).astype(np.float32)
        np.testing.assert_array_equal(ans, data1)

    def test_readHBM(self):
        type2info = {
            'f32-f32': ('float32', '000', 4, 'float32', '000', 4),
            'f32-i16': ('float32', '000', 4, 'int16', '010', 2),
            'f32-i8': ('float32', '000', 4, 'int8', '100', 1),
            'f16-i16': ('float16', '001', 2, 'int16', '010', 2),
            'f16-i8': ('float16', '001', 2, 'int8', '100', 1),
            'bf16-i16': ('bfloat16', '011', 2, 'int16', '010', 2),
            'bf16-i8': ('bfloat16', '011', 2, 'int8', '100', 1),
            'i16-i16': ('int16', '010', 2, 'int16', '010', 2),
            'i8-i8': ('int8', '100', 1, 'int8', '100', 1),
            'f16-f32': ('float16', '001', 2, 'float32', '000', 4),
            'bf16-f32': ('bfloat16', '011', 2, 'float32', '000', 4),
        }
        for t in ('f32-f32', 'f32-i16', 'f32-i8', 'f16-i16',
            'f16-i8', 'bf16-i16', 'bf16-i8', 'i16-i16', 'i8-i8',
            'f16-f32', 'bf16-f32'):
            print "HBM test case ", t
            max_val = 16383
            info = type2info[t]
            length = 1024 * info[2]
            asm = '''
            rset zero
            core_id id
            bne id, zero, end
            addi header, zero, {header}
            fix2float header, header
            addi len, zero, {length}
            muli len, len, 16
            # HBM => L2D
            addi dst_desc, zero, 0b110{dst_type}0000
            addi src_desc, zero, 0b000{src_type}0000
            addi lk, zero, 6
            lock zero, lk, zero, 0
            l2_mov_cfg dst_desc, src_desc, header
            l2_mov zero, zero, len, 0b01
            unlock zero, lk, zero, 0
            # HBM => L2W
            addi dst_desc, zero, 0b100{dst_type}0000
            addi lk, zero, 5
            lock zero, lk, zero, 0
            l2_mov_cfg dst_desc, src_desc, header
            l2_mov zero, zero, len, 0b01
            unlock zero, lk, zero, 0
            end:
            ret
            '''.format(header=max_val, src_type=info[1],
                    dst_type=info[4], length=length)
            sim = None; sim = CDNNSimulator()
            sim.set_asm_source(code=asm)
            if info[0] == 'float32':
                data = np.arange(16 * 1024).astype(np.float32).tobytes()
            elif info[0] == 'float16':
                data = np.arange(16 * 1024).astype(np.float16).tobytes()
            elif info[0] == 'bfloat16':
                data = np.arange(16 * 1024).astype(np.float32)
                data = [fp32_to_bf16(x) for x in data]
                data = np.array(data).astype(np.uint16).tobytes()
            elif info[0] == 'int16':
                data = np.arange(16 * 1024).astype(np.int16).tobytes()
            elif info[0] == 'int8':
                data = np.arange(16 * 1024).astype(np.int8).tobytes()
            sim.set_global(0, data)
            sim.run()
            size = 16 * 1024 * info[5]
            cluster_id = 0
            if t == 'f32-f32':
                data1 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_D, 0, 0,
                    size), dtype=np.float32)
                data2 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_W, 0, 0,
                    size), dtype=np.float32)
                ans = np.arange(len(data1)).astype(np.float32)
                np.testing.assert_array_equal(ans, data1)
                np.testing.assert_array_equal(ans, data2)
            elif t == 'f32-i8':
                data1 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_D, 0, 0,
                    size), dtype=np.int8)
                data2 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_W, 0, 0,
                    size), dtype=np.int8)
                ans = np.arange(len(data1)).astype(np.float32) * (127. / float(max_val))
                ans = np.rint(ans).astype(np.int8)
                np.testing.assert_array_equal(ans, data1)
                np.testing.assert_array_equal(ans, data2)
            elif t == 'f32-i16':
                data1 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_D, 0, 0,
                    size), dtype=np.int16)
                data2 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_W, 0, 0,
                    size), dtype=np.int16)
                ans = np.arange(len(data1)).astype(np.float32) * (32767. / float(max_val))
                ans = np.rint(ans).astype(np.int16)
                np.testing.assert_array_equal(ans, data1)
                np.testing.assert_array_equal(ans, data2)
            elif t == 'f16-i16':
                data1 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_D, 0, 0,
                    size), dtype=np.int16)
                data2 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_W, 0, 0,
                    size), dtype=np.int16)
                ans = np.arange(len(data1)).astype(np.float16) * (32767. / float(max_val))
                np.testing.assert_allclose(ans, data1, 1e-3, 1e-3)
                np.testing.assert_allclose(ans, data2, 1e-3, 1e-3)
            elif t == 'f16-i8':
                data1 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_D, 0, 0,
                    size), dtype=np.int8)
                data2 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_W, 0, 0,
                    size), dtype=np.int8)
                ans = np.arange(len(data1)).astype(np.float16).astype(np.float32) * (127. / float(max_val))
                ans = np.rint(ans).astype(np.int8)
                np.testing.assert_allclose(ans, data1, 1e-3, 1e-3)
                np.testing.assert_allclose(ans, data2, 1e-3, 1e-3)
            elif t == 'bf16-i16':
                data1 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_D, 0, 0,
                    size), dtype=np.int16)
                data2 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_W, 0, 0,
                    size), dtype=np.int16)
                ans = np.arange(len(data1)).astype(np.float32)
                ans = [fp32_to_bf16(x) for x in ans]
                ans = [bf16_to_fp32(x) for x in ans]
                ans = np.array(ans).astype(np.float32)
                ans = ans * (32767. / float(max_val))
                ans = np.rint(ans).astype(np.int16)
                np.testing.assert_allclose(ans, data1, 1e-3, 1e-3)
                np.testing.assert_allclose(ans, data2, 1e-3, 1e-3)
            elif t == 'bf16-i8':
                data1 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_D, 0, 0,
                    size), dtype=np.int8)
                data2 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_W, 0, 0,
                    size), dtype=np.int8)
                ans = np.arange(len(data1)).astype(np.float32)
                ans = [fp32_to_bf16(x) for x in ans]
                ans = [bf16_to_fp32(x) for x in ans]
                ans = np.array(ans).astype(np.float32)
                ans = ans * (127. / float(max_val))
                ans = np.rint(ans).astype(np.int8)
                np.testing.assert_allclose(ans, data1, 1e-3, 1e-3)
                np.testing.assert_allclose(ans, data2, 1e-3, 1e-3)
            elif t == 'i16-i16':
                data1 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_D, 0, 0,
                    size), dtype=np.int16)
                data2 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_W, 0, 0,
                    size), dtype=np.int16)
                ans = np.arange(len(data1)).astype(np.int16)
                np.testing.assert_array_equal(ans, data1)
                np.testing.assert_array_equal(ans, data2)
            elif t == 'i8-i8':
                data1 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_D, 0, 0,
                    size), dtype=np.int8)
                data2 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_W, 0, 0,
                    size), dtype=np.int8)
                ans = np.arange(len(data1)).astype(np.int8)
                np.testing.assert_array_equal(ans, data1)
                np.testing.assert_array_equal(ans, data2)
            elif t == 'f16-f32':
                data1 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_D, 0, 0,
                    size), dtype=np.float32)
                data2 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_W, 0, 0,
                    size), dtype=np.float32)
                ans = np.arange(len(data1)).astype(np.float32)
                np.testing.assert_allclose(ans, data1, 1e-3, 1e-4)
                np.testing.assert_allclose(ans, data2, 1e-3, 1e-4)
            elif t == 'bf16-f32':
                data1 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_D, 0, 0,
                    size), dtype=np.float32)
                data2 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_W, 0, 0,
                    size), dtype=np.float32)
                ans = np.arange(len(data1)).astype(np.float32)
                np.testing.assert_allclose(ans, data1, 1e-2, 1e-3)
                np.testing.assert_allclose(ans, data2, 1e-2, 1e-3)

    def test_readL2D(self):
        type2info = {
            'f32-f32': ('float32', '000', 4, 'float32', '000', 4),
            'f32-i16': ('float32', '000', 4, 'int16', '010', 2),
            'f32-i8': ('float32', '000', 4, 'int8', '100', 1),
            'f16-i16': ('float16', '001', 2, 'int16', '010', 2),
            'f16-i8': ('float16', '001', 2, 'int8', '100', 1),
            'bf16-i16': ('bfloat16', '011', 2, 'int16', '010', 2),
            'bf16-i8': ('bfloat16', '011', 2, 'int8', '100', 1),
            'i16-i16': ('int16', '010', 2, 'int16', '010', 2),
            'i8-i8': ('int8', '100', 1, 'int8', '100', 1),
        }
        for t in ('f32-f32', 'f32-i16', 'f32-i8', 'f16-i16',
            'f16-i8', 'bf16-i16', 'bf16-i8', 'i16-i16', 'i8-i8',):
            print "L2D test case ", t
            max_val = 16383
            info = type2info[t]
            length = 1024 * info[2]
            asm = '''
            rset zero
            core_id id
            bne id, zero, end
            addi header, zero, {header}
            fix2float header, header
            addi len, zero, {length}
            muli len, len, 16
            # L2D => HBM
            addi dst_desc, zero, 0b000{dst_type}0000
            addi src_desc, zero, 0b110{src_type}0000
            addi lk, zero, 7
            lock zero, lk, zero, 0
            l2_mov_cfg dst_desc, src_desc, header
            l2_mov zero, zero, len, 0
            unlock zero, lk, zero, 0
            # L2D => L2W
            addi dst_desc, zero, 0b100{dst_type}0000
            lock zero, lk, zero, 0
            l2_mov_cfg dst_desc, src_desc, header
            l2_mov zero, zero, len, 0
            unlock zero, lk, zero, 0
            # L2D => L2D
            addi dst_desc, zero, 0b110{dst_type}0000
            lock zero, lk, zero, 0
            l2_mov_cfg dst_desc, src_desc, header
            l2_mov zero, zero, len, 0
            unlock zero, lk, zero, 0
            end:
            ret
            '''.format(header=max_val, src_type=info[1],
                    dst_type=info[4], length=length)
            sim = None
            sim = CDNNSimulator()
            sim.set_asm_source(code=asm)
            if info[0] == 'float32':
                data = np.arange(16 * 1024).astype(np.float32).tobytes()
            elif info[0] == 'float16':
                data = np.arange(16 * 1024).astype(np.float16).tobytes()
            elif info[0] == 'bfloat16':
                data = np.arange(16 * 1024).astype(np.float32)
                data = [fp32_to_bf16(x) for x in data]
                data = np.array(data).astype(np.uint16).tobytes()
            elif info[0] == 'int16':
                data = np.arange(16 * 1024).astype(np.int16).tobytes()
            elif info[0] == 'int8':
                data = np.arange(16 * 1024).astype(np.int8).tobytes()
            #sim.set_global(0, data)
            cluster_id = 0
            sim.set_sram_data_bybank(cluster_id, sim.L2_SRAM_D, 0, 0, data)
            sim.run()
            size = 16 * 1024 * info[5]
            if t == 'f32-f32':
                data1 = np.frombuffer(sim.get_global(0, size), dtype=np.float32)
                data2 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_W, 0, 0,
                    size), dtype=np.float32)
                data3 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_D, 0, 0,
                    size), dtype=np.float32)
                ans = np.arange(len(data1)).astype(np.float32)
                np.testing.assert_array_equal(ans, data1)
                np.testing.assert_array_equal(ans, data2)
                np.testing.assert_array_equal(ans, data3)
            elif t == 'f32-i8':
                data1 = np.frombuffer(sim.get_global(0, size), dtype=np.int8)
                data2 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_W, 0, 0,
                    size), dtype=np.int8)
                data3 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_D, 0, 0,
                    size), dtype=np.int8)
                ans = np.arange(len(data1)).astype(np.float32) * (127. / float(max_val))
                ans = np.rint(ans).astype(np.int8)
                np.testing.assert_array_equal(ans, data1)
                np.testing.assert_array_equal(ans, data2)
                np.testing.assert_array_equal(ans, data3)
            elif t == 'f32-i16':
                data1 = np.frombuffer(sim.get_global(0, size), dtype=np.int16)
                data2 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_W, 0, 0,
                    size), dtype=np.int16)
                data3 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_D, 0, 0,
                    size), dtype=np.int16)
                ans = np.arange(len(data1)).astype(np.float32) * (32767. / float(max_val))
                ans = np.rint(ans).astype(np.int16)
                np.testing.assert_array_equal(ans, data1)
                np.testing.assert_array_equal(ans, data2)
                np.testing.assert_array_equal(ans, data3)
            elif t == 'f16-i16':
                data1 = np.frombuffer(sim.get_global(0, size), dtype=np.int16)
                data2 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_W, 0, 0,
                    size), dtype=np.int16)
                data3 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_D, 0, 0,
                    size), dtype=np.int16)
                ans = np.arange(len(data1)).astype(np.float16) * (32767. / float(max_val))
                np.testing.assert_allclose(ans, data1, 1e-3, 1e-3)
                np.testing.assert_allclose(ans, data2, 1e-3, 1e-3)
                np.testing.assert_allclose(ans, data3, 1e-3, 1e-3)
            elif t == 'f16-i8':
                data1 = np.frombuffer(sim.get_global(0, size), dtype=np.int8)
                data2 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_W, 0, 0,
                    size), dtype=np.int8)
                data3 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_D, 0, 0,
                    size), dtype=np.int8)
                ans = np.arange(len(data1)).astype(np.float16).astype(np.float32) \
                        * (127. / float(max_val))
                ans = np.rint(ans).astype(np.int8)
                np.testing.assert_allclose(ans, data1, 1e-3, 1e-3)
                np.testing.assert_allclose(ans, data2, 1e-3, 1e-3)
                np.testing.assert_allclose(ans, data3, 1e-3, 1e-3)
            elif t == 'bf16-i16':
                data1 = np.frombuffer(sim.get_global(0, size), dtype=np.int16)
                data2 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_W, 0, 0,
                    size), dtype=np.int16)
                data3 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_D, 0, 0,
                    size), dtype=np.int16)
                ans = np.arange(len(data1)).astype(np.float32)
                ans = [fp32_to_bf16(x) for x in ans]
                ans = [bf16_to_fp32(x) for x in ans]
                ans = np.array(ans).astype(np.float32)
                ans = ans * (32767. / float(max_val))
                ans = np.rint(ans).astype(np.int16)
                np.testing.assert_allclose(ans, data1, 1e-3, 1e-3)
                np.testing.assert_allclose(ans, data2, 1e-3, 1e-3)
                np.testing.assert_allclose(ans, data3, 1e-3, 1e-3)
            elif t == 'bf16-i8':
                data1 = np.frombuffer(sim.get_global(0, size), dtype=np.int8)
                data2 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_W, 0, 0,
                    size), dtype=np.int8)
                data3 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_D, 0, 0,
                    size), dtype=np.int8)
                ans = np.arange(len(data1)).astype(np.float32)
                ans = [fp32_to_bf16(x) for x in ans]
                ans = [bf16_to_fp32(x) for x in ans]
                ans = np.array(ans).astype(np.float32)
                ans = ans * (127. / float(max_val))
                ans = np.rint(ans).astype(np.int8)
                np.testing.assert_allclose(ans, data1, 1e-3, 1e-3)
                np.testing.assert_allclose(ans, data2, 1e-3, 1e-3)
                np.testing.assert_allclose(ans, data3, 1e-3, 1e-3)
            elif t == 'i16-i16':
                data1 = np.frombuffer(sim.get_global(0, size), dtype=np.int16)
                data2 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_W, 0, 0,
                    size), dtype=np.int16)
                data3 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_D, 0, 0,
                    size), dtype=np.int16)
                ans = np.arange(len(data1)).astype(np.int16)
                np.testing.assert_array_equal(ans, data1)
                np.testing.assert_array_equal(ans, data2)
                np.testing.assert_array_equal(ans, data3)
            elif t == 'i8-i8':
                data1 = np.frombuffer(sim.get_global(0, size), dtype=np.int8)
                data2 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_W, 0, 0,
                    size), dtype=np.int8)
                data3 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_D, 0, 0,
                    size), dtype=np.int8)
                ans = np.arange(len(data1)).astype(np.int8)
                np.testing.assert_array_equal(ans, data1)
                np.testing.assert_array_equal(ans, data2)
                np.testing.assert_array_equal(ans, data3)

    def test_2D_readL2R(self):
        type2info = {
            'float32': ('000', 4),
            'int8': ('100', 1),
            'int16': ('010', 2),
            'float16': ('001', 2),
            'bfloat16': ('011', 2),
        }
        for t in ('int8', 'int16', 'float16', 'bfloat16', 'float32'):
            max_val = 16383
            info = type2info[t]
            loop = 200 # should be LE to 256, as only 256 * 64 number has been inited in src
            length = 33
            src_off = 64
            dst_off = length + 5
            asm = '''
            rset zero
            core_id id
            bne id, zero, end
            addi header, zero, {header}
            fix2float header, header
            addi src_off, zero, {src_off}
            addi loop, zero, {loop}
            addi dst_off, zero, {dst_off}
            l2_mov_2d_cfg dst_off, src_off, loop
            # L2R => HBM
            addi dst_desc, zero, 0b000{type}0000
            addi src_desc, zero, 0b0100000000
            addi lk, zero, 7
            lock zero, lk, zero, 0
            l2_mov_cfg dst_desc, src_desc, header
            addi len, zero, {length}
            addi src, zero, 0;              # logical address on L2R
            addi dst, zero, 1024;           # compute absolute offset on GM
            muli dst, dst, 16
            muli dst, dst, {data_size}
            mul dst, dst, id
            l2_mov dst, src, len, 0b100
            unlock zero, lk, zero, 0
            # L2R => L2D
            addi dst_desc, zero, 0b110{type}0000
            lock zero, lk, zero, 0
            l2_mov_cfg dst_desc, src_desc, header
            l2_mov zero, src, len, 0b100
            unlock zero, lk, zero, 0
            # L2R => L2W
            addi dst_desc, zero, 0b100{type}0000
            lock zero, lk, zero, 0
            l2_mov_cfg dst_desc, src_desc, header
            l2_mov zero, src, len, 0b100
            unlock zero, lk, zero, 0
            end:
            ret
            '''.format(header=max_val, type=info[0], data_size=info[1], length=length * 4,
                    src_off=src_off * 4, dst_off=dst_off * info[1], loop=loop)
            sim = None
            sim = CDNNSimulator()
            sim.set_asm_source(code=asm)
            cluster_id = 0

            size = src_off * loop * info[1]
            init_hbm = np.zeros(size).astype(np.float32).tobytes()
            sim.set_global(0, init_hbm)
            reset_sram(sim)

            # generate data in special format, to read "0...16383" back from L2R
            for i in range(16):
                start = i % 16
                stop = start + 16 * 1024
                data = np.arange(start, stop, 16).astype(np.float32).tobytes()
                sim.set_sram_data_bybank(cluster_id, sim.L2_SRAM_R, i, 0, data)
            sim.run()
            if t == 'float32':
                data1 = np.frombuffer(sim.get_global(0, size), dtype=np.float32)
                data2 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_D, 0, 0,
                    size), dtype=np.float32)
                data3 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_W, 0, 0,
                    size), dtype=np.float32)
                org = np.arange(len(data1)).astype(np.float32)
                ans = np.zeros(len(data1)).astype(np.float32)
                for l in range(loop):
                    ans[l * dst_off: l * dst_off + length] = org[l * src_off: l * src_off + length]
                np.testing.assert_array_equal(ans, data1)
                np.testing.assert_array_equal(ans, data2)
                np.testing.assert_array_equal(ans, data3)
            elif t == 'int8':
                data1 = np.frombuffer(sim.get_global(0, size), dtype=np.int8)
                data2 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_D, 0, 0,
                    size), dtype=np.int8)
                data3 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_W, 0, 0,
                    size), dtype=np.int8)
                org = np.arange(len(data1)).astype(np.float32) * (127. / float(max_val))
                org = np.rint(org).astype(np.int8)
                ans = np.zeros(len(org)).astype(np.int8)
                for l in range(loop):
                    ans[l * dst_off: l * dst_off + length] = org[l * src_off: l * src_off + length]
                np.testing.assert_array_equal(ans, data1)
                np.testing.assert_array_equal(ans, data2)
                np.testing.assert_array_equal(ans, data3)
            elif t == 'int16':
                data1 = np.frombuffer(sim.get_global(0, size), dtype=np.int16)
                data2 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_D, 0, 0,
                    size), dtype=np.int16)
                data3 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_W, 0, 0,
                    size), dtype=np.int16)
                org = np.arange(len(data1)).astype(np.float32) * (32767. / float(max_val))
                org = np.rint(org).astype(np.int16)
                ans = np.zeros(len(org)).astype(np.int16)
                for l in range(loop):
                    ans[l * dst_off: l * dst_off + length] = org[l * src_off: l * src_off + length]
                np.testing.assert_array_equal(ans, data1)
                np.testing.assert_array_equal(ans, data2)
                np.testing.assert_array_equal(ans, data3)
            elif t == 'float16':
                data1 = np.frombuffer(sim.get_global(0, size), dtype=np.float16)
                data2 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_D, 0, 0,
                    size), dtype=np.float16)
                data3 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_W, 0, 0,
                    size), dtype=np.float16)
                org = np.arange(len(data1)).astype(np.float16)
                ans = np.zeros(len(org)).astype(np.float16)
                for l in range(loop):
                    ans[l * dst_off: l * dst_off + length] = org[l * src_off: l * src_off + length]
                np.testing.assert_allclose(ans, data1, rtol=1e-3, atol=1e-3)
                np.testing.assert_allclose(ans, data2, rtol=1e-3, atol=1e-3)
                np.testing.assert_allclose(ans, data3, rtol=1e-3, atol=1e-3)
            elif t == 'bfloat16':
                data1 = np.frombuffer(sim.get_global(0, size), dtype=np.int16)
                # x = data1[1]
                # print x, bf16_to_fp32(x)
                data1 = [bf16_to_fp32(x) for x in data1]
                data2 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_D, 0, 0,
                    size), dtype=np.int16)
                data2 = [bf16_to_fp32(x) for x in data2]
                data3 = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_W, 0, 0,
                    size), dtype=np.int16)
                data3 = [bf16_to_fp32(x) for x in data3]
                org = np.arange(len(data1)).astype(np.float32)
                ans = np.zeros(len(org)).astype(np.float32)
                for l in range(loop):
                    ans[l * dst_off: l * dst_off + length] = org[l * src_off: l * src_off + length]
                np.testing.assert_allclose(ans, data1, rtol=1e-2, atol=1e-3)
                np.testing.assert_allclose(ans, data2, rtol=1e-2, atol=1e-3)
                np.testing.assert_allclose(ans, data3, rtol=1e-2, atol=1e-3)

    def test_Int31(self):
        return
        type2info = {
            'f32-i31': ('float32', '000', 4, 'int31', '101', 2),
        }
        def get_scale_parts(scale):
            """convert the float into int, and return the high 16 bits and low 16 bits to caller"""
            from struct import pack, unpack
            scstr = pack('f', scale)
            int_dqt = unpack('i', scstr)[0]
            s_hi=(int_dqt >> 16)
            s_lo=(int_dqt & 0xFFFF)
            return s_hi, s_lo

        for t in ('f32-i31',):
            print "DMA int31 test case ", t
            max_val = 16383876
            info = type2info[t]
            k = 10
            m = 16 # should not be changed
            n = m
            dma_length = m * k * info[2]
            # int31_l2_hi = align(dma_length, 16)
            int31_l2_hi = 400
            int31_l1_hi = bankoff2addr(0, k * info[5], bank_bits=4)
            f_scale = 1.0 * max_val / (2 ** 30 - 1) * max_val / (2 ** 30 - 1)
            lls_hi, lls_lo = get_scale_parts(f_scale)
            f_scale = f_scale * (2 ** 15)
            lhs_hi, lhs_lo = get_scale_parts(f_scale)
            f_scale = f_scale * (2 ** 15)
            hhs_hi, hhs_lo = get_scale_parts(f_scale)
            asm = '''
            rset zero
            core_id id
            bne id, zero, end
            ori dma_max, zero, {max_hi}
            slli dma_max, dma_max, 16
            ori dma_max, dma_max, {max_lo}
            fix2float dma_max, dma_max
            addi len, zero, {length}
            ori hi_off_tmp, zero, {l2_hi}
            slli l2_hi, hi_off_tmp, 12
            # HBM => L2D
            addi dst_desc, zero, 0b110{dst_type}0000
            or dst_desc, dst_desc, l2_hi
            addi src_desc, zero, 0b000{src_type}0000
            addi lk, zero, 6
            lock zero, lk, zero, 0
            l2_mov_cfg dst_desc, src_desc, dma_max
            l2_mov zero, zero, len, 0b01
            unlock zero, lk, zero, 0
            # HBM => L2W
            addi dst_desc, zero, 0b100{dst_type}0000
            or dst_desc, dst_desc, l2_hi
            addi lk, zero, 5
            lock zero, lk, zero, 0
            l2_mov_cfg dst_desc, src_desc, dma_max
            l2_mov zero, zero, len, 0b01
            unlock zero, lk, zero, 0
            addi len, zero, {k}
            addi ds_type, zero, 1 // int16
            addi blkx, zero, {m}
            addi l1_hi, zero, {l1_hi}
            addi out_bank, zero, 16
            addi lk, zero, 0
            lock zero, lk, zero, 0
            ds_cfg ds_type, zero, zero, 0
            ds_cfg blkx, zero, zero, 1
            ds_cfg out_bank, zero, zero, 11
            shuffle_coa len, zero, zero, 1
            shuffle_coa len, hi_off_tmp, l1_hi, 1
            unlock zero, lk, zero, 0
            addi lk, zero, 1
            lock zero, lk, zero, 0
            ds_cfg ds_type, zero, zero, 0
            ds_cfg blkx, zero, zero, 1
            ds_cfg out_bank, zero, zero, 11
            shuffle_coa len, zero, zero, 2
            shuffle_coa len, hi_off_tmp, l1_hi, 2
            unlock zero, lk, zero, 0
            addi m, zero, {m}
            addi k, zero, {k}
            addi stride, zero, 1
            addi lk, zero, 2
            lock zero, lk, zero, 0
            mm_cfg zero, k, m, 0
            ori dqsc, zero, {lls_hi}
            slli dqsc, dqsc, 16
            ori dqsc, dqsc, {lls_lo}
            mm_cfg zero, dqsc, dqsc, 1
            mm_cfg zero, stride, zero, 2
            mm zero, zero, zero, 0b00000
            ori dqsc, zero, {lhs_hi}
            slli dqsc, dqsc, 16
            ori dqsc, dqsc, {lhs_lo}
            mm_cfg zero, dqsc, dqsc, 1
            mm_acc zero, zero, l1_hi, 0b00000
            mm_acc zero, l1_hi, zero, 0b00000
            ori dqsc, zero, {hhs_hi}
            slli dqsc, dqsc, 16
            ori dqsc, dqsc, {hhs_lo}
            mm_cfg zero, dqsc, dqsc, 1
            mm_acc zero, l1_hi, l1_hi, 0b00000
            unlock zero, lk, zero, 0
            end:
            ret
            '''.format(max_hi=(max_val >> 16), max_lo=(max_val & 0xFFFF),
                    src_type=info[1], k=k, m=m, dst_type=info[4], length=dma_length,
                    l2_hi=int31_l2_hi, l1_hi=int31_l1_hi, lls_hi=lls_hi, lls_lo=lls_lo,
                    lhs_hi=lhs_hi, lhs_lo=lhs_lo, hhs_hi=hhs_hi, hhs_lo=hhs_lo)
            sim = None; sim = CDNNSimulator()
            sim.set_asm_source(code=asm)
            if info[0] == 'float32':
                src_data = (np.random.rand(m * k) - 0.5) * max_val
                src_data = src_data.astype(np.float32)
                # src_data[0] = 59328.90625
                # print src_data
                data = src_data.tobytes()
            sim.set_global(0, data)
            sim.run()
            # size = m * k * info[5]
            cluster_id = 0

            # data1_lo = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_D, 0, 0,
            #     size), dtype=np.int16)
            # data1_hi = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_D, 0,
            #     int31_l2_hi, size), dtype=np.int16)

            # data2_lo = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_W, 0, 0,
            #     size), dtype=np.int16)
            # data2_hi = np.frombuffer(sim.get_sram_data_bybank(cluster_id, sim.L2_SRAM_W, 0,
            #     int31_l2_hi, size), dtype=np.int16)

            # ans = src_data * (1073741823. / float(max_val))
            # ans = np.rint(ans)
            # ans = ans.astype(np.int32)
            # ans_hi = ans >> 15
            # ans_lo = ans & 0x7FFF
            # print ans_hi
            # print data1_hi
            # print data2_hi
            # print ans_lo
            # print data1_lo
            # print data2_lo
            # np.testing.assert_array_equal(ans_hi, data1_hi)
            # np.testing.assert_allclose(ans_lo, data1_lo, 1e-3, 1e-3)

            # l1d = []
            # l1d_h = []
            # l1w = []
            # l1w_h = []
            # for bank_id in range(16):
            #     ans = sim.get_sram_data_bybank(cluster_id, sim.L1_SRAM_D, bank_id, 0, k * ctypes.sizeof(ctypes.c_int16))
            #     y = np.frombuffer(ans, dtype=np.int16)
            #     l1d = np.hstack((l1d, y))
            #     ans = sim.get_sram_data_bybank(cluster_id, sim.L1_SRAM_D, bank_id, 2 * k, k * ctypes.sizeof(ctypes.c_int16))
            #     y = np.frombuffer(ans, dtype=np.int16)
            #     l1d_h = np.hstack((l1d_h, y))
            #     ans = sim.get_sram_data_bybank(cluster_id, sim.L1_SRAM_W, bank_id, 0, k * ctypes.sizeof(ctypes.c_int16))
            #     y = np.frombuffer(ans, dtype=np.int16)
            #     l1w = np.hstack((l1w, y))
            #     ans = sim.get_sram_data_bybank(cluster_id, sim.L1_SRAM_W, bank_id, 2 * k, k * ctypes.sizeof(ctypes.c_int16))
            #     y = np.frombuffer(ans, dtype=np.int16)
            #     l1w_h = np.hstack((l1w_h, y))

            # print l1d
            # print l1d.reshape(m, k)
            # print l1d_h.reshape(m, k)
            # print l1w.reshape(m, k)
            # print l1w_h.reshape(m, k)

            result = []
            for i in range(0, m):
                # print "bank " + str(i)
                ans = sim.get_sram_data_bybank(cluster_id,
                        sim.L1_SRAM_E, i, 0, m * ctypes.sizeof(ctypes.c_float))
                y = np.frombuffer(ans, dtype=np.float32)
                result = np.hstack((result, y))

            result = result.reshape(m, m).T.astype(np.float32)
            # print "result:"
            # print result

            src_data = src_data.reshape(k, m).astype(np.float32).T
            # print "src_data :"
            # print src_data
            # print "src_data.T :"
            # print src_data.T

            np_res = np.dot(src_data, src_data.T).astype(np.float32)
            # print "np_res:"
            # print np_res

            diff = abs(result - np_res)
            # print "diff:"
            # print diff
            max = diff.max()
            # print max
            idx = np.argmax(diff)
            idx = np.unravel_index(idx, diff.shape)
            # print idx
            # print result[idx], np_res[idx]

            np.set_printoptions(threshold=np.nan)
            # TODO: k = 10, Relative tolerance 1e-2, see no diff. k = 20, lots of diff
            np.testing.assert_allclose(result, np_res, 1e-2, 1e-4)

if __name__ == '__main__':
    unittest.main()
