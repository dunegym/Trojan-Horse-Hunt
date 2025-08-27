#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <gflags/gflags.h>
#include <assert.h>

DEFINE_string(output, "./mac_cmd_seq.dat", "mac_cmd_file");
DEFINE_string(mode, "random", "MAC mode int16, int8 or random");
DEFINE_int32(seed, 0, "random seed");

#define MAX_ITER 30

#define BANK_BITS 4
#define ALIGN_BITS 2

static uint32_t round_down(uint32_t n, uint32_t k) {
    return n / k * k;
}

static uint32_t round_up(uint32_t n, uint32_t k) {
    return (n + k - 1)/ k * k;
}

static uint32_t roundup_div(uint32_t n, uint32_t k) {
    return (n + k - 1)/ k;
}

uint32_t unified_hw_addr(uint32_t bank, uint32_t offset, int bank_bits = BANK_BITS,
                         int align_bits = ALIGN_BITS) {
    uint32_t off_lo = offset & ((1 << align_bits) - 1);
    uint32_t off_hi = (offset >> align_bits) << (bank_bits + align_bits);
    uint32_t bank_f = bank << align_bits;
    uint32_t addr = off_hi | bank_f | off_lo;
    return addr;
}
#define NBANK (16)
#define MAX_RANGE 10000
#define L1D_DEPTH (4096)
#define L1D_WIDTH (1024)
#define L1W_DEPTH (4096)
#define L1W_WIDTH (4096)

#define L1E_DEPTH (4096u)
void gen_mac_seq(std::ostream& out, bool is_int8) {
    uint32_t a_addr = 0;
    uint32_t b_addr = 0;
    uint32_t d_addr = 0;
    uint32_t m = 0;
    uint32_t k = 0;
    uint32_t a_off = 0;
    uint32_t b_off = 0;
    uint32_t d_off = 0;

    uint32_t int8_off = 0;

    float dequant_scale = 1.0f;

    const uint32_t l1d_bytes = L1D_WIDTH * L1D_DEPTH / 8 / NBANK;
    const uint32_t l1w_bytes = L1W_WIDTH * L1W_DEPTH / 8 / NBANK;

    dequant_scale = ((double)rand() / (double)RAND_MAX) * (double)(rand() % MAX_RANGE);

    if (dequant_scale == 0.0f) {
        dequant_scale = 1e-5;
    }

    uint32_t* p_scale_32bit_repr = (uint32_t*)(&dequant_scale);

    int stride = rand() % 7 + 1;
    int stride_sel = rand() % 10;
    if (stride_sel >= 6) {
        stride = rand() % (is_int8? 2047: 4095) + 1; // change to 12bit
    }

    unsigned int mac_types[] = {
        0b00000,
    };

    unsigned int int8_types[] = {
        0b10000,
        0b10001,
        0b10010,
        0b10011,
    };

    unsigned int mac_type = 0;
    int a_len = 0;
    int m_sel = rand() % 3; // limit m's range

    if (is_int8) {
        int m_range = (m_sel < 2)? 32: 2048;
        m = rand() % m_range + 1;
        // m * stride <= l1e_depth
        m = std::min(m, (L1E_DEPTH - 1) / stride / 2);
        // m / 16 * round_k << l1d_depth
        uint32_t round_k_max = (l1d_bytes / 2 - 1) / sizeof(int8_t) / roundup_div(m, 32) / 2;
        round_k_max = round_down(round_k_max, 16);
        k = rand() % round_k_max + 1;
        int round_k = round_up(k, 16);
        a_len = roundup_div(m, 32) * round_k * 2 * sizeof(int8_t);
        a_off = rand() % ((l1d_bytes / 2) - a_len);
        a_off = round_down(a_off, 32);
        b_off = rand() % ((l1w_bytes / 2) - round_k * sizeof(int8_t));
        b_off = round_down(b_off, 32);
        d_off = rand() % (L1E_DEPTH - 2 * m * stride);
        d_off = std::min(d_off, (uint32_t)(L1E_DEPTH - m * stride * 2));
        int8_off = m * stride;
        //std::cout << "int8_off = " << int8_off << std::endl;
        //std::cout << "d_off = " << d_off << std::endl;
        m = rand() % m + 1;
        k = rand() % k + 1;
        assert(m <= 2048);
        assert(k <= (l1d_bytes));
        assert(a_off >= 0);
        assert(a_off < l1d_bytes);
        assert(a_off + a_len < l1d_bytes);
        assert((int8_off + d_off) <= (L1E_DEPTH - m * stride));
        int idx = rand() % (sizeof(int8_types) / sizeof(int8_types[0]));
        mac_type = int8_types[idx];
    } else {
        int m_range = (m_sel < 2)? 16: 4096;
        m = rand() % m_range + 1;
        // m * stride <= l1e_depth
        m = std::min(m, (L1E_DEPTH - 1) / stride);
        // m / 16 * round_k << l1d_depth
        uint32_t round_k_max = (l1d_bytes - 1) / sizeof(int16_t) / roundup_div(m, 16);
        round_k_max = round_down(round_k_max, 16);
        k = rand() % round_k_max + 1;
        int round_k = round_up(k, 16);
        //std::cout << "m = " << m << std::endl;
        //std::cout << "round_k = " << round_k << std::endl;
        //std::cout << "m div = " << roundup_div(m, 16) << std::endl;
        a_len = roundup_div(m, 16) * round_k * sizeof(int16_t);
        //std::cout << "a_len = " << a_len << std::endl;
        a_off = rand() % ((l1d_bytes) - a_len);
        a_off = round_down(a_off, 32);
        b_off = rand() % ((l1w_bytes) - round_k * sizeof(int16_t));
        b_off = round_down(b_off, 32);
        d_off = rand() % (L1E_DEPTH - m * stride);
        d_off = std::min(d_off, (uint32_t)(L1E_DEPTH - m * stride));
        mac_type = mac_types[0];
        m = rand() % m + 1;
        k = rand() % k + 1;
        assert(m <= 4096);
        assert(k <= (l1d_bytes));
        assert(a_off >= 0);
        assert(a_off < l1d_bytes);
        assert(a_off + a_len < l1d_bytes);
    }
    //std::cout << "is_int8 = " << (is_int8? "True": "False") << std::endl;
    //std::cout << "m = " << m << std::endl;
    //std::cout << "k = " << k << std::endl;
    //std::cout << "a_len = " << a_len << std::endl;
    //std::cout << "l1d_bytes = " << l1d_bytes << std::endl;
    //std::cout << "stride = " << stride << std::endl;

    d_off *= 4;
    a_addr = unified_hw_addr(0, a_off, 4, 1);
    b_addr = unified_hw_addr(0, b_off, 4, 1);
    d_addr = unified_hw_addr(0, d_off, 4, 2);
    //std::cout << "addr_a = " << std::hex << a_addr << std::endl;
    //std::cout << "addr_b = " << std::hex << b_addr << std::endl;
    //std::cout << "addr_d = " << std::hex << d_addr << std::endl;
    //std::cout << "dequant_scale = " << dequant_scale << std::endl;

    assert(m > 0);
    assert(k > 0);
    assert(a_addr % 512 == 0);
    assert(b_addr % 512 == 0);

    assert(a_addr < L1D_DEPTH * L1D_WIDTH / 8);
    assert(b_addr < L1W_DEPTH * L1W_WIDTH / 8);
    assert(d_addr < L1E_DEPTH * 32 / 8 * 16);

    std::vector<std::string> mac_instrs{
        std::string("mm"),
        std::string("mm_acc"),
    };
    int idx = rand() % mac_instrs.size();
    int cmd_repeat = rand() % 3 + 1;

    out << "mm_cfg " << "rs=" << k << " rt=" << m << " rd=" << int8_off << " shamt=0" << std::endl;
    out << "mm_cfg " << "rs=" << *p_scale_32bit_repr << " rt=" << 0
        << " rd=" << 0 << " shamt=1" << std::endl;
    out << "mm_cfg " << "rs=" << stride << " rt=" << 0 << " rd=" << 0 << " shamt=2" << std::endl;
    for (int k = 0; k < cmd_repeat; ++k) {
        out << mac_instrs[idx] << " rs=" << a_addr << " rt=" << b_addr << " rd=" << d_addr
            << " shamt=" << mac_type << std::endl;
    }
}

int main(int argc, char** argv) {
    google::ParseCommandLineFlags(&argc, &argv, true);

    srand(FLAGS_seed);

    std::ofstream cmd_seq;
    cmd_seq.open(FLAGS_output);
    int cmd_loop = rand() % 3 + 1;

    if (FLAGS_mode == "int8") {
        for (int k = 0; k < cmd_loop; ++k) {
            gen_mac_seq(cmd_seq, true);
        }
    } else if (FLAGS_mode == "int16") {
        for (int k = 0; k < cmd_loop; ++k) {
            gen_mac_seq(cmd_seq, false);
        }
    } else if (FLAGS_mode == "random") {
        bool sel = (rand() % 2 == 0);
        for (int k = 0; k < cmd_loop; ++k) {
            gen_mac_seq(cmd_seq, sel);
        }
    }

    cmd_seq << "xfence" << std::endl;
    cmd_seq.close();
    return 0;
}
