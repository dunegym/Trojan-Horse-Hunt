#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <gflags/gflags.h>
#include <assert.h>

//#define DEBUG 1

static constexpr std::size_t kFpBankDepth = 2048;
static constexpr std::size_t kFpBankNum = 16;

static constexpr std::size_t kL2DBankSize = (2048 * 512 / 8);
static constexpr std::size_t kL2DBankNum = 1;
static constexpr std::size_t kL2WBankSize = (2048 * 512 / 8);
static constexpr std::size_t kL2WBankNum = 1;
static constexpr std::size_t kL2EBankSize = (kFpBankDepth * 32 / 8);
static constexpr std::size_t kL2EBankBits = 4;
static constexpr std::size_t kL2EBankNum = kFpBankNum;
static constexpr std::size_t kL2RBankSize = (kFpBankDepth * 32 / 8);
static constexpr std::size_t kL2RBankNum = kFpBankNum;

static constexpr unsigned int kL1DBankBits = 4;
static constexpr std::size_t kL1DBankSize = (2 * 1024 * 128 / 8);
static constexpr std::size_t kL1DBankNum = (1 << kL1DBankBits);
static constexpr unsigned int kL1WBankBits = 4;
static constexpr std::size_t kL1WBankSize = (2 * 4096 * 128 / 8);
static constexpr std::size_t kL1WBankNum = (1 << kL1WBankBits);
static constexpr unsigned int kL1EBankBits = 4;
static constexpr std::size_t kL1EBankSize = (4096 * 32 / 8);
static constexpr std::size_t kL1EBankValidSize = ((4096 - 4) * 32 / 8); //reserve 4 items
static constexpr std::size_t kL1EBankNum = kFpBankNum;
static constexpr std::size_t kL1EBankDepth = 4096; //bit width:32bits 4B float
static constexpr std::size_t kL1EBankValidDepth = 4096 - 4; //reserve 4 items; bit width:32bits 4B float

static constexpr unsigned int BANK_BITS = 4;
static constexpr unsigned int ALIGN_BITS = 2;
static constexpr unsigned int MAX_RANGE = 10000;
static constexpr unsigned int CMD_TIMES = 10;

static constexpr unsigned int kEWTanhTableLengh = 512;
static constexpr float kEWTanhTableInterval = 1.0f / 64;

DEFINE_string(func, "dsmadd", "ew function dsmadd or dsmul or dscmpnsel or sspooling or ssvsum or ssresize or mix or fussion");
DEFINE_string(output, "./sd_cdnn_ew_cmd_seq.dat", "ew_cmd_file");
DEFINE_int32(seed, 0, "random seed");
DEFINE_string(l1e_file, "./l1e_sram_init.dat", "l1e sram image file");
DEFINE_int32(limit, MAX_RANGE, "abs_max of l1e data");

int roundup(int n, int k) {
    return ((n + k - 1) / k) * k;
}

uint32_t unified_hw_addr(uint32_t bank, uint32_t offset, int bank_bits=BANK_BITS, int align_bits=ALIGN_BITS) {
    uint32_t off_lo = offset & ((1 << align_bits) - 1);
    uint32_t off_hi = (offset >> align_bits) << (bank_bits + align_bits);
    uint32_t bank_f = bank << align_bits;
    uint32_t addr = off_hi | bank_f | off_lo;
    return addr;
}

/* rand a float in [-range, range]*/
static float rand_float(unsigned long range) {
    float fv = ((double)rand() / (double)RAND_MAX -  0.5) * 2 * (double)range;
    return fv;
}

void gen_tanh_tab(uint32_t length, float interval, std::vector<float> &tab) {
    tab.resize(length * 2);
    for (int i = 0; i < length; i++) {
        // use double to get better accuracy
        double x0 = i * interval;
        double x1 = (i + 1) * interval;
        double y0 = tanh(x0);
        double y1 = tanh(x1);
        if (i == length - 2) {
            y1 = 1.0;
        }
        double double_k = (y1 - y0) / (x1 - x0);
        double double_b = y0 - double_k * x0;
        float k = double_k;
        float b = double_b;
        if (i == length - 1) {
            k = 0.0f;
            b = 1.0f;
        }
        tab[i] = k;
        tab[length + i] = b;
    }
}

void gen_sqr_tab(uint32_t length, float interval, std::vector<float> &tab) {
    tab.resize(length * 2);
    for (int i = 0; i < length; i++) {
        // use double to get better accuracy
        double x0 = i * interval;
        double x1 = (i + 1) * interval;
        double y0 = x0 * x0;
        double y1 = x1 * x1;

        double double_k = (y1 - y0) / (x1 - x0);
        double double_b = y0 - double_k * x0;
        float k = double_k;
        float b = double_b;

        tab[i] = k;
        tab[length + i] = b;
    }
}

void gen_double_side_tab(uint32_t length, float interval, std::vector<float> &tab) {
    tab.resize(length * 2);
    int32_t half_len = (int32_t)length / 2;
    for (int i = 0; i < length; i++) {
        // use double to get better accuracy
        double x0 = (i - half_len) * interval;
        double x1 = (i - half_len + 1) * interval;
        double y0 = exp(x0 * -1.0);
        double y1 = exp(x1 * -1.0);

        double double_k = (y1 - y0) / (x1 - x0);
        double double_b = y0 - double_k * x0;
        float k = double_k;
        float b = double_b;

        tab[i] = k;
        tab[length + i] = b;
    }
}

void gen_nexp_tab(uint32_t length, float interval, std::vector<float> &tab) {
    tab.resize(length * 2);

    for (int i = 0; i < length; i++) {
        // use double to get better accuracy
        double x0 = -1.0 * i * interval;
        double x1 = -1.0 * (i + 1) * interval;
        double y0 = exp(x0);
        double y1 = exp(x1);

        double double_k = (y1 - y0) / (x1 - x0);
        double double_b = y0 - double_k * x0;
        float k = double_k;
        float b = double_b;

        if (i == length - 1) {
            k = 0.0f;
            b = 0.0f;
        }

        tab[i] = k;
        tab[length + i] = b;
    }
}

/* if act_type != "null", put the activation table in each bank of l1e start from act_offset */
static void gen_l1e(int act_offset, std::string act_type, uint32_t precision) {
    std::ofstream out;
    std::vector<float> table;

    int32_t l1e_size = (kL1EBankNum * kL1EBankSize / 4);
    int32_t l1e_per_row_num = 16;
    int32_t tab_start = kL1EBankSize / 4 + 1;
    int32_t tab_stop = -1;

    float factor = 1.0;
    switch (precision) {
        case 0:
            factor = factor * 2;
            break;
        case 2:
            factor = factor / 2;
            break;
        default:
        case 1:
            break;
    }

    if (act_type == "tanh") {
        gen_tanh_tab(kEWTanhTableLengh, factor * kEWTanhTableInterval, table);
        tab_start = act_offset;
        tab_stop = act_offset + 1024;
    }
    else if (act_type == "nexp") {
        gen_nexp_tab(kEWTanhTableLengh, factor * kEWTanhTableInterval, table);
        tab_start = act_offset;
        tab_stop = act_offset + 1024;
    }
    else if (act_type == "sqr") {
        gen_sqr_tab(kEWTanhTableLengh, factor * kEWTanhTableInterval, table);
        tab_start = act_offset;
        tab_stop = act_offset + 1024;
    }
    else if (act_type == "double-side") {
        gen_double_side_tab(kEWTanhTableLengh, factor * kEWTanhTableInterval, table);
        tab_start = act_offset;
        tab_stop = act_offset + 1024;
    }

    out.open(FLAGS_l1e_file);

    unsigned long maxv = FLAGS_limit;

    for (int i = 0; i < (l1e_size / l1e_per_row_num); i++) {
        float a = 0.0;
        for (int j = 0; j < l1e_per_row_num; j++) {
            if ((i >= tab_start) && (i < tab_stop)) {
                a = table[i - tab_start];
            }
            else {
                a = rand_float(maxv);
            }
            out << std::setw(8) << std::hex << std::setfill('0') << *((uint32_t *)&a);
        }
        out << std::endl;
    }
    out.close();
}

void gen_ew_tab_cmd(std::ostream &out, uint32_t active_type) {
    std::string tab_type;
    uint32_t activ_mode = 0;
    uint32_t precision = rand() % 3;
    if ((active_type == 0) || (active_type == 1)) {
        tab_type = "none";
        return;
    }
    else if ((active_type == 2) || (active_type == 3)) {
        tab_type = "tanh";
    }
    else {
        activ_mode = rand() % 4 + 1;
        switch (activ_mode) {
            case 0: // tanh + sigmoid
            case 1: // cento
            default:
                tab_type = "tanh";
                break;
            case 2: // mirror
                tab_type = "sqr";
                break;
            case 3: // negative side
                tab_type = "nexp";
                break;
            case 4: // both sides
                tab_type = "double-side";
                break;
        }
    }

    uint32_t tab_off = rand() % (kL1EBankSize / sizeof(float)); // 1:findmax 0:bypass
    tab_off = std::min((size_t)tab_off, (kL1EBankSize / sizeof(float) - kEWTanhTableLengh * 2));
    gen_l1e(tab_off, tab_type, precision);

    uint32_t k_addr = unified_hw_addr(0, tab_off * 4, 4, 2);
    uint32_t b_addr = unified_hw_addr(0, (tab_off + kEWTanhTableLengh) * 4, 4, 2);
    uint32_t table_id = rand() % 5000 + 1;

    out << "ew_cfg " << "rs=" << precision << " rt=" << table_id
        << " rd=" << activ_mode << " shamt=" << 9 << std::endl;
    out << "ewtable_cfg " << "rs=" << k_addr << " rt=0"
        << " rd=0" << " shamt=" << 0 << std::endl;
    out << "ewtable_cfg " << "rs=" << b_addr << " rt=0"
        << " rd=0" << " shamt=" << 1 << std::endl;
}

void gen_ew_dsmadd_rand(std::ostream& out) {

    float coeff0 = rand() / (double)(RAND_MAX) * MAX_RANGE - (MAX_RANGE / 2);
    float coeff1 = rand() / (double)(RAND_MAX) * MAX_RANGE - (MAX_RANGE / 2);
    float coeff2 = rand() / (double)(RAND_MAX) * MAX_RANGE - (MAX_RANGE / 2);
    uint32_t* coeff0_ptr = reinterpret_cast<uint32_t*>(&coeff0);
    uint32_t* coeff1_ptr = reinterpret_cast<uint32_t*>(&coeff1);
    uint32_t* coeff2_ptr = reinterpret_cast<uint32_t*>(&coeff2);
    uint32_t coeff0_i32 = *coeff0_ptr;
    uint32_t coeff1_i32 = *coeff1_ptr;
    uint32_t coeff2_i32 = *coeff2_ptr;

    uint32_t find_max = rand() % 2; // 1:findmax 0:bypass
    float old_max = rand() / (double)(RAND_MAX) * MAX_RANGE;
    uint32_t* old_max_ptr = reinterpret_cast<uint32_t*>(&old_max);
    uint32_t old_max_u32 = *old_max_ptr;

    uint32_t vld_core_num = rand() % 16 + 1;

    uint32_t active_type = rand() % 9; // 0:null 1:relu 2:sigmoid 3:tanh
    uint32_t dst_sram_id = rand() % 2; // 0:l2-e-sram 1:l1-e-sram
    uint32_t dsmadd_shamt = dst_sram_id;

    uint32_t src_bank_l1e_sram = 0;
    uint32_t dst_bank_sram = 0;

    uint32_t coeff0_type = rand() % 2; //0:staitc scalar coeff   1:static scalar-vector coeff
    uint32_t coeff1_type = rand() % 2; //0:staitc scalar coeff   1:static scalar-vector coeff
    uint32_t coeff2_type = rand() % 2; //0:staitc scalar coeff   1:static scalar-vector coeff
    uint32_t ewcoeff_cfg_type = coeff0_type | (coeff1_type << 1) | (coeff2_type << 2); //coeff0 coeff1 coeff2

    uint32_t coeff0_vector_addr_sram = 0;
    uint32_t coeff1_vector_addr_sram = 0;
    uint32_t coeff2_vector_addr_sram = 0;
    uint32_t coeff0_vector_offset_l1e_sram = 0;
    uint32_t coeff1_vector_offset_l1e_sram = 0;
    uint32_t coeff2_vector_offset_l1e_sram = 0;

    if (coeff0_type == 0) {
        coeff0_vector_addr_sram = coeff0_i32;
    } else if (coeff0_type == 1) {
        coeff0_vector_offset_l1e_sram = (rand() % (kL1EBankValidSize / 4)) * 4;
        coeff0_vector_offset_l1e_sram = coeff0_vector_offset_l1e_sram > kL1EBankValidSize ? kL1EBankValidSize : coeff0_vector_offset_l1e_sram;
        coeff0_vector_addr_sram = (coeff0_vector_offset_l1e_sram >> 2) << (kL1EBankBits + 2) | (src_bank_l1e_sram << 2);
        assert(coeff0_vector_offset_l1e_sram <= kL1EBankValidSize);
    }
    if (coeff1_type == 0) {
        coeff1_vector_addr_sram = coeff1_i32;
    } else if (coeff1_type == 1) {
        coeff1_vector_offset_l1e_sram = (rand() % (kL1EBankValidSize / 4)) * 4;
        coeff1_vector_offset_l1e_sram = coeff1_vector_offset_l1e_sram > kL1EBankValidSize ? kL1EBankValidSize : coeff1_vector_offset_l1e_sram;
        coeff1_vector_addr_sram = (coeff1_vector_offset_l1e_sram >> 2) << (kL1EBankBits + 2) | (src_bank_l1e_sram << 2);
        assert(coeff1_vector_offset_l1e_sram <= kL1EBankValidSize);
    }
    if (coeff2_type == 0) {
        coeff2_vector_addr_sram = coeff2_i32;
    } else if (coeff2_type == 1) {
        coeff2_vector_offset_l1e_sram = (rand() % (kL1EBankValidSize / 4)) * 4;
        coeff2_vector_offset_l1e_sram = coeff2_vector_offset_l1e_sram > kL1EBankValidSize ? kL1EBankValidSize : coeff2_vector_offset_l1e_sram;
        coeff2_vector_addr_sram = (coeff2_vector_offset_l1e_sram >> 2) << (kL1EBankBits + 2) | (src_bank_l1e_sram << 2);
        assert(coeff2_vector_offset_l1e_sram <= kL1EBankValidSize);
    }
    out << "ewcoeff_cfg " << "rs=" << coeff1_vector_addr_sram << " rt=" << coeff2_vector_addr_sram
        << " rd=" << coeff0_vector_addr_sram << " shamt=" << ewcoeff_cfg_type << std::endl;

#if DEBUG
    std::cout << "ewcoeff_cfg_type is " << ewcoeff_cfg_type << std::endl;
    std::cout << "coeff0 is " << coeff0 << std::endl;
    std::cout << "coeff1 is " << coeff1 << std::endl;
    std::cout << "coeff2 is " << coeff2 << std::endl;
    std::cout << "active_type is " << active_type << std::endl;
    std::cout << "dst_sram_id is " << dst_sram_id << std::endl;
    std::cout << "dst_bank_sram is " << dst_bank_sram << std::endl;
    std::cout << "coeff2_vector_offset_l1e_sram is " << coeff2_vector_offset_l1e_sram << std::endl;
    std::cout << "coeff1_vector_offset_l1e_sram is " << coeff1_vector_offset_l1e_sram << std::endl;
    std::cout << "coeff0_vector_offset_l1e_sram is " << coeff0_vector_offset_l1e_sram << std::endl;
#endif

    uint32_t ew_cfg_type = 0;

    ew_cfg_type = 10;
    out << "ew_cfg " << "rs=0" << " rt=0"
        << " rd=" << active_type << " shamt=" << ew_cfg_type << std::endl;
    gen_ew_tab_cmd(out, active_type);

    int cmd_times= rand() % CMD_TIMES + 1;
    for (int i = 0; i < cmd_times; i++) {
        ew_cfg_type = 0;
        find_max = rand() % 2; // 1:findmax 0:bypass
        old_max = rand() / (double)(RAND_MAX) * MAX_RANGE;
        uint32_t* old_max_ptr = reinterpret_cast<uint32_t*>(&old_max);
        old_max_u32 = *old_max_ptr;
        out << "ew_cfg " << "rs=" << old_max_u32 << " rt=0"
            << " rd=" << find_max << " shamt=" << ew_cfg_type << std::endl;
        vld_core_num = rand() % 16 + 1;
        ew_cfg_type = 1;
        out << "ew_cfg " << "rs=0" << " rt=0"
            << " rd=" << vld_core_num << " shamt=" << ew_cfg_type << std::endl;

        uint32_t calcu_stream_size = 0;
        if (dst_sram_id == 0) {
            // l2esram
            calcu_stream_size = rand() % (kL2EBankSize / 4);
        } else if (dst_sram_id == 1) {
            // l1esram
            calcu_stream_size = rand() % (kL1EBankValidSize / 4);
        }

        uint32_t src0_offset_l1e_sram = (rand() % (kL1EBankValidSize / 4)) * 4;
        src0_offset_l1e_sram = (src0_offset_l1e_sram + calcu_stream_size * 4) > kL1EBankValidSize ? 0 : src0_offset_l1e_sram;
        uint32_t src0_offset_l1e_sram_end = src0_offset_l1e_sram  + calcu_stream_size * 4;
        uint32_t src0_addr_l1e_sram = (src0_offset_l1e_sram >> 2) << (kL1EBankBits + 2) | (src_bank_l1e_sram << 2);

        uint32_t src1_offset_l1e_sram = (rand() % (kL1EBankValidSize / 4)) * 4;
        src1_offset_l1e_sram = (src1_offset_l1e_sram + calcu_stream_size * 4) > kL1EBankValidSize ? 0 : src1_offset_l1e_sram;
        uint32_t src1_offset_l1e_sram_end = src1_offset_l1e_sram + calcu_stream_size * 4;
        uint32_t src1_addr_l1e_sram = (src1_offset_l1e_sram >> 2) << (kL1EBankBits + 2) | (src_bank_l1e_sram << 2);

        uint32_t dst_offset_sram = 0;
        uint32_t dst_addr_sram = 0;
        if (dst_sram_id == 0) {
            // l2esram
            dst_offset_sram = (rand() % (kL2EBankSize / 4)) * 4;
            dst_offset_sram = (dst_offset_sram + calcu_stream_size * 4) > kL2EBankSize ? 0 : dst_offset_sram;
        } else if (dst_sram_id == 1) {
            // l1esram
            uint32_t max_src_offset_l1e_sram_end = std::max(src0_offset_l1e_sram_end, src1_offset_l1e_sram_end);
            uint32_t min_src_offset_l1e_sram_start = std::min(src0_offset_l1e_sram, src1_offset_l1e_sram);
            if (max_src_offset_l1e_sram_end < kL1EBankValidSize) {
                dst_offset_sram = max_src_offset_l1e_sram_end + (rand() % (kL1EBankValidSize - max_src_offset_l1e_sram_end)) / 4 * 4;
            }
            if ((dst_offset_sram + calcu_stream_size * 4) >= kL1EBankValidSize) {
                dst_offset_sram = min_src_offset_l1e_sram_start;
            }
        }
        dst_addr_sram = ((dst_offset_sram >> 2) << (kL2EBankBits + 2)) | (dst_bank_sram << 2);

        assert(vld_core_num <= 16);
        assert(src0_offset_l1e_sram + calcu_stream_size * 4 <= kL1EBankValidSize);
        assert(src1_offset_l1e_sram + calcu_stream_size * 4 <= kL1EBankValidSize);
#if DEBUG
        std::cout << std::endl;
        std::cout << "calcu_stream_size is " << calcu_stream_size << std::endl;
        std::cout << "src0_offset_l1e_sram is " << src0_offset_l1e_sram << std::endl;
        std::cout << "src1_offset_l1e_sram is " << src1_offset_l1e_sram << std::endl;
        std::cout << "dst_offset_sram is " << dst_offset_sram << std::endl;
        std::cout << "calcu_stream_size * 4 is " << (calcu_stream_size * 4) << std::endl;
        std::cout << "vld_core_num is " << vld_core_num << std::endl;
        std::cout << "find_max is " << find_max << std::endl;
        std::cout << "old_max is " << old_max << std::endl;
#endif

        ew_cfg_type = 2;
        out << "ew_cfg " << "rs=0" << " rt=0"
            << " rd=" << calcu_stream_size << " shamt=" << ew_cfg_type << std::endl;
        out << "dsmadd " << "rs=" << src0_addr_l1e_sram << " rt=" << src1_addr_l1e_sram
            << " rd=" << dst_addr_sram << " shamt=" << dsmadd_shamt << std::endl;
        if (i % 3 == 0) {
            uint32_t coprocessor_type = 3;
            out << "ld_sd " << "rs=0" << " rt=0" << " rd=0" << " shamt=" << coprocessor_type << std::endl;
            out << "ld_sd " << "rs=1" << " rt=0" << " rd=0" << " shamt=" << coprocessor_type << std::endl;
        }
    }

}
void gen_ew_dsmul_rand(std::ostream& out) {
    int cmd_times= rand() % CMD_TIMES + 1;

    float coeff0 = rand() / (double)(RAND_MAX) * MAX_RANGE - (MAX_RANGE / 2);
    float coeff1 = rand() / (double)(RAND_MAX) * MAX_RANGE - (MAX_RANGE / 2);
    float coeff2 = rand() / (double)(RAND_MAX) * MAX_RANGE - (MAX_RANGE / 2);
    uint32_t* coeff0_ptr = reinterpret_cast<uint32_t*>(&coeff0);
    uint32_t* coeff1_ptr = reinterpret_cast<uint32_t*>(&coeff1);
    uint32_t* coeff2_ptr = reinterpret_cast<uint32_t*>(&coeff2);
    uint32_t coeff0_i32 = *coeff0_ptr;
    uint32_t coeff1_i32 = *coeff1_ptr;
    uint32_t coeff2_i32 = *coeff2_ptr;

    uint32_t find_max = rand() % 2; // 1:findmax 0:bypass
    float old_max = rand() / (double)(RAND_MAX / 100);
    uint32_t* old_max_ptr = reinterpret_cast<uint32_t*>(&old_max);
    uint32_t old_max_u32 = *old_max_ptr;

    uint32_t vld_core_num = rand() % 16 + 1;

    uint32_t active_type = rand() % 9; // 0:null 1:relu 2:sigmoid 3:tanh
    uint32_t dst_sram_id = rand() % 2; // 0:l2-e-sram 1:l1-e-sram
    uint32_t dsmul_shamt = dst_sram_id;

    uint32_t src_bank_l1e_sram = 0;
    uint32_t dst_bank_sram = 0;

    uint32_t coeff0_type = rand() % 2; //0:staitc scalar coeff   1:static scalar-vector coeff
    uint32_t coeff1_type = rand() % 2; //0:staitc scalar coeff   1:static scalar-vector coeff
    uint32_t coeff2_type = rand() % 2; //0:staitc scalar coeff   1:static scalar-vector coeff
    uint32_t ewcoeff_cfg_type = coeff0_type | (coeff1_type << 1) | (coeff2_type << 2); //coeff0 coeff1 coeff2

    uint32_t coeff0_vector_addr_sram = 0;
    uint32_t coeff1_vector_addr_sram = 0;
    uint32_t coeff2_vector_addr_sram = 0;
    uint32_t coeff0_vector_offset_l1e_sram = 0;
    uint32_t coeff1_vector_offset_l1e_sram = 0;
    uint32_t coeff2_vector_offset_l1e_sram = 0;

    if (coeff0_type == 0) {
        coeff0_vector_addr_sram = coeff0_i32;
    } else if (coeff0_type == 1) {
        coeff0_vector_offset_l1e_sram = (rand() % (kL1EBankValidSize / 4)) * 4;
        coeff0_vector_offset_l1e_sram = coeff0_vector_offset_l1e_sram > kL1EBankValidSize ? kL1EBankValidSize : coeff0_vector_offset_l1e_sram;
        coeff0_vector_addr_sram = (coeff0_vector_offset_l1e_sram >> 2) << (kL1EBankBits + 2) | (src_bank_l1e_sram << 2);
        assert(coeff0_vector_offset_l1e_sram <= kL1EBankValidSize);
    }
    if (coeff1_type == 0) {
        coeff1_vector_addr_sram = coeff1_i32;
    } else if (coeff1_type == 1) {
        coeff1_vector_offset_l1e_sram = (rand() % (kL1EBankValidSize / 4)) * 4;
        coeff1_vector_offset_l1e_sram = coeff1_vector_offset_l1e_sram > kL1EBankValidSize ? kL1EBankValidSize : coeff1_vector_offset_l1e_sram;
        coeff1_vector_addr_sram = (coeff1_vector_offset_l1e_sram >> 2) << (kL1EBankBits + 2) | (src_bank_l1e_sram << 2);
        assert(coeff1_vector_offset_l1e_sram <= kL1EBankValidSize);
    }
    if (coeff2_type == 0) {
        coeff2_vector_addr_sram = coeff2_i32;
    } else if (coeff2_type == 1) {
        coeff2_vector_offset_l1e_sram = (rand() % (kL1EBankValidSize / 4)) * 4;
        coeff2_vector_offset_l1e_sram = coeff2_vector_offset_l1e_sram > kL1EBankValidSize ? kL1EBankValidSize : coeff2_vector_offset_l1e_sram;
        coeff2_vector_addr_sram = (coeff2_vector_offset_l1e_sram >> 2) << (kL1EBankBits + 2) | (src_bank_l1e_sram << 2);
        assert(coeff2_vector_offset_l1e_sram <= kL1EBankValidSize);
    }
    out << "ewcoeff_cfg " << "rs=" << coeff1_vector_addr_sram << " rt=" << coeff2_vector_addr_sram
        << " rd=" << coeff0_vector_addr_sram << " shamt=" << ewcoeff_cfg_type << std::endl;

    uint32_t ew_cfg_type = 0;

#if DEBUG
    std::cout << "ewcoeff_cfg_type is " << ewcoeff_cfg_type << std::endl;
    std::cout << "coeff0 is " << coeff0 << std::endl;
    std::cout << "coeff1 is " << coeff1 << std::endl;
    std::cout << "coeff2 is " << coeff2 << std::endl;
    std::cout << "active_type is " << active_type << std::endl;
    std::cout << "dst_sram_id is " << dst_sram_id << std::endl;
    std::cout << "dst_bank_sram is " << dst_bank_sram << std::endl;
    std::cout << "coeff2_vector_offset_l1e_sram is " << coeff2_vector_offset_l1e_sram << std::endl;
    std::cout << "coeff1_vector_offset_l1e_sram is " << coeff1_vector_offset_l1e_sram << std::endl;
    std::cout << "coeff0_vector_offset_l1e_sram is " << coeff0_vector_offset_l1e_sram << std::endl;
#endif
    ew_cfg_type = 10;
    out << "ew_cfg " << "rs=0" << " rt=0"
        << " rd=" << active_type << " shamt=" << ew_cfg_type << std::endl;
    gen_ew_tab_cmd(out, active_type);

    for (int i = 0; i < cmd_times; i++) {
        ew_cfg_type = 0;
        find_max = rand() % 2; // 1:findmax 0:bypass
        old_max = rand() / (double)(RAND_MAX / 100);
        uint32_t* old_max_ptr = reinterpret_cast<uint32_t*>(&old_max);
        old_max_u32 = *old_max_ptr;
        out << "ew_cfg " << "rs=" << old_max_u32 << " rt=0"
            << " rd=" << find_max << " shamt=" << ew_cfg_type << std::endl;
        ew_cfg_type = 1;
        vld_core_num = rand() % 16 + 1;
        out << "ew_cfg " << "rs=0" << " rt=0"
            << " rd=" << vld_core_num << " shamt=" << ew_cfg_type << std::endl;

        uint32_t calcu_stream_size = 0;
        if (dst_sram_id == 0) {
            // l2esram
            calcu_stream_size = rand() % (kL2EBankSize / 4);
        } else if (dst_sram_id == 1) {
            // l1esram
            calcu_stream_size = rand() % (kL1EBankValidSize / 4);
        }

        uint32_t src0_offset_l1e_sram = (rand() % (kL1EBankValidSize / 4)) * 4;
        src0_offset_l1e_sram = (src0_offset_l1e_sram + calcu_stream_size * 4) > kL1EBankValidSize ? 0 : src0_offset_l1e_sram;
        uint32_t src0_offset_l1e_sram_end = src0_offset_l1e_sram  + calcu_stream_size * 4;
        uint32_t src0_addr_l1e_sram = (src0_offset_l1e_sram >> 2) << (kL1EBankBits + 2) | (src_bank_l1e_sram << 2);

        uint32_t src1_offset_l1e_sram = (rand() % (kL1EBankValidSize / 4)) * 4;
        src1_offset_l1e_sram = (src1_offset_l1e_sram + calcu_stream_size * 4) > kL1EBankValidSize ? 0 : src1_offset_l1e_sram;
        uint32_t src1_offset_l1e_sram_end = src1_offset_l1e_sram + calcu_stream_size * 4;
        uint32_t src1_addr_l1e_sram = (src1_offset_l1e_sram >> 2) << (kL1EBankBits + 2) | (src_bank_l1e_sram << 2);

        uint32_t dst_offset_sram = 0;
        uint32_t dst_addr_sram = 0;
        if (dst_sram_id == 0) {
            // l2esram
            dst_offset_sram = (rand() % (kL2EBankSize / 4)) * 4;
            dst_offset_sram = (dst_offset_sram + calcu_stream_size * 4) > kL2EBankSize ? 0 : dst_offset_sram;
        } else if (dst_sram_id == 1) {
            // l1esram
            uint32_t max_src_offset_l1e_sram_end = std::max(src0_offset_l1e_sram_end, src1_offset_l1e_sram_end);
            uint32_t min_src_offset_l1e_sram_start = std::min(src0_offset_l1e_sram, src1_offset_l1e_sram);
            if (max_src_offset_l1e_sram_end < kL1EBankValidSize) {
                dst_offset_sram = max_src_offset_l1e_sram_end + (rand() % (kL1EBankValidSize - max_src_offset_l1e_sram_end)) / 4 * 4;
            }
            if ((dst_offset_sram + calcu_stream_size * 4) >= kL1EBankValidSize) {
                dst_offset_sram = min_src_offset_l1e_sram_start;
            }
        }
        dst_addr_sram = ((dst_offset_sram >> 2) << (kL2EBankBits + 2)) | (dst_bank_sram << 2);

        assert(vld_core_num <= 16);
        assert(src0_offset_l1e_sram + calcu_stream_size * 4 <= kL1EBankValidSize);
        assert(src1_offset_l1e_sram + calcu_stream_size * 4 <= kL1EBankValidSize);

#if DEBUG
        std::cout << std::endl;
        std::cout << "calcu_stream_size is " << calcu_stream_size << std::endl;
        std::cout << "src0_offset_l1e_sram is " << src0_offset_l1e_sram << std::endl;
        std::cout << "src1_offset_l1e_sram is " << src1_offset_l1e_sram << std::endl;
        std::cout << "dst_offset_sram is " << dst_offset_sram << std::endl;
        std::cout << "calcu_stream_size * 4 is " << (calcu_stream_size * 4) << std::endl;
        std::cout << "vld_core_num is " << vld_core_num << std::endl;
        std::cout << "find_max is " << find_max << std::endl;
        std::cout << "old_max is " << old_max << std::endl;
#endif
        ew_cfg_type = 2;
        out << "ew_cfg " << "rs=0" << " rt=0"
            << " rd=" << calcu_stream_size << " shamt=" << ew_cfg_type << std::endl;
        out << "dsmul " << "rs=" << src0_addr_l1e_sram << " rt=" << src1_addr_l1e_sram
            << " rd=" << dst_addr_sram << " shamt=" << dsmul_shamt << std::endl;
        if (i % 3 == 0) {
            uint32_t coprocessor_type = 3;
            out << "ld_sd " << "rs=0" << " rt=0" << " rd=0" << " shamt=" << coprocessor_type << std::endl;
            out << "ld_sd " << "rs=1" << " rt=0" << " rd=0" << " shamt=" << coprocessor_type << std::endl;
        }
    }

}

void gen_ew_dscmpnsel_rand(std::ostream& out) {
    int cmd_times= rand() % CMD_TIMES + 1;

    uint32_t find_max = rand() % 2; // 1:findmax 0:bypass
    float old_max = rand() / (double)(RAND_MAX / 100);
    uint32_t* old_max_ptr = reinterpret_cast<uint32_t*>(&old_max);
    uint32_t old_max_u32 = *old_max_ptr;

    uint32_t vld_core_num = rand() % 16 + 1;

    uint32_t active_type = rand() % 9; // 0:null 1:relu 2:sigmoid 3:tanh
    uint32_t dst_sram_id = rand() % 2; // 0:l2-e-sram 1:l1-e-sram
    uint32_t op_type = rand() % 2; // 0:max 1:min

    uint32_t ew_cfg_type = 0;
    uint32_t dscmpnsel_shamt = (op_type << 2) | dst_sram_id;

    uint32_t src_bank_l1e_sram = 0;
    uint32_t dst_bank_sram = 0;
#if DEBUG
    std::cout << "find_max is " << find_max << std::endl;
    std::cout << "old_max is " << old_max << std::endl;
    std::cout << "vld_core_num is " << vld_core_num << std::endl;
    std::cout << "active_type is " << active_type << std::endl;
    std::cout << "dst_sram_id is " << dst_sram_id << std::endl;
    std::cout << "op_type is " << op_type << std::endl;
#endif

    ew_cfg_type = 10;
    out << "ew_cfg " << "rs=0" << " rt=0"
        << " rd=" << active_type << " shamt=" << ew_cfg_type << std::endl;
    gen_ew_tab_cmd(out, active_type);

    for (int i = 0; i < cmd_times; i++) {
        ew_cfg_type = 0;
        find_max = rand() % 2; // 1:findmax 0:bypass
        old_max = rand() / (double)(RAND_MAX / 100);
        uint32_t* old_max_ptr = reinterpret_cast<uint32_t*>(&old_max);
        old_max_u32 = *old_max_ptr;
        out << "ew_cfg " << "rs=" << old_max_u32 << " rt=0"
            << " rd=" << find_max << " shamt=" << ew_cfg_type << std::endl;
        ew_cfg_type = 1;
        out << "ew_cfg " << "rs=0" << " rt=0"
            << " rd=" << vld_core_num << " shamt=" << ew_cfg_type << std::endl;

        uint32_t calcu_stream_size = 0;
        if (dst_sram_id == 0) {
            // l2esram
            calcu_stream_size = rand() % (kL2EBankSize / 4);
        } else if (dst_sram_id == 1) {
            // l1esram
            calcu_stream_size = rand() % (kL1EBankValidSize / 4);
        }

        uint32_t src0_offset_l1e_sram = (rand() % (kL1EBankValidSize / 4)) * 4;
        src0_offset_l1e_sram = (src0_offset_l1e_sram + calcu_stream_size * 4) > kL1EBankValidSize ? 0 : src0_offset_l1e_sram;
        uint32_t src0_offset_l1e_sram_end = src0_offset_l1e_sram  + calcu_stream_size * 4;
        uint32_t src0_addr_l1e_sram = (src0_offset_l1e_sram >> 2) << (kL1EBankBits + 2) | (src_bank_l1e_sram << 2);

        uint32_t src1_offset_l1e_sram = (rand() % (kL1EBankValidSize / 4)) * 4;
        src1_offset_l1e_sram = (src1_offset_l1e_sram + calcu_stream_size * 4) > kL1EBankValidSize ? 0 : src1_offset_l1e_sram;
        uint32_t src1_offset_l1e_sram_end = src1_offset_l1e_sram + calcu_stream_size * 4;
        uint32_t src1_addr_l1e_sram = (src1_offset_l1e_sram >> 2) << (kL1EBankBits + 2) | (src_bank_l1e_sram << 2);

        uint32_t dst_offset_sram = 0;
        uint32_t dst_addr_sram = 0;
        if (dst_sram_id == 0) {
            // l2esram
            dst_offset_sram = (rand() % (kL2EBankSize / 4)) * 4;
            dst_offset_sram = (dst_offset_sram + calcu_stream_size * 4) > kL2EBankSize ? 0 : dst_offset_sram;
        } else if (dst_sram_id == 1) {
            // l1esram
            uint32_t max_src_offset_l1e_sram_end = std::max(src0_offset_l1e_sram_end, src1_offset_l1e_sram_end);
            uint32_t min_src_offset_l1e_sram_start = std::min(src0_offset_l1e_sram, src1_offset_l1e_sram);
            if (max_src_offset_l1e_sram_end < kL1EBankValidSize) {
                dst_offset_sram = max_src_offset_l1e_sram_end + (rand() % (kL1EBankValidSize - max_src_offset_l1e_sram_end)) / 4 * 4;
            }
            if ((dst_offset_sram + calcu_stream_size * 4) >= kL1EBankValidSize) {
                dst_offset_sram = min_src_offset_l1e_sram_start;
            }
        }
        dst_addr_sram = ((dst_offset_sram >> 2) << (kL2EBankBits + 2)) | (dst_bank_sram << 2);

        assert(vld_core_num <= 16);
        assert(src0_offset_l1e_sram + calcu_stream_size * 4 <= kL1EBankValidSize);
        assert(src1_offset_l1e_sram + calcu_stream_size * 4 <= kL1EBankValidSize);

#if DEBUG
        std::cout << std::endl;
        std::cout << "calcu_stream_size is " << calcu_stream_size << std::endl;
        std::cout << "src0_offset_l1e_sram is " << src0_offset_l1e_sram << std::endl;
        std::cout << "src1_offset_l1e_sram is " << src1_offset_l1e_sram << std::endl;
        std::cout << "dst_offset_sram is " << dst_offset_sram << std::endl;
        std::cout << "calcu_stream_size * 4 is " << (calcu_stream_size * 4) << std::endl;
#endif
        ew_cfg_type = 2;
        out << "ew_cfg " << "rs=0" << " rt=0"
            << " rd=" << calcu_stream_size << " shamt=" << ew_cfg_type << std::endl;
        out << "dscmpnsel " << "rs=" << src0_addr_l1e_sram << " rt=" << src1_addr_l1e_sram
            << " rd=" << dst_addr_sram << " shamt=" << dscmpnsel_shamt << std::endl;
        if (i % 3 == 0) {
            uint32_t coprocessor_type = 3;
            out << "ld_sd " << "rs=0" << " rt=0" << " rd=0" << " shamt=" << coprocessor_type << std::endl;
            out << "ld_sd " << "rs=1" << " rt=0" << " rd=0" << " shamt=" << coprocessor_type << std::endl;
        }
    }

}

void gen_ew_sspooling(std::ostream& out) {

    int cfg_shamt = 0;
    int find_max = rand() % 2; // 0-bypass 1-findmax
    float old_max = rand() / (double)(RAND_MAX) * MAX_RANGE;
    uint32_t* old_max_ptr = reinterpret_cast<uint32_t*>(&old_max);
    uint32_t old_max_u32 = *old_max_ptr;

    int ave_pad = rand() % 2; // 0:ave pooling with pad 1:ave pooling without pad
    int max_or_ave = rand() % 2; // 0:max pooling 1:ave pooling
    int max_index_enable = rand() % 2; // 0:no need to calcu max index 1:output max index
    int dst_sram = rand() % 2; // 0-l2e, 1-l1e
    int dst_sram_depth = kFpBankDepth;
    if (dst_sram == 1) {
        dst_sram_depth = kL1EBankValidDepth;
    }
    uint32_t pool_shamt = (ave_pad << 4) | (max_or_ave << 3) | (max_index_enable << 2) | dst_sram;

    uint32_t max_input_hw = 64;
    if (((max_or_ave == 0) && (max_index_enable == 1)) || (dst_sram == 1)) {
        max_input_hw = 32;
    }

    int input_w = rand() % max_input_hw + 1;
    int input_h = rand() % max_input_hw + 1;
    input_w = std::max(input_w, 1);
    input_h = std::max(input_h, 1);
    int input_size = input_w * input_h;

    int filter_w = rand() % input_w + 1;
    int filter_h = rand() % input_h + 1;

    int pad_left = rand() % input_w;
    int pad_right = rand() % input_w;
    if (max_or_ave == 1) {
        pad_left = rand() % filter_w;
        pad_right = rand() % filter_w;
    }

    int pad_up = rand() % input_h;
    int pad_down = rand() % input_h;
    if (max_or_ave == 1) {
        pad_up = rand() % filter_h;
        pad_down = rand() % filter_h;
    }

    int stride_w = rand() % input_w + 1;
    int stride_h = rand() % input_h + 1;

    assert (filter_w <= input_w + pad_left + pad_right);
    assert (filter_h <= input_h + pad_up + pad_down);

    int output_w = (input_w + pad_left + pad_right - filter_w) / stride_w + 1;
    int output_h = (input_h + pad_up + pad_down - filter_h) / stride_h + 1;
    int output_range_w = (input_w + pad_left + pad_right) / stride_w + 1;
    int output_range_h = (input_h + pad_up + pad_down) / stride_h + 1;
    if (output_range_w > output_w) {
        output_w = rand() % (output_range_w - output_w) + 1 + output_w;
    }
    if (output_range_h > output_h) {
        output_h = rand() % (output_range_h - output_h) + 1 + output_h;
    }
    int output_size = output_w * output_h;

    if (output_size >= dst_sram_depth) {
        pad_left = 0;
        pad_right = 0;
        pad_up = 0;
        pad_down = 0;
        stride_w = 2;
        stride_h = 2;
        output_w = (input_w + pad_left + pad_right - filter_w) / stride_w + 1;
        output_h = (input_h + pad_up + pad_down - filter_h) / stride_h + 1;
        output_size = output_w * output_h;
        assert(output_size < dst_sram_depth);
    }

    cfg_shamt = 4;
    out << "ew_cfg " << "rs=" << input_h << " rt=0"
        << " rd=" << input_w << " shamt=" << cfg_shamt << std::endl;

    cfg_shamt = 5;
    out << "ew_cfg " << "rs=" << filter_h << " rt=0"
        << " rd=" << filter_w << " shamt=" << cfg_shamt << std::endl;

    cfg_shamt = 7;
    out << "ew_cfg " << "rs=" << pad_right << " rt=0"
        << " rd=" << pad_left << " shamt=" << cfg_shamt << std::endl;

    cfg_shamt = 8;
    out << "ew_cfg " << "rs=" << pad_down << " rt=0"
        << " rd=" << pad_up << " shamt=" << cfg_shamt << std::endl;

    cfg_shamt = 6;
    out << "ew_cfg " << "rs=" << stride_h << " rt=0"
        << " rd=" << stride_w << " shamt=" << cfg_shamt << std::endl;

    cfg_shamt = 3;
    out << "ew_cfg " << "rs=" << output_h << " rt=0"
        << " rd=" << output_w << " shamt=" << cfg_shamt << std::endl;

    if ((max_or_ave == 0) && (max_index_enable == 1)) {
        assert((input_size + output_size) < kL1EBankValidDepth);
    }

#if DEBUG
    std::cout << "input_h = " << input_h << std::endl;
    std::cout << "input_w = " << input_w << std::endl;
    std::cout << "input_size = " << input_size << std::endl;
    std::cout << "pad_left = " << pad_left << std::endl;
    std::cout << "pad_right = " << pad_right << std::endl;
    std::cout << "pad_up = " << pad_up << std::endl;
    std::cout << "pad_down = " << pad_down << std::endl;
    std::cout << "stride_w = " << stride_w << std::endl;
    std::cout << "stride_h = " << stride_h << std::endl;
    std::cout << "filter_w = " << filter_w << std::endl;
    std::cout << "filter_h = " << filter_h << std::endl;
    std::cout << "output_w = " << output_w << std::endl;
    std::cout << "output_h = " << output_h << std::endl;
    std::cout << "output_range_w = " << output_range_w << std::endl;
    std::cout << "output_range_h = " << output_range_h << std::endl;
    std::cout << "output_size = " << output_size << std::endl;
#endif
    int cmd_times= rand() % CMD_TIMES + 1;

    int src_off = 0;
    uint32_t dst_off = 0;
    uint32_t dst_index_off = 0;
    uint32_t dst_index_addr = 0;

    for (int i = 0; i < cmd_times; i++) {
        cfg_shamt = 0;
        find_max = rand() % 2; // 0-bypass 1-findmax
        old_max = rand() / (double)(RAND_MAX) * MAX_RANGE;
        uint32_t* old_max_ptr = reinterpret_cast<uint32_t*>(&old_max);
        old_max_u32 = *old_max_ptr;
        out << "ew_cfg " << "rs=" << old_max_u32 << " rt=0"
            << " rd=" << find_max << " shamt=" << cfg_shamt << std::endl;

        cfg_shamt = 1;
        int vld_core_num = rand() % 16 + 1;
        out << "ew_cfg " << "rs=0" << " rt=0"
            << " rd=" << vld_core_num << " shamt=" << cfg_shamt << std::endl;

        if ((max_or_ave == 0) && (max_index_enable == 1)) {
            // max pooling
            assert((input_size + output_size) < kL1EBankValidDepth);
            src_off = (rand() % (kL1EBankValidDepth - input_size - output_size));
        } else {
            assert(input_size < kL1EBankValidDepth);
            src_off = (rand() % (kL1EBankValidDepth - input_size));
        }
        assert(src_off >= 0);
        uint32_t src_addr = unified_hw_addr(0, src_off * 4, 4, 2);

        if (dst_sram == 1) {
            // result to l1e
            assert((src_off + input_size) < kL1EBankValidDepth);
            dst_off = src_off + input_size + (rand() % (kL1EBankValidDepth - src_off - input_size));
            if ((dst_off + output_size) > kL1EBankValidDepth) {
                src_off = 0;
                src_addr = unified_hw_addr(0, src_off * 4, 4, 2);
                dst_off = src_off + input_size;
            }
            if ((max_or_ave == 0) && (max_index_enable == 1)) {
                // index result to l2e
                assert(output_size < kFpBankDepth);
                dst_index_off = (rand() % (kFpBankDepth - output_size));
                if ((dst_index_off + output_size) > kFpBankDepth) {
                    dst_index_off = 0;
                }
                dst_index_addr = unified_hw_addr(0, dst_index_off * 4, 4, 2);
            }
            assert((src_off + input_size) <= dst_off);
        } else {
            // result to l2e
            assert(output_size < kFpBankDepth);
            dst_off = (rand() % (kFpBankDepth - output_size));
            if ((max_or_ave == 0) && (max_index_enable == 1)) {
                // index result to l1e
                assert((src_off + input_size) != kL1EBankValidDepth);
                dst_index_off = src_off + input_size + (rand() % (kL1EBankValidDepth - src_off - input_size));
                if ((dst_index_off + output_w * output_h) > kL1EBankValidDepth) {
                    src_off = 0;
                    src_addr = unified_hw_addr(0, src_off * 4, 4, 2);
                    dst_index_off = src_off + input_size;
                }
                dst_index_addr = unified_hw_addr(0, dst_index_off * 4, 4, 2);
                assert((src_off + input_size) <= dst_index_off);
            }

        }
        uint32_t dst_addr = unified_hw_addr(0, dst_off * 4, 4, 2);

#if DEBUG
        std::cout << "src_off = " << src_off << std::endl;
        std::cout << "dst_off = " << dst_off << std::endl;
        std::cout << "dst_index_off = " << dst_index_off << std::endl;
        std::cout << "src_addr = " << src_addr << std::endl;
        std::cout << "dst_addr = " << dst_addr << std::endl;
#endif
        assert(src_addr < 2048 * 4 * 64);
        assert(dst_addr < 2048 * 4 * 64);

        out << "sspooling " << "rs=" << src_addr << " rt=" << dst_index_addr
            << " rd=" << dst_addr << " shamt=" << pool_shamt << std::endl;
        if (i % 3 == 0) {
            int coprocessor_type = 3;
            out << "ld_sd " << "rs=0" << " rt=0" << " rd=0" << " shamt=" << coprocessor_type << std::endl;
            out << "ld_sd " << "rs=1" << " rt=0" << " rd=0" << " shamt=" << coprocessor_type << std::endl;
        }
    }

}

void gen_ew_ssresize(std::ostream& out) {

    bool config_flag = false;

    int cfg_shamt = 0;
    int find_max = rand() % 2; // 0-bypass 1-findmax
    float old_max = rand() / (double)(RAND_MAX) * MAX_RANGE;
    uint32_t* old_max_ptr = reinterpret_cast<uint32_t*>(&old_max);
    uint32_t old_max_u32 = *old_max_ptr;

    int dst_sram = rand() % 2; // 0-l2e, 1-l1e
    uint32_t resize_shamt = dst_sram;

    uint32_t max_src_l1e_sram_offset = kL1EBankValidDepth; // 4096
    uint32_t max_input_hw = 64;

    if (dst_sram == 1) {
        max_src_l1e_sram_offset = kL1EBankValidDepth / 2;
        max_input_hw = 32;
    }

    uint32_t factor_w_u32 = 0;
    uint32_t factor_h_u32 = 0;
    int input_size = 0;
    int output_size = 0;
    float factor_w = 1;
    float factor_h = 1;
    int resize_output_w = 1;
    int resize_output_h = 1;
    int resize_input_w = 1;
    int resize_input_h = 1;
    int resize_output_offset_w = 1;
    int resize_output_offset_h = 1;
    int resize_input_offset_w = 1;
    int resize_input_offset_h = 1;

    while (config_flag == false) {
        uint32_t max_output_hw = 32;
        resize_output_w = rand() % max_output_hw + 1;
        resize_output_h = rand() % max_output_hw + 1;
        output_size = resize_output_w * resize_output_h;

        resize_input_w = rand() % max_input_hw + 1;
        resize_input_h = rand() % max_input_hw + 1;
        if (resize_input_w * resize_input_h >= max_src_l1e_sram_offset) {
            resize_input_w = resize_input_w / 2;
            resize_input_h = resize_input_h / 2;
        }
        input_size = resize_input_w * resize_input_h;

        resize_output_offset_w = rand() % 10;
        resize_output_offset_h = rand() % 10;

        resize_input_offset_w = rand() % 10;
        resize_input_offset_h = rand() % 10;

        if (resize_output_offset_h == 0) {
            resize_input_offset_h = 0;
        }
        if (resize_output_offset_w == 0) {
            resize_input_offset_w = 0;
        }

        float factor_w_max = 1;
        float factor_w_min = 1;
        float factor_h_max = 1;
        float factor_h_min = 1;

        factor_w = 1;
        if (resize_output_w > 1) {
            factor_w_max = (float)(resize_input_w - 1 + resize_output_offset_w) / (float)(resize_output_w - 1);
        } else {
            factor_w_max = (float)(resize_input_w - 1 + resize_output_offset_w) / (float)(resize_output_w);
        }
        if (resize_output_offset_w > 0) {
            factor_w_min = (float)resize_input_offset_w / (float)resize_output_offset_w;
        } else {
            factor_w_min = factor_w_max;
        }
        if (factor_w_min > factor_w_max) {
            continue;
        }
        factor_w = (factor_w_min + factor_w_max) / 2;
        uint32_t* factor_w_ptr = reinterpret_cast<uint32_t*>(&factor_w);
        factor_w_u32 = *factor_w_ptr;

        if (resize_output_h > 1) {
            factor_h_max = (float)(resize_input_h - 1 + resize_output_offset_h) / (float)(resize_output_h - 1);
        } else {
            factor_h_max = (float)(resize_input_h - 1 + resize_output_offset_h) / (float)(resize_output_h);
        }
        if (resize_output_offset_h > 0) {
            factor_h_min = (float)resize_input_offset_h / (float)resize_output_offset_h;
        } else {
            factor_h_min = factor_h_max;
        }
        if (factor_h_min > factor_h_max) {
            continue;
        }
        factor_h = (factor_h_min + factor_h_max) / 2;
        uint32_t* factor_h_ptr = reinterpret_cast<uint32_t*>(&factor_h);
        factor_h_u32 = *factor_h_ptr;

        config_flag = true;
    } // end of while

    cfg_shamt = 12;
    out << "ew_cfg " << "rs=" << resize_output_h << " rt=0"
        << " rd=" << resize_output_w << " shamt=" << cfg_shamt << std::endl;

    cfg_shamt = 13;
    out << "ew_cfg " << "rs=" << resize_input_h << " rt=0"
        << " rd=" << resize_input_w << " shamt=" << cfg_shamt << std::endl;

    cfg_shamt = 14;
    out << "ew_cfg " << "rs=" << resize_output_offset_h << " rt=0"
        << " rd=" << resize_output_offset_w << " shamt=" << cfg_shamt << std::endl;

    cfg_shamt = 15;
    out << "ew_cfg " << "rs=" << resize_input_offset_h << " rt=0"
        << " rd=" << resize_input_offset_w << " shamt=" << cfg_shamt << std::endl;

    cfg_shamt = 11;
    out << "ew_cfg " << "rs=" << factor_h_u32 << " rt=0"
        << " rd=" << factor_w_u32 << " shamt=" << cfg_shamt << std::endl;

    int cmd_times= rand() % CMD_TIMES + 1;

    uint32_t dst_off = 0;

#if DEBUG
    std::cout << "resize_output_w= " << resize_output_w << std::endl;
    std::cout << "resize_output_h= " << resize_output_h << std::endl;
    std::cout << "resize_input_w= " << resize_input_w << std::endl;
    std::cout << "resize_input_h= " << resize_input_h << std::endl;
    std::cout << "resize_output_offset_w= " << resize_output_offset_w << std::endl;
    std::cout << "resize_output_offset_h= " << resize_output_offset_h << std::endl;
    std::cout << "resize_input_offset_w= " << resize_input_offset_w << std::endl;
    std::cout << "resize_input_offset_h= " << resize_input_offset_h << std::endl;
    std::cout << "factor_w= " << factor_w << std::endl;
    std::cout << "factor_w_min= " << factor_w_min << std::endl;
    std::cout << "factor_w_max= " << factor_w_max << std::endl;
    std::cout << "factor_h= " << factor_h << std::endl;
    std::cout << "factor_h_min= " << factor_h_min << std::endl;
    std::cout << "factor_h_max= " << factor_h_max << std::endl;
#endif
    for (int i = 0; i < cmd_times; i++) {
        cfg_shamt = 0;
        find_max = rand() % 2; // 0-bypass 1-findmax
        old_max = rand() / (double)(RAND_MAX) * MAX_RANGE;
        uint32_t* old_max_ptr = reinterpret_cast<uint32_t*>(&old_max);
        old_max_u32 = *old_max_ptr;
        out << "ew_cfg " << "rs=" << old_max_u32 << " rt=0"
            << " rd=" << find_max << " shamt=" << cfg_shamt << std::endl;

        cfg_shamt = 1;
        int vld_core_num = rand() % 16 + 1;
        out << "ew_cfg " << "rs=0" << " rt=0"
            << " rd=" << vld_core_num << " shamt=" << cfg_shamt << std::endl;

        assert(input_size <= max_src_l1e_sram_offset);
        int src_off = (rand() % (max_src_l1e_sram_offset - input_size));
        assert(src_off >= 0);
        uint32_t src_addr = unified_hw_addr(0, src_off * 4, 4, 2);

        if (dst_sram == 1) {
            // result to l1e
            assert((src_off + input_size) < max_src_l1e_sram_offset);
            dst_off = src_off + input_size + (rand() % (kL1EBankValidDepth - src_off - input_size));
            if ((dst_off + output_size) > kL1EBankValidDepth) {
                src_off = 0;
                src_addr = unified_hw_addr(0, src_off * 4, 4, 2);
                dst_off = src_off + input_size;
            }
            assert((src_off + input_size) <= dst_off);
            assert((dst_off + output_size) <= kL1EBankValidDepth);
        } else {
            // result to l2e
            assert(output_size <= kFpBankDepth);
            dst_off = (rand() % (kFpBankDepth - output_size));
        }
        uint32_t dst_addr = unified_hw_addr(0, dst_off * 4, 4, 2);

#if DEBUG
        std::cout << "src_off = " << src_off << std::endl;
        std::cout << "dst_off = " << dst_off << std::endl;
        std::cout << "src_addr = " << src_addr << std::endl;
        std::cout << "dst_addr = " << dst_addr << std::endl;
#endif

        out << "ssresize " << "rs=" << src_addr << " rt=0"
            << " rd=" << dst_addr << " shamt=" << resize_shamt << std::endl;
        if (i % 3 == 0) {
            int coprocessor_type = 3;
            out << "ld_sd " << "rs=0" << " rt=0" << " rd=0" << " shamt=" << coprocessor_type << std::endl;
            out << "ld_sd " << "rs=1" << " rt=0" << " rd=0" << " shamt=" << coprocessor_type << std::endl;
        }
    }

}

void gen_ew_ssvsum_rand(std::ostream& out) {
    int cmd_times= rand() % CMD_TIMES + 1;

    uint32_t find_max = rand() % 2; // 1:findmax 0:bypass
    float old_max = rand() / (double)(RAND_MAX / 100);
    uint32_t* old_max_ptr = reinterpret_cast<uint32_t*>(&old_max);
    uint32_t old_max_u32 = *old_max_ptr;

    uint32_t vld_core_num = rand() % 16 + 1;

    uint32_t active_type = rand() % 9; // 0:null 1:relu 2:sigmoid 3:tanh
    uint32_t dst_sram_id = rand() % 2; // 0:l2-e-sram 1:l1-e-sram

    uint32_t ew_cfg_type = 0;
    uint32_t ssvsum_shamt = dst_sram_id;

    uint32_t src_bank_l1e_sram = 0;
    uint32_t dst_bank_sram = 0;
#if DEBUG
    std::cout << "find_max is " << find_max << std::endl;
    std::cout << "old_max is " << old_max << std::endl;
    std::cout << "vld_core_num is " << vld_core_num << std::endl;
    std::cout << "active_type is " << active_type << std::endl;
    std::cout << "dst_sram_id is " << dst_sram_id << std::endl;
#endif

    ew_cfg_type = 10;
    out << "ew_cfg " << "rs=0" << " rt=0"
        << " rd=" << active_type << " shamt=" << ew_cfg_type << std::endl;
    gen_ew_tab_cmd(out, active_type);

    for (int i = 0; i < cmd_times; i++) {
        ew_cfg_type = 0;
        find_max = rand() % 2; // 1:findmax 0:bypass
        old_max = rand() / (double)(RAND_MAX / 100);
        uint32_t* old_max_ptr = reinterpret_cast<uint32_t*>(&old_max);
        old_max_u32 = *old_max_ptr;
        out << "ew_cfg " << "rs=" << old_max_u32 << " rt=0"
            << " rd=" << find_max << " shamt=" << ew_cfg_type << std::endl;

        ew_cfg_type = 1;
        out << "ew_cfg " << "rs=0" << " rt=0"
            << " rd=" << vld_core_num << " shamt=" << ew_cfg_type << std::endl;

        uint32_t calcu_stream_size = 0;
        if (dst_sram_id == 0) {
            // l2esram
            calcu_stream_size = rand() % (kL2EBankSize / 4);
        } else if (dst_sram_id == 1) {
            // l1esram
            calcu_stream_size = rand() % (kL1EBankValidSize / 4);
        }

        uint32_t src0_offset_l1e_sram = (rand() % (kL1EBankValidSize / 4)) * 4;
        src0_offset_l1e_sram = (src0_offset_l1e_sram + calcu_stream_size * 4) > kL1EBankValidSize ? 0 : src0_offset_l1e_sram;
        uint32_t src0_offset_l1e_sram_end = src0_offset_l1e_sram  + calcu_stream_size * 4;
        uint32_t src0_addr_l1e_sram = (src0_offset_l1e_sram >> 2) << (kL1EBankBits + 2) | (src_bank_l1e_sram << 2);

        uint32_t dst_offset_sram = 0;
        uint32_t dst_addr_sram = 0;
        if (dst_sram_id == 0) {
            // l2esram
            dst_offset_sram = (rand() % (kL2EBankSize / 4)) * 4;
            dst_offset_sram = dst_offset_sram >= kL2EBankSize ? 0 : dst_offset_sram;
        } else if (dst_sram_id == 1) {
            // l1esram
            if (kL1EBankValidSize > src0_offset_l1e_sram_end) {
                dst_offset_sram = src0_offset_l1e_sram_end + (rand() % (kL1EBankValidSize - src0_offset_l1e_sram_end)) / 4 * 4;
            } else {
                dst_offset_sram = src0_offset_l1e_sram;
            }
            if (dst_offset_sram >= kL1EBankValidSize) {
                dst_offset_sram = src0_offset_l1e_sram;
            }
        }
        dst_addr_sram = ((dst_offset_sram >> 2) << (kL2EBankBits + 2)) | (dst_bank_sram << 2);

        assert(vld_core_num <= 16);
        assert(src0_offset_l1e_sram + calcu_stream_size * 4 <= kL1EBankValidSize);

#if DEBUG
        std::cout << std::endl;
        std::cout << "calcu_stream_size is " << calcu_stream_size << std::endl;
        std::cout << "src0_offset_l1e_sram is " << src0_offset_l1e_sram << std::endl;
        std::cout << "dst_offset_sram is " << dst_offset_sram << std::endl;
#endif
        ew_cfg_type = 2;
        out << "ew_cfg " << "rs=0" << " rt=0"
            << " rd=" << calcu_stream_size << " shamt=" << ew_cfg_type << std::endl;
        out << "ssvsum " << "rs=" << src0_addr_l1e_sram << " rt=" << 0
            << " rd=" << dst_addr_sram << " shamt=" << ssvsum_shamt << std::endl;
        if (i % 3 == 0) {
            uint32_t coprocessor_type = 3;
            out << "ld_sd " << "rs=0" << " rt=0" << " rd=0" << " shamt=" << coprocessor_type << std::endl;
            out << "ld_sd " << "rs=1" << " rt=0" << " rd=0" << " shamt=" << coprocessor_type << std::endl;
        }
    }

}

void gen_ew_dsdiv_rand(std::ostream& out) {
    int cmd_times= rand() % CMD_TIMES + 1;

    float coeff0 = rand() / (double)(RAND_MAX) * MAX_RANGE - (MAX_RANGE / 2);
    uint32_t* coeff0_ptr = reinterpret_cast<uint32_t*>(&coeff0);
    uint32_t coeff0_i32 = *coeff0_ptr;
    uint32_t ewcoeff_cfg_type = 0;

    uint32_t find_max = rand() % 2; // 1:findmax 0:bypass
    float old_max = rand() / (double)(RAND_MAX / 100);
    uint32_t* old_max_ptr = reinterpret_cast<uint32_t*>(&old_max);
    uint32_t old_max_u32 = *old_max_ptr;

    uint32_t vld_core_num = rand() % 16 + 1;

    uint32_t active_type = rand() % 9; // 0:null 1:relu 2:sigmoid 3:tanh
    uint32_t dst_sram_id = rand() % 2; // 0:l2-e-sram 1:l1-e-sram
    uint32_t dsdiv_mode = rand() % 3; // 0: scalar/vector 1: vector/scalar 2: vector/vector
    uint32_t dsdiv_shamt = (dsdiv_mode << 1) | dst_sram_id;

    uint32_t src_bank_l1e_sram = 0;
    uint32_t dst_bank_sram = 0;

    out << "ewcoeff_cfg " << "rs=0" << " rt=0"
        << " rd=" << coeff0_i32 << " shamt=" << ewcoeff_cfg_type << std::endl;
    uint32_t ew_cfg_type = 0;

    ew_cfg_type = 10;
    out << "ew_cfg " << "rs=0" << " rt=0"
        << " rd=" << active_type << " shamt=" << ew_cfg_type << std::endl;
    gen_ew_tab_cmd(out, active_type);

#if DEBUG
    std::cout << "ewcoeff_cfg_type is " << ewcoeff_cfg_type << std::endl;
    std::cout << "coeff0 is " << coeff0 << std::endl;
    std::cout << "active_type is " << active_type << std::endl;
    std::cout << "dst_sram_id is " << dst_sram_id << std::endl;
    std::cout << "dst_bank_sram is " << dst_bank_sram << std::endl;
#endif

    for (int i = 0; i < cmd_times; i++) {
        ew_cfg_type = 0;
        find_max = rand() % 2; // 1:findmax 0:bypass
        old_max = rand() / (double)(RAND_MAX / 100);
        uint32_t* old_max_ptr = reinterpret_cast<uint32_t*>(&old_max);
        old_max_u32 = *old_max_ptr;
        out << "ew_cfg " << "rs=" << old_max_u32 << " rt=0"
            << " rd=" << find_max << " shamt=" << ew_cfg_type << std::endl;
        ew_cfg_type = 1;
        vld_core_num = rand() % 16 + 1;
        out << "ew_cfg " << "rs=0" << " rt=0"
            << " rd=" << vld_core_num << " shamt=" << ew_cfg_type << std::endl;

        uint32_t calcu_stream_size = 0;
        if (dst_sram_id == 0) {
            // l2esram
            calcu_stream_size = rand() % (kL2EBankSize / 4);
        } else if (dst_sram_id == 1) {
            // l1esram
            calcu_stream_size = rand() % (kL1EBankValidSize / 4);
        }

        uint32_t src0_offset_l1e_sram = (rand() % (kL1EBankValidSize / 4)) * 4;
        src0_offset_l1e_sram = (src0_offset_l1e_sram + calcu_stream_size * 4) > kL1EBankValidSize ? 0 : src0_offset_l1e_sram;
        uint32_t src0_offset_l1e_sram_end = src0_offset_l1e_sram  + calcu_stream_size * 4;
        uint32_t src0_addr_l1e_sram = (src0_offset_l1e_sram >> 2) << (kL1EBankBits + 2) | (src_bank_l1e_sram << 2);

        uint32_t src1_offset_l1e_sram = (rand() % (kL1EBankValidSize / 4)) * 4;
        src1_offset_l1e_sram = (src1_offset_l1e_sram + calcu_stream_size * 4) > kL1EBankValidSize ? 0 : src1_offset_l1e_sram;
        uint32_t src1_offset_l1e_sram_end = src1_offset_l1e_sram + calcu_stream_size * 4;
        uint32_t src1_addr_l1e_sram = (src1_offset_l1e_sram >> 2) << (kL1EBankBits + 2) | (src_bank_l1e_sram << 2);

        uint32_t dst_offset_sram = 0;
        uint32_t dst_addr_sram = 0;
        if (dst_sram_id == 0) {
            // l2esram
            dst_offset_sram = (rand() % (kL2EBankSize / 4)) * 4;
            dst_offset_sram = (dst_offset_sram + calcu_stream_size * 4) > kL2EBankSize ? 0 : dst_offset_sram;
        } else if (dst_sram_id == 1) {
            // l1esram
            uint32_t max_src_offset_l1e_sram_end = std::max(src0_offset_l1e_sram_end, src1_offset_l1e_sram_end);
            uint32_t min_src_offset_l1e_sram_start = std::min(src0_offset_l1e_sram, src1_offset_l1e_sram);
            if (max_src_offset_l1e_sram_end < kL1EBankValidSize) {
                dst_offset_sram = max_src_offset_l1e_sram_end + (rand() % (kL1EBankValidSize - max_src_offset_l1e_sram_end)) / 4 * 4;
            }
            if ((dst_offset_sram + calcu_stream_size * 4) >= kL1EBankValidSize) {
                dst_offset_sram = min_src_offset_l1e_sram_start;
            }
        }
        dst_addr_sram = ((dst_offset_sram >> 2) << (kL2EBankBits + 2)) | (dst_bank_sram << 2);

        assert(vld_core_num <= 16);
        assert(src0_offset_l1e_sram + calcu_stream_size * 4 <= kL1EBankValidSize);
        assert(src1_offset_l1e_sram + calcu_stream_size * 4 <= kL1EBankValidSize);

#if DEBUG
        std::cout << std::endl;
        std::cout << "calcu_stream_size is " << calcu_stream_size << std::endl;
        std::cout << "src0_offset_l1e_sram is " << src0_offset_l1e_sram << std::endl;
        std::cout << "src1_offset_l1e_sram is " << src1_offset_l1e_sram << std::endl;
        std::cout << "dst_offset_sram is " << dst_offset_sram << std::endl;
        std::cout << "calcu_stream_size * 4 is " << (calcu_stream_size * 4) << std::endl;
        std::cout << "vld_core_num is " << vld_core_num << std::endl;
        std::cout << "find_max is " << find_max << std::endl;
        std::cout << "old_max is " << old_max << std::endl;
#endif
        ew_cfg_type = 2;
        out << "ew_cfg " << "rs=0" << " rt=0"
            << " rd=" << calcu_stream_size << " shamt=" << ew_cfg_type << std::endl;
        out << "dsdiv " << "rs=" << src0_addr_l1e_sram << " rt=" << src1_addr_l1e_sram
            << " rd=" << dst_addr_sram << " shamt=" << dsdiv_shamt << std::endl;
        if (i % 3 == 0) {
            uint32_t coprocessor_type = 3;
            out << "ld_sd " << "rs=0" << " rt=0" << " rd=0" << " shamt=" << coprocessor_type << std::endl;
            out << "ld_sd " << "rs=1" << " rt=0" << " rd=0" << " shamt=" << coprocessor_type << std::endl;
        }
    }

}

void gen_ew_fussion_rand(std::ostream& out) {

    float coeff0 = rand() / (double)(RAND_MAX) * MAX_RANGE - (MAX_RANGE / 2);
    float coeff1 = rand() / (double)(RAND_MAX) * MAX_RANGE - (MAX_RANGE / 2);
    float coeff2 = rand() / (double)(RAND_MAX) * MAX_RANGE - (MAX_RANGE / 2);
    uint32_t* coeff0_ptr = reinterpret_cast<uint32_t*>(&coeff0);
    uint32_t* coeff1_ptr = reinterpret_cast<uint32_t*>(&coeff1);
    uint32_t* coeff2_ptr = reinterpret_cast<uint32_t*>(&coeff2);
    uint32_t coeff0_i32 = *coeff0_ptr;
    uint32_t coeff1_i32 = *coeff1_ptr;
    uint32_t coeff2_i32 = *coeff2_ptr;

    uint32_t find_max = rand() % 2; // 1:findmax 0:bypass
    float old_max = rand() / (double)(RAND_MAX) * MAX_RANGE;
    uint32_t* old_max_ptr = reinterpret_cast<uint32_t*>(&old_max);
    uint32_t old_max_u32 = *old_max_ptr;

    uint32_t vld_core_num = rand() % 16 + 1;

    uint32_t active_type = rand() % 9; // 0:null 1:relu 2:sigmoid 3:tanh
    uint32_t dst_sram_id = rand() % 2; // 0:l2-e-sram 1:l1-e-sram
    uint32_t dsmadd_shamt = dst_sram_id;
    uint32_t dsmul_shamt = dst_sram_id;
    uint32_t dscmpnsel_op_type = rand() % 2; // dscmpnsel cmd: 0-max 1-min
    uint32_t dscmpnsel_shamt = (dscmpnsel_op_type << 2) | dst_sram_id;
    uint32_t dsdiv_mode = rand() % 3; // 0: scalar/vector 1: vector/scalar 2: vector/vector
    uint32_t dsdiv_shamt = (dsdiv_mode << 1) | dst_sram_id;

    uint32_t src_bank_l1e_sram = 0;
    uint32_t dst_bank_sram = 0;

    uint32_t coeff0_type = rand() % 2; //0:staitc scalar coeff   1:static scalar-vector coeff
    uint32_t coeff1_type = rand() % 2; //0:staitc scalar coeff   1:static scalar-vector coeff
    uint32_t coeff2_type = rand() % 2; //0:staitc scalar coeff   1:static scalar-vector coeff
    uint32_t ewcoeff_cfg_type = coeff0_type | (coeff1_type << 1) | (coeff2_type << 2); //coeff0 coeff1 coeff2

    uint32_t coeff0_vector_addr_sram = 0;
    uint32_t coeff1_vector_addr_sram = 0;
    uint32_t coeff2_vector_addr_sram = 0;
    uint32_t coeff0_vector_offset_l1e_sram = 0;
    uint32_t coeff1_vector_offset_l1e_sram = 0;
    uint32_t coeff2_vector_offset_l1e_sram = 0;

    if (coeff0_type == 0) {
        coeff0_vector_addr_sram = coeff0_i32;
    } else if (coeff0_type == 1) {
        coeff0_vector_offset_l1e_sram = (rand() % (kL1EBankValidSize / 4)) * 4;
        coeff0_vector_offset_l1e_sram = coeff0_vector_offset_l1e_sram > kL1EBankValidSize ? kL1EBankValidSize : coeff0_vector_offset_l1e_sram;
        coeff0_vector_addr_sram = (coeff0_vector_offset_l1e_sram >> 2) << (kL1EBankBits + 2) | (src_bank_l1e_sram << 2);
        assert(coeff0_vector_offset_l1e_sram <= kL1EBankValidSize);
    }
    if (coeff1_type == 0) {
        coeff1_vector_addr_sram = coeff1_i32;
    } else if (coeff1_type == 1) {
        coeff1_vector_offset_l1e_sram = (rand() % (kL1EBankValidSize / 4)) * 4;
        coeff1_vector_offset_l1e_sram = coeff1_vector_offset_l1e_sram > kL1EBankValidSize ? kL1EBankValidSize : coeff1_vector_offset_l1e_sram;
        coeff1_vector_addr_sram = (coeff1_vector_offset_l1e_sram >> 2) << (kL1EBankBits + 2) | (src_bank_l1e_sram << 2);
        assert(coeff1_vector_offset_l1e_sram <= kL1EBankValidSize);
    }
    if (coeff2_type == 0) {
        coeff2_vector_addr_sram = coeff2_i32;
    } else if (coeff2_type == 1) {
        coeff2_vector_offset_l1e_sram = (rand() % (kL1EBankValidSize / 4)) * 4;
        coeff2_vector_offset_l1e_sram = coeff2_vector_offset_l1e_sram > kL1EBankValidSize ? kL1EBankValidSize : coeff2_vector_offset_l1e_sram;
        coeff2_vector_addr_sram = (coeff2_vector_offset_l1e_sram >> 2) << (kL1EBankBits + 2) | (src_bank_l1e_sram << 2);
        assert(coeff2_vector_offset_l1e_sram <= kL1EBankValidSize);
    }
    out << "ewcoeff_cfg " << "rs=" << coeff1_vector_addr_sram << " rt=" << coeff2_vector_addr_sram
        << " rd=" << coeff0_vector_addr_sram << " shamt=" << ewcoeff_cfg_type << std::endl;

#if DEBUG
    std::cout << "ewcoeff_cfg_type is " << ewcoeff_cfg_type << std::endl;
    std::cout << "coeff0 is " << coeff0 << std::endl;
    std::cout << "coeff1 is " << coeff1 << std::endl;
    std::cout << "coeff2 is " << coeff2 << std::endl;
    std::cout << "active_type is " << active_type << std::endl;
    std::cout << "dst_sram_id is " << dst_sram_id << std::endl;
    std::cout << "dst_bank_sram is " << dst_bank_sram << std::endl;
    std::cout << "coeff2_vector_offset_l1e_sram is " << coeff2_vector_offset_l1e_sram << std::endl;
    std::cout << "coeff1_vector_offset_l1e_sram is " << coeff1_vector_offset_l1e_sram << std::endl;
    std::cout << "coeff0_vector_offset_l1e_sram is " << coeff0_vector_offset_l1e_sram << std::endl;
#endif

    uint32_t ew_cfg_type = 0;

    ew_cfg_type = 10;
    out << "ew_cfg " << "rs=0" << " rt=0"
        << " rd=" << active_type << " shamt=" << ew_cfg_type << std::endl;
    gen_ew_tab_cmd(out, active_type);

    int cmd_times= rand() % CMD_TIMES + 1;
    for (int i = 0; i < cmd_times; i++) {
        ew_cfg_type = 0;
        find_max = rand() % 2; // 1:findmax 0:bypass
        old_max = rand() / (double)(RAND_MAX) * MAX_RANGE;
        uint32_t* old_max_ptr = reinterpret_cast<uint32_t*>(&old_max);
        old_max_u32 = *old_max_ptr;
        out << "ew_cfg " << "rs=" << old_max_u32 << " rt=0"
            << " rd=" << find_max << " shamt=" << ew_cfg_type << std::endl;
        vld_core_num = rand() % 16 + 1;
        ew_cfg_type = 1;
        out << "ew_cfg " << "rs=0" << " rt=0"
            << " rd=" << vld_core_num << " shamt=" << ew_cfg_type << std::endl;

        uint32_t calcu_stream_size = 0;
        if (dst_sram_id == 0) {
            // l2esram
            calcu_stream_size = rand() % (kL2EBankSize / 4);
        } else if (dst_sram_id == 1) {
            // l1esram
            calcu_stream_size = rand() % (kL1EBankValidSize / 4);
        }

        uint32_t src0_offset_l1e_sram = (rand() % (kL1EBankValidSize / 4)) * 4;
        src0_offset_l1e_sram = (src0_offset_l1e_sram + calcu_stream_size * 4) > kL1EBankValidSize ? 0 : src0_offset_l1e_sram;
        uint32_t src0_offset_l1e_sram_end = src0_offset_l1e_sram  + calcu_stream_size * 4;
        uint32_t src0_addr_l1e_sram = (src0_offset_l1e_sram >> 2) << (kL1EBankBits + 2) | (src_bank_l1e_sram << 2);

        uint32_t src1_offset_l1e_sram = (rand() % (kL1EBankValidSize / 4)) * 4;
        src1_offset_l1e_sram = (src1_offset_l1e_sram + calcu_stream_size * 4) > kL1EBankValidSize ? 0 : src1_offset_l1e_sram;
        uint32_t src1_offset_l1e_sram_end = src1_offset_l1e_sram + calcu_stream_size * 4;
        uint32_t src1_addr_l1e_sram = (src1_offset_l1e_sram >> 2) << (kL1EBankBits + 2) | (src_bank_l1e_sram << 2);

        uint32_t dst_offset_sram = 0;
        uint32_t dst_addr_sram = 0;
        if (dst_sram_id == 0) {
            // l2esram
            dst_offset_sram = (rand() % (kL2EBankSize / 4)) * 4;
            dst_offset_sram = (dst_offset_sram + calcu_stream_size * 4) > kL2EBankSize ? 0 : dst_offset_sram;
        } else if (dst_sram_id == 1) {
            // l1esram
            uint32_t max_src_offset_l1e_sram_end = std::max(src0_offset_l1e_sram_end, src1_offset_l1e_sram_end);
            uint32_t min_src_offset_l1e_sram_start = std::min(src0_offset_l1e_sram, src1_offset_l1e_sram);
            if (max_src_offset_l1e_sram_end < kL1EBankValidSize) {
                dst_offset_sram = max_src_offset_l1e_sram_end + (rand() % (kL1EBankValidSize - max_src_offset_l1e_sram_end)) / 4 * 4;
            }
            if ((dst_offset_sram + calcu_stream_size * 4) >= kL1EBankValidSize) {
                dst_offset_sram = min_src_offset_l1e_sram_start;
            }
        }
        dst_addr_sram = ((dst_offset_sram >> 2) << (kL2EBankBits + 2)) | (dst_bank_sram << 2);

        assert(vld_core_num <= 16);
        assert(src0_offset_l1e_sram + calcu_stream_size * 4 <= kL1EBankValidSize);
        assert(src1_offset_l1e_sram + calcu_stream_size * 4 <= kL1EBankValidSize);
#if DEBUG
        std::cout << std::endl;
        std::cout << "calcu_stream_size is " << calcu_stream_size << std::endl;
        std::cout << "src0_offset_l1e_sram is " << src0_offset_l1e_sram << std::endl;
        std::cout << "src1_offset_l1e_sram is " << src1_offset_l1e_sram << std::endl;
        std::cout << "dst_offset_sram is " << dst_offset_sram << std::endl;
        std::cout << "calcu_stream_size * 4 is " << (calcu_stream_size * 4) << std::endl;
        std::cout << "vld_core_num is " << vld_core_num << std::endl;
        std::cout << "find_max is " << find_max << std::endl;
        std::cout << "old_max is " << old_max << std::endl;
#endif

        ew_cfg_type = 2;
        out << "ew_cfg " << "rs=0" << " rt=0"
            << " rd=" << calcu_stream_size << " shamt=" << ew_cfg_type << std::endl;
        out << "dsmadd " << "rs=" << src0_addr_l1e_sram << " rt=" << src1_addr_l1e_sram
            << " rd=" << dst_addr_sram << " shamt=" << dsmadd_shamt << std::endl;
        out << "dsmul " << "rs=" << src0_addr_l1e_sram << " rt=" << src1_addr_l1e_sram
            << " rd=" << dst_addr_sram << " shamt=" << dsmul_shamt << std::endl;
        out << "dscmpnsel " << "rs=" << src0_addr_l1e_sram << " rt=" << src1_addr_l1e_sram
            << " rd=" << dst_addr_sram << " shamt=" << dscmpnsel_shamt << std::endl;
        out << "dsdiv " << "rs=" << src0_addr_l1e_sram << " rt=" << src1_addr_l1e_sram
            << " rd=" << dst_addr_sram << " shamt=" << dsdiv_shamt << std::endl;
        if (i % 3 == 0) {
            uint32_t coprocessor_type = 3;
            out << "ld_sd " << "rs=0" << " rt=0" << " rd=0" << " shamt=" << coprocessor_type << std::endl;
            out << "ld_sd " << "rs=1" << " rt=0" << " rd=0" << " shamt=" << coprocessor_type << std::endl;
        }
    }
}

int main(int argc, char** argv) {
    google::ParseCommandLineFlags(&argc, &argv, true);

    srand(FLAGS_seed);

    std::ofstream cmd_seq;
    cmd_seq.open(FLAGS_output);

    uint32_t module_id = 3;
    cmd_seq << "lock " <<  module_id << std::endl;

    if (FLAGS_func == "dsmadd") {
        gen_ew_dsmadd_rand(cmd_seq);
    }
    else if (FLAGS_func == "dsmul") {
        gen_ew_dsmul_rand(cmd_seq);
    }
    else if (FLAGS_func == "dscmpnsel") {
        gen_ew_dscmpnsel_rand(cmd_seq);
    }
    else if (FLAGS_func == "sspooling") {
        gen_ew_sspooling(cmd_seq);
    }
    else if (FLAGS_func == "ssvsum") {
        gen_ew_ssvsum_rand(cmd_seq);
    } else if (FLAGS_func == "dsdiv") {
        gen_ew_dsdiv_rand(cmd_seq);
    } else if (FLAGS_func == "ssresize") {
        gen_ew_ssresize(cmd_seq);
    } else if (FLAGS_func == "mix") {
        gen_ew_dsmadd_rand(cmd_seq);
        gen_ew_dsmul_rand(cmd_seq);
        gen_ew_dscmpnsel_rand(cmd_seq);
        gen_ew_dsdiv_rand(cmd_seq);
        gen_ew_sspooling(cmd_seq);
        gen_ew_ssvsum_rand(cmd_seq);
        gen_ew_ssresize(cmd_seq);
    } else if (FLAGS_func == "fussion") {
        gen_ew_fussion_rand(cmd_seq);
    }

    cmd_seq << "xfence" << std::endl;
    cmd_seq << "unlock " << module_id << std::endl;

    cmd_seq.close();

    return 0;
}
