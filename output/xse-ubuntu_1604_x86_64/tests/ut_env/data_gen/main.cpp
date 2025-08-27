#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <gflags/gflags.h>

static constexpr std::size_t kFpBankDepth = 2048;
static constexpr std::size_t kFpBankNum = 16;

static constexpr std::size_t kL2DBankSize = (2048 * 512 / 8);
static constexpr std::size_t kL2DBankNum = 1;
static constexpr std::size_t kL2WBankSize = (2048 * 512 / 8);
static constexpr std::size_t kL2WBankNum = 1;
static constexpr std::size_t kL2EBankSize = (kFpBankDepth * 32 / 8);
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
static constexpr std::size_t kL1EBankNum = kFpBankNum;

static constexpr std::size_t kGlobalMemorySize = 4 * 1024 * 1024;

static constexpr unsigned int MAX_RANGE = 10000;

DEFINE_string(hbm_file, "./hbm_init.dat", "HBM image file");
DEFINE_string(l2w_file, "./l2w_sram_init.dat", "l2w sram image file");
DEFINE_string(l2d_file, "./l2d_sram_init.dat", "l2d sram image file");
DEFINE_string(l1w_file, "./l1w_sram_init.dat", "l1w sram image file");
DEFINE_string(l1d_file, "./l1d_sram_init.dat", "l1d sram image file");
DEFINE_string(l1e_file, "./l1e_sram_init.dat", "l1e sram image file");
DEFINE_string(l2e_file, "./l2e_sram_init.dat", "l2e sram image file");
DEFINE_string(l2r_file, "./l2r_sram_init.dat", "l2r sram image file");
DEFINE_string(data_type, "8bit", "data type 8bit/16bit");
DEFINE_int32(seed, 0, "random seed");

typedef union {
    int16_t n16;
    int8_t n8[2];
} npack_t;

/* rand a float in [-range, range]*/
static float rand_float(unsigned long range) {
    float fv = ((double)rand() / (double)RAND_MAX -  0.5) * 2 * (double)range;
    return fv;
}

int main(int argc, char** argv) {
    google::ParseCommandLineFlags(&argc, &argv, true);
    std::ofstream out;
    npack_t pack_tmp;

    std::vector<int32_t> vec_hbm;
    std::vector<int8_t> vec_l1w;
    std::vector<int8_t> vec_l1d;
    std::vector<int32_t> vec_l1e;
    std::vector<int32_t> vec_l2e;
    std::vector<int32_t> vec_l2r;

    vec_hbm.resize(kGlobalMemorySize / 4);
    vec_l1w.resize(kL1WBankSize);
    vec_l1d.resize(kL1DBankSize);
    vec_l1e.resize(kL1EBankSize);
    vec_l2e.resize(kL2EBankSize);
    vec_l2r.resize(kL2RBankSize);

    srand(FLAGS_seed);

    for (int i = 0; i < vec_hbm.size(); i++) {
        float tmp = (rand() - (RAND_MAX / 2 )) * 2 / (double)(RAND_MAX / 10000.00);
        int32_t* tmp_ptr = reinterpret_cast<int32_t*>(&tmp);
        vec_hbm[i]  = *tmp_ptr;
    }

    out.open(FLAGS_hbm_file);
    uint32_t hbm_addr = 0;
    out << "ADDRESS:" << std::setw(8) << std::hex << std::setfill('0') << hbm_addr << ";";
    out << "LENGTH:" << std::setw(8) << std::hex << std::setfill('0') << kGlobalMemorySize;
    out << std::endl;
    for (int i = 0; i < vec_hbm.size(); i++) {
        int32_t a = *((int32_t*)(&vec_hbm[i]));
        out << std::setw(8) << std::hex << std::setfill('0') << a << std::endl;
    }
    out.close();

    int32_t l1e_size = (kL1EBankNum * kL1EBankSize / 4);
    int32_t l2e_size = (kL2EBankNum * kL2EBankSize / 4);
    int32_t l2r_size = (kL2RBankNum * kL2RBankSize / 4);
    int32_t l1d_size = 0;
    int32_t l1w_size = 0;

    l1d_size = (kL1DBankNum * kL1DBankSize);
    l1w_size = (kL1WBankNum * kL1WBankSize);

    int32_t l1d_per_row_in_byte = 512;
    int32_t l1w_per_row_in_byte = 512;

    if (FLAGS_data_type == "8bit") {
        //L2DSram
        out.open(FLAGS_l2d_file);
        for (int i = 0; i < kL2DBankSize / 64; i++) {
            for (int j = 0; j < 64; j += 2) {
                pack_tmp.n8[0] = (rand() % 255) - 127;
                pack_tmp.n8[1] = (rand() % 255) - 127;
                out << std::setw(4) << std::hex << std::setfill('0') << pack_tmp.n16;
            }
            out << std::endl;
        }
        out.close();

        //L2WSram
        out.open(FLAGS_l2w_file);
        for (int i = 0; i < kL2WBankSize / 64; i++) {
            for (int j = 0; j < 64; j += 2) {
                pack_tmp.n8[0] = (rand() % 255) - 127;
                pack_tmp.n8[1] = (rand() % 255) - 127;
                out << std::setw(4) << std::hex << std::setfill('0') << pack_tmp.n16;
            }
            out << std::endl;
        }
        out.close();

        out.open(FLAGS_l1d_file);
        for (int i = 0; i < l1d_size / l1d_per_row_in_byte; i++) {
            for (int j = 0; j < l1d_per_row_in_byte / 2; j++) {
                npack_t a;
                // HW req: -128 shoule be avoided, [-127, 127] accepted
                a.n8[0] = (rand() % 255) - 127;
                a.n8[1] = (rand() % 255) - 127;
                out << std::setw(4) << std::hex << std::setfill('0') << a.n16;
            }
            out << std::endl;
        }
        out.close();

        out.open(FLAGS_l1w_file);
        for (int i = 0; i < l1w_size / l1w_per_row_in_byte; i++) {
            for (int j = 0; j < l1w_per_row_in_byte / 2; j++) {
                npack_t a;
                a.n8[0] = (rand() % 255) - 127;
                a.n8[1] = (rand() % 255) - 127;
                out << std::setw(4) << std::hex << std::setfill('0') << a.n16;
            }
            out << std::endl;
        }
        out.close();

    } else if (FLAGS_data_type == "16bit") {
        //L2DSram
        out.open(FLAGS_l2d_file);
        for (int i = 0; i < kL2DBankSize / 32; i++) {
            for (int j = 0; j < 32; j++) {
                pack_tmp.n16 = (rand() % 65535) - 32767;
                out << std::setw(4) << std::hex << std::setfill('0') << pack_tmp.n16;
            }
            out << std::endl;
        }
        out.close();

        //L2WSram
        out.open(FLAGS_l2w_file);
        for (int i = 0; i < kL2WBankSize / 32; i++) {
            for (int j = 0; j < 32; j++) {
                pack_tmp.n16 = (rand() % 65535) - 32767;
                out << std::setw(4) << std::hex << std::setfill('0') << pack_tmp.n16;
            }
            out << std::endl;
        }
        out.close();

        out.open(FLAGS_l1d_file);
        for (int i = 0; i < l1d_size / l1d_per_row_in_byte; i++) {
            for (int j = 0; j < l1d_per_row_in_byte / 2; j++) {
                // HW req: -32768 shoule be avoided, [-32767, 32767] accepted
                int16_t a = (rand() % 65535) - 32767;
                out << std::setw(4) << std::hex << std::setfill('0') << a;
            }
            out << std::endl;
        }
        out.close();

        out.open(FLAGS_l1w_file);
        for (int i = 0; i < l1w_size / l1w_per_row_in_byte; i++) {
            for (int j = 0; j < l1w_per_row_in_byte / 2; j++) {
                int16_t a = (rand() % 65535) - 32767;
                out << std::setw(4) << std::hex << std::setfill('0') << a;
            }
            out << std::endl;
        }
        out.close();
    }

    int32_t l1e_per_row_num = 16;
    out.open(FLAGS_l1e_file);

    for (int i = 0; i < (l1e_size / l1e_per_row_num); i++) {
        for (int j = 0; j < l1e_per_row_num; j++) {
            float a = rand_float(MAX_RANGE);
            out << std::setw(8) << std::hex << std::setfill('0') << *((uint32_t *)&a);
        }
        out << std::endl;
    }

    out.close();

    int32_t l2e_per_row_num = 16;
    out.open(FLAGS_l2e_file);

    for (int i = 0; i < (l2e_size / l2e_per_row_num); i++) {
        for (int j = 0; j < l2e_per_row_num; j++) {
            float a = rand_float(MAX_RANGE);
            out << std::setw(8) << std::hex << std::setfill('0') << *((uint32_t *)&a);
        }
        out << std::endl;
    }

    out.close();

    int32_t l2r_per_row_num = 16;
    out.open(FLAGS_l2r_file);

    for (int i = 0; i < (l2r_size / l2r_per_row_num); i++) {
        for (int j = 0; j < l2r_per_row_num; j++) {
            float a = rand_float(MAX_RANGE);
            out << std::setw(8) << std::hex << std::setfill('0') << *((uint32_t *)&a);
        }
        out << std::endl;
    }

    out.close();
    return 0;
}
