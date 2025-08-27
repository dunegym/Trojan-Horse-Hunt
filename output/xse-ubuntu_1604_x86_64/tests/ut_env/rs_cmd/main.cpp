#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <gflags/gflags.h>
#include <assert.h>

DEFINE_int32(core_id, 0, "number of core_running");
DEFINE_string(output, "./rs_cmd_seq.dat", "rs_cmd_file");
DEFINE_string(func, "rs_col", "cmd  type");
DEFINE_string(mode, "single", "bath type");
DEFINE_int32(seed, 0, "random seed");
DEFINE_bool(alias_dbg, false, "print info for debugging address aliasing across cores");
DEFINE_int32(num_instr, 0, "Specify how many rs command to generate");

#define L2E_DEPTH 2048
#define L2R_DEPTH 2048
#define BANK_NUM 16

void gen_single_mode_row(std::ostream& out, int32_t core_id) {
    uint32_t src_addr = 0;
    uint32_t dst_addr = 0;
    uint32_t len = 0;
    uint32_t loop = 0;
    src_addr = (rand() % L2E_DEPTH) * (4 * BANK_NUM);
    dst_addr = (rand() % L2R_DEPTH) * (4 * BANK_NUM);
    len = rand() % 16 + 1;
    assert(src_addr % (4 * 16) == 0);
    assert(dst_addr % (4 * 16) == 0);
    assert((src_addr >= 0) && (src_addr < (L2E_DEPTH * BANK_NUM * 4)));
    assert((dst_addr >= 0) && (dst_addr < (L2R_DEPTH * BANK_NUM * 4)));

    assert((len >= 1) && (len <= 16));

    out << "rs_row " << "rs=" << src_addr << " rt=" << dst_addr << " rd=" << len << " shamt=0" <<
        std::endl;
}

void gen_single_mode_col(std::ostream& out, int32_t core_id) {
    uint32_t src_addr = 0;
    uint32_t dst_addr = 0;
    uint32_t len = 0;
    uint32_t loop = 0;
    uint32_t dst_row_stride = 0;
    uint32_t dst_col_stride = 0;
    uint32_t src_row_stride = 0;

    src_addr = rand() % (L2E_DEPTH) * BANK_NUM * 4;
    dst_addr = rand() % (L2R_DEPTH) * BANK_NUM * 4 ;
    len = rand() % 16 + 1;
    dst_row_stride = rand() % L2R_DEPTH;

    assert(src_addr % (4) == 0);
    assert(dst_addr % (4) == 0);
    assert((src_addr >= 0) && (src_addr < (L2E_DEPTH * BANK_NUM * 4)));
    assert((dst_addr >= 0) && (dst_addr < (L2R_DEPTH * BANK_NUM * 4)));
    assert((len >= 1) && (len <= 16));
    out << "rs_cfg " << "rs=" << src_row_stride << " rt=" << dst_row_stride << " rd=" << dst_col_stride
        << " shamt=2"
        << std::endl;
    out << "rs_col " << "rs=" << src_addr << " rt=" << dst_addr << " rd=" << len << " shamt=0" <<
        std::endl;
}

void gen_batch_mode_row(std::ostream& out, int32_t core_id) {
    uint32_t src_addr = 0;
    uint32_t dst_addr = 0;
    uint32_t len = 0;
    uint32_t loop = 0;
    uint32_t dst_row_stride = 0;
    uint32_t dst_col_stride = 0;
    uint32_t src_row_stride = 0;

    src_addr = (rand() % L2E_DEPTH) * (4 * BANK_NUM);
    dst_addr = (rand() % L2R_DEPTH) * (4 * BANK_NUM);
    len = rand() % 16 + 1;
    dst_row_stride = rand() % L2R_DEPTH;
    dst_col_stride = 0;
    src_row_stride = rand() % L2E_DEPTH;
    loop = rand() % std::min(L2R_DEPTH, L2E_DEPTH) + 1;


    assert(src_addr % (4 * BANK_NUM) == 0);
    assert(dst_addr % (4 * BANK_NUM) == 0);
    assert((src_addr >= 0) && (src_addr < (L2E_DEPTH * BANK_NUM * 4)));
    assert((dst_addr >= 0) && (dst_addr < (L2R_DEPTH * BANK_NUM * 4)));
    assert((len >= 1) && (len <= 16));
    assert((loop >= 1) && (loop <= std::min(L2R_DEPTH, L2E_DEPTH)));
    assert((src_row_stride >= 0) && (src_row_stride < L2E_DEPTH));
    assert((dst_row_stride >= 0) && (dst_row_stride < L2R_DEPTH));
    assert(dst_col_stride == 0);

    out << "rs_cfg " << "rs=" << src_row_stride << " rt=" << dst_row_stride << " rd=" << dst_col_stride
        << " shamt=1"
        << std::endl;
    out << "rs_cfg " << "rs=" << loop << " rt=0" << " rd=0" << " shamt=3" << std::endl;
    out << "rs_row " << "rs=" << src_addr << " rt=" << dst_addr << " rd=" << len << " shamt=16" <<
        std::endl;
}

void gen_batch_mode_col(std::ostream& out, int32_t core_id) {
    uint32_t src_addr = 0;
    uint32_t dst_addr = 0;
    uint32_t len = 0;
    uint32_t loop = 0;
    uint32_t dst_row_stride = 0;
    uint32_t dst_col_stride = 0;
    uint32_t src_row_stride = 0;

    src_addr = rand() % (L2E_DEPTH) * BANK_NUM * 4;
    dst_addr = rand() % (L2R_DEPTH) * BANK_NUM * 4  ;
    len = rand() % 16 + 1;

    dst_row_stride = rand() % L2R_DEPTH;
    src_row_stride = rand() % L2E_DEPTH;
    dst_col_stride = rand() % BANK_NUM;

    loop = rand() % std::min(L2R_DEPTH, L2E_DEPTH) + 1;

    assert(src_addr % 4 == 0);
    assert(dst_addr % 4 == 0);

    assert((src_addr >= 0) && (src_addr < (L2E_DEPTH * BANK_NUM * 4)));
    assert((dst_addr >= 0) && (dst_addr < (L2R_DEPTH * BANK_NUM * 4)));
    assert((len >= 1) && (len <= 16));
    assert((loop >= 1) && (loop <= std::min(L2R_DEPTH, L2E_DEPTH)));

    out << "rs_cfg " << "rs=" << src_row_stride << " rt=" << dst_row_stride << " rd=" << dst_col_stride
        << " shamt=2"
        << std::endl;
    out << "rs_cfg " << "rs=" << loop << " rt=0" << " rd=0" << " shamt=4" << std::endl;
    out << "rs_col " << "rs=" << src_addr << " rt=" << dst_addr << " rd=" << len << " shamt=16" <<
        std::endl;
}

#define MAX_ITER 25
void gen_complex(std::ostream& out, int32_t core_id) {
    uint32_t src_addr = 0;
    uint32_t dst_addr = 0;
    uint32_t len = 0;
    uint32_t loop = 0;
    uint32_t dst_row_stride = 0;
    uint32_t dst_col_stride = 0;
    uint32_t src_row_stride = 0;

    uint32_t times = rand() % MAX_ITER;
    bool initialized = false;

    uint32_t need_recfg = 0;
    uint32_t row_or_col = 0;

    if (!initialized) {
        dst_row_stride = rand() % L2R_DEPTH;
        src_row_stride = rand() % L2E_DEPTH;
        dst_col_stride = 0;
        out << "rs_cfg " << "rs=" << src_row_stride << " rt=" << dst_row_stride << " rd=" << dst_col_stride
            << " shamt=1" << std::endl;

        dst_row_stride = rand() % L2R_DEPTH;
        src_row_stride = rand() % L2E_DEPTH;
        dst_col_stride = rand() % BANK_NUM;
        out << "rs_cfg " << "rs=" << src_row_stride << " rt=" << dst_row_stride << " rd=" << dst_col_stride
            << " shamt=2" << std::endl;

        loop = rand() % std::min(L2R_DEPTH, L2E_DEPTH) + 1;
        out << "rs_cfg " << "rs=" << loop << " rt=0" << " rd=0" << " shamt=3" << std::endl;

        loop = rand() % std::min(L2R_DEPTH, L2E_DEPTH) + 1;
        out << "rs_cfg " << "rs=" << loop << " rt=0" << " rd=0" << " shamt=4" << std::endl;
        initialized = true;
    }

    for (int idx = 0; idx < times; ++idx) {
        src_addr = rand() % (L2E_DEPTH) * BANK_NUM * 4;
        dst_addr = rand() % (L2R_DEPTH) * BANK_NUM * 4;
        len = rand() % 16 + 1;

        dst_row_stride = rand() % L2R_DEPTH;
        src_row_stride = rand() % L2E_DEPTH;

        loop = rand() % std::min(L2R_DEPTH, L2E_DEPTH) + 1;

        need_recfg = rand() % 10;
        row_or_col = rand() % 2;

        assert(src_addr % 4 == 0);
        assert(dst_addr % 4 == 0);

        assert((src_addr >= 0) && (src_addr < (L2E_DEPTH * BANK_NUM * 4)));
        assert((dst_addr >= 0) && (dst_addr < (L2R_DEPTH * BANK_NUM * 4)));
        assert((len >= 1) && (len <= 16));
        assert((loop >= 1) && (loop <= std::min(L2R_DEPTH, L2E_DEPTH)));

        if (need_recfg == 0) {
            if (row_or_col == 0) {
                dst_col_stride = 0;
                out << "rs_cfg " << "rs=" << src_row_stride << " rt=" << dst_row_stride << " rd=" << dst_col_stride
                    << " shamt=1" << std::endl;
                out << "rs_cfg " << "rs=" << loop << " rt=0" << " rd=0" << " shamt=3" << std::endl;
            } else if (row_or_col == 1) {
                dst_col_stride = rand() % BANK_NUM;
                out << "rs_cfg " << "rs=" << src_row_stride << " rt=" << dst_row_stride << " rd=" << dst_col_stride
                    << " shamt=2" << std::endl;
                out << "rs_cfg " << "rs=" << loop << " rt=0" << " rd=0" << " shamt=4" << std::endl;
            }
        }

        if (row_or_col == 0) {
            out << "rs_row " << "rs=" << src_addr << " rt=" << dst_addr << " rd=" << len << " shamt=16" <<
                std::endl;
        } else if (row_or_col == 1) {
            out << "rs_col " << "rs=" << src_addr << " rt=" << dst_addr << " rd=" << len << " shamt=16" <<
                std::endl;
        }
    }
}

#if 0
#define BANK_BITS 6
#define ALIGN_BITS 2
uint32_t unified_hw_addr(uint32_t bank, uint32_t offset, int bank_bits = BANK_BITS,
                         int align_bits = ALIGN_BITS) {
    uint32_t off_lo = offset & ((1 << align_bits) - 1);
    uint32_t off_hi = (offset >> align_bits) << (bank_bits + align_bits);
    uint32_t bank_f = bank << align_bits;
    uint32_t addr = off_hi | bank_f | off_lo;
    return addr;
}

uint32_t addr_bank(uint32_t addr, int bank_bits = BANK_BITS, int align_bits = ALIGN_BITS) {
    uint32_t bank = (addr >> align_bits) & ((1 << bank_bits) - 1);
    return bank;
}

uint32_t addr_off(uint32_t addr, int bank_bits = BANK_BITS, int align_bits = ALIGN_BITS) {
    uint32_t off_lo = addr & ((1 << bank_bits) - 1);
    uint32_t off_hi = (addr >> (bank_bits + align_bits)) << align_bits;
    uint32_t offset = off_hi | off_lo;
    return offset;
}

// limit the data addr of destination into sections according to core_id (with dst_row_stride)
void gen_multi_core(std::ostream& out, int32_t core_id) {
    uint32_t src_addr = 0;
    uint32_t dst_addr = 0;
    uint32_t len = 0;
    uint32_t row_loop = 0;
    uint32_t col_loop = 0;
    uint32_t row_dst_row_stride = 0;
    uint32_t row_dst_col_stride = 0;
    uint32_t row_src_row_stride = 0;
    uint32_t col_dst_row_stride = 0;
    uint32_t col_dst_col_stride = 0;
    uint32_t col_src_row_stride = 0;
    uint32_t stride_x_loop = 0;

    uint32_t times = rand() % MAX_ITER;

    if (FLAGS_num_instr != 0) {
        times = std::min(times, (uint32_t)FLAGS_num_instr);
    }

    bool initialized = false;

    uint32_t need_recfg = 0;
    uint32_t row_or_col = 0;

    if (!initialized) {
        stride_x_loop = rand() % (2048 / 4);
        row_dst_row_stride = rand() % (2048 / 4);
        row_dst_row_stride = std::min(row_dst_row_stride, stride_x_loop);
        row_src_row_stride = rand() % 2048;
        row_dst_col_stride = (rand() % 4) * 16;
        out << "rs_cfg " << "rs=" << row_src_row_stride << " rt=" << row_dst_row_stride << " rd=" <<
            row_dst_col_stride
            << " shamt=1" << std::endl;

        if (row_dst_row_stride != 0) {
            row_loop = std::min(stride_x_loop / row_dst_row_stride, stride_x_loop % 254 + 1);
        } else {
            row_loop = stride_x_loop % 254 + 1;
        }

        out << "rs_cfg " << "rs=" << row_loop << " rt=0" << " rd=0" << " shamt=3" << std::endl;
        //std::cout << "stride_x_loop = " << stride_x_loop << std::endl;
        //std::cout << "row_dst_row_stride = " << row_dst_row_stride << std::endl;
        //std::cout << "row_loop = " << row_loop << std::endl;

        col_dst_row_stride = rand() % (2048 / 4);
        col_dst_row_stride = std::min(col_dst_row_stride, stride_x_loop);
        col_src_row_stride = rand() % 2048;
        col_dst_col_stride = rand() % 64;
        out << "rs_cfg " << "rs=" << col_src_row_stride << " rt=" << col_dst_row_stride << " rd=" <<
            col_dst_col_stride
            << " shamt=2" << std::endl;

        if (col_dst_row_stride != 0) {
            col_loop = std::min(stride_x_loop / col_dst_row_stride, stride_x_loop % 254 + 1);
        } else {
            col_loop = stride_x_loop % 254 + 1;
        }

        out << "rs_cfg " << "rs=" << col_loop << " rt=0" << " rd=0" << " shamt=4" << std::endl;
        initialized = true;
    }

    for (int idx = 0; idx < times; ++idx) {

        need_recfg = rand() % 10;

        if (need_recfg == 0) {
            stride_x_loop = rand() % (2048 / 4);
            row_dst_row_stride = rand() % (2048 / 4);
            row_dst_row_stride = std::min(row_dst_row_stride, stride_x_loop);
            row_src_row_stride = rand() % 2048;

            if (row_dst_row_stride != 0) {
                row_loop = std::min(stride_x_loop / row_dst_row_stride, stride_x_loop % 254 + 1);
            } else {
                row_loop = stride_x_loop % 254 + 1;
            }

            //std::cout << "loop = " << row_loop << " stride_x_loop = " << stride_x_loop << " dst_row_stride = " << row_dst_row_stride << std::endl;
            assert((row_loop >= 1) && (row_loop <= 255));
            row_dst_col_stride = (rand() % 4) * 16;
            out << "rs_cfg " << "rs=" << row_src_row_stride << " rt=" << row_dst_row_stride << " rd=" <<
                row_dst_col_stride
                << " shamt=1" << std::endl;
            out << "rs_cfg " << "rs=" << row_loop << " rt=0" << " rd=0" << " shamt=3" << std::endl;
            col_dst_row_stride = rand() % (2048 / 4);
            col_dst_row_stride = std::min(col_dst_row_stride, stride_x_loop);
            col_src_row_stride = rand() % 2048;

            if (col_dst_row_stride != 0) {
                col_loop = std::min(stride_x_loop / col_dst_row_stride, stride_x_loop % 254 + 1);
            } else {
                col_loop = stride_x_loop % 254 + 1;
            }

            //std::cout << "loop = " << col_loop << " stride_x_loop = " << stride_x_loop << " dst_row_stride = " << col_dst_row_stride << std::endl;
            assert((col_loop >= 1) && (col_loop <= 255));
            col_dst_col_stride = rand() % 64;
            out << "rs_cfg " << "rs=" << col_src_row_stride << " rt=" << col_dst_row_stride << " rd=" <<
                col_dst_col_stride
                << " shamt=2" << std::endl;
            out << "rs_cfg " << "rs=" << col_loop << " rt=0" << " rd=0" << " shamt=4" << std::endl;
        }

        uint32_t dst_watermark = std::min(stride_x_loop, (uint32_t)(2048 / 4 - 1));
        //std::cout << "stride_x_loop = " << stride_x_loop << std::endl;
        //std::cout << "dst_watermark = " << dst_watermark << std::endl;
        src_addr = (rand() % (2048 / 4) + core_id * (2048 / 4)) * 64 * 4 + core_id * 4 * 16;
        dst_addr = (rand() % (2048 / 4 - dst_watermark) + core_id * (2048 / 4)) * 64 * 4  ;
        len = rand() % 16 + 1;

        row_or_col = rand() % 2;

        assert(src_addr % (4 * 64) / (4 * 16) == core_id);
        assert(src_addr % 4 == 0);
        assert(dst_addr % 4 == 0);

        assert((src_addr >= 0) && (src_addr < (2048 * 64 * 4)));
        assert((dst_addr >= 0) && (dst_addr < (2048 * 64 * 4)));
        assert((len >= 1) && (len <= 16));

        if (row_or_col == 0) {
            out << "rs_row " << "rs=" << src_addr << " rt=" << dst_addr << " rd=" << len << " shamt=16" <<
                std::endl;

            if (FLAGS_alias_dbg) {
                uint32_t src_bank = addr_bank(src_addr, 6, 2);
                uint32_t src_off = addr_off(src_addr, 6, 2);
                uint32_t dst_bank = addr_bank(dst_addr, 6, 2);
                uint32_t dst_off = addr_off(dst_addr, 6, 2);
                std::cout << "src " << src_addr << " bank " << src_bank << " offset " << src_off / 4 << std::endl;
                std::cout << "dst " << dst_addr << " bank " << dst_bank << " offset " << dst_off / 4 << std::endl;
                std::cout << "src_row_stride " << row_src_row_stride << " dst_row_stride " <<
                          row_dst_row_stride << " dst_col_stride " << row_dst_col_stride << std::endl;
                std::cout << "row_loop  " << row_loop << std::endl;
                std::cout << "rs_row " << "rs=" << src_addr << " rt=" << dst_addr << " rd=" << len << " shamt=16" <<
                          std::endl;
                std::cout << "rs_row on core " << core_id << std::endl;

                for (int li = 0; li < row_loop; ++li) {
                    std::cout << "\t read " << len << " elements from (bank = " << (src_bank % 64) <<
                              ", offset = " << ((src_off / 4 + row_src_row_stride * li) % 2048) << ")" << std::endl;

                    for (int bi = 0; bi < len; ++bi) {
                        std::cout << "\t\twrite to (bank =" << ((dst_bank + row_dst_col_stride * li + bi) %
                                                                64) << ", offset = " << ((dst_off / 4 + row_dst_row_stride * li) % 2048) << ")" << std::endl;
                        assert(((dst_off / 4 + row_dst_row_stride * li) % 2048) < (core_id + 1) * 512);
                        assert(((dst_off / 4 + row_dst_row_stride * li) % 2048) >= (core_id) * 512);
                    }
                }
            }
        } else if (row_or_col == 1) {
            if (col_dst_row_stride > 0) {
                uint32_t dst_bank = addr_bank(dst_addr, 6, 2);
                uint32_t dst_off = addr_off(dst_addr, 6, 2);
                //std::cout << "dst " << dst_addr << " bank " << dst_bank << " offset " << dst_off / 4 << std::endl;
                //std::cout << "rest in offset is " << (2048 / 4 - dst_off / 4) << std::endl;
                len = std::min(len, (((2048 / 4) * (core_id + 1) - dst_off / 4 + col_dst_row_stride - 1) /
                                     col_dst_row_stride));
            }

            out << "rs_col " << "rs=" << src_addr << " rt=" << dst_addr << " rd=" << len << " shamt=16" <<
                std::endl;

            if (FLAGS_alias_dbg) {
                uint32_t src_bank = addr_bank(src_addr, 6, 2);
                uint32_t src_off = addr_off(src_addr, 6, 2);
                uint32_t dst_bank = addr_bank(dst_addr, 6, 2);
                uint32_t dst_off = addr_off(dst_addr, 6, 2);
                std::cout << "src " << src_addr << " bank " << src_bank << " offset " << src_off / 4 << std::endl;
                std::cout << "dst " << dst_addr << " bank " << dst_bank << " offset " << dst_off / 4 << std::endl;
                std::cout << "src_row_stride " << col_src_row_stride << " dst_row_stride " <<
                          col_dst_row_stride << " dst_col_stride " << col_dst_col_stride << std::endl;
                std::cout << "col_loop  " << col_loop << std::endl;
                std::cout << "rs_col " << "rs=" << src_addr << " rt=" << dst_addr << " rd=" << len << " shamt=16" <<
                          std::endl;
                std::cout << "rs_col on core " << core_id << std::endl;

                for (int li = 0; li < col_loop; ++li) {
                    std::cout << "\t read " << len << " elements from (bank = " << (src_bank % 64) <<
                              ", offset = " << ((src_off / 4 + col_src_row_stride * li) % 2048) << ")" << std::endl;

                    for (int bi = 0; bi < len; ++bi) {
                        std::cout << "\t\twrite to (bank =" << ((dst_bank + col_dst_col_stride * li) %
                                                                64) << ", offset = " << ((dst_off / 4 + col_dst_row_stride * bi) % 2048) << ")" << std::endl;
                        assert(((dst_off / 4 + col_dst_row_stride * bi) % 2048) < (core_id + 1) * 512);
                        assert(((dst_off / 4 + col_dst_row_stride * bi) % 2048) >= (core_id) * 512);
                    }
                }
            }
        }
    }
}
#endif

int main(int argc, char** argv) {
    google::ParseCommandLineFlags(&argc, &argv, true);

    srand(FLAGS_seed);

    std::ofstream cmd_seq;
    cmd_seq.open(FLAGS_output);

    if (FLAGS_mode == "single") {
        if (FLAGS_func == "rs_row") {
            gen_single_mode_row(cmd_seq, FLAGS_core_id);
        } else {
            gen_single_mode_col(cmd_seq, FLAGS_core_id);
        }
    } else if (FLAGS_mode == "batch") {
        if (FLAGS_func == "rs_row") {
            gen_batch_mode_row(cmd_seq, FLAGS_core_id);
        } else {
            gen_batch_mode_col(cmd_seq, FLAGS_core_id);
        }
    } else if (FLAGS_mode == "complex") {
        gen_complex(cmd_seq, FLAGS_core_id);
    }
#if 0
    else if (FLAGS_mode == "multi-core") {
        gen_multi_core(cmd_seq, FLAGS_core_id);
    }
#endif

    cmd_seq << "xfence" << std::endl;
    cmd_seq.close();

    return 0;
}
