#include <iostream>
#include <fstream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <gflags/gflags.h>
#include <assert.h>

// Per Bank: L2D Size == L2W Size
// Per Bank: L2D/L2W Size > L1W Size > L1D Size > L1E Size
// L2D/L2D Depth 2048
// L1W Depth 4096
// L1D Depth 4096
// L1E Depth 4096
static constexpr std::size_t kFpBankDepth = 2048;
static constexpr std::size_t kFpBankNum = 16;

static constexpr std::size_t kL2WDBankSize = (2048 * 512 / 8); // L2D/L2W
static constexpr std::size_t kL1WDBankNum = 16;
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
static constexpr std::size_t kL1DBankSize = (2 * 1024 * 128 / 8); //32768
static constexpr std::size_t kL1DBankNum = (1 << kL1DBankBits);
static constexpr unsigned int kL1WBankBits = 4;
static constexpr std::size_t kL1WBankSize = (2 * 4096 * 128 / 8); //131072
static constexpr std::size_t kL1WBankNum = (1 << kL1WBankBits);
static constexpr unsigned int kL1EBankBits = 4;
static constexpr std::size_t kL1EBankSize = (4096 * 32 / 8); //16384
static constexpr std::size_t kL1EBankNum = kFpBankNum;

static constexpr unsigned int BANK_BITS = 4;
static constexpr unsigned int ALIGN_BITS = 1;
static constexpr unsigned int MAX_RANGE = 10000;
static constexpr unsigned int CMD_TIMES = 5;

DEFINE_int32(core_id, 0, "number of core_running");
DEFINE_string(func, "shuffle", "ds function shuffle or shuffle_batch or shuffle_coa or win2vec");
DEFINE_string(output, "./sd_cdnn_ds_cmd_seq.dat", "ds_cmd_file");
DEFINE_int32(seed, 0, "random seed");

#define ENABLE_L2_WB 1

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

void gen_ds_shuffle(std::ostream& out, int core_id) {
    int cmd_times = rand() % CMD_TIMES + 1;
    int module_id = core_id;    // 0:DS0-L2W; 1:DS1-L2D
    int ds_cfg_type = 0;
    int shuffle_shamt = 1;
    uint32_t src_addr = 0;
    uint32_t len = 0;
    uint32_t max_len = 0;
    int data_type_size = 1; //0-int8-1 1-int16-2 2-fp32-4
    int dst_addr_align = 1; //if ALIGN: int8-32B int16-32B fp32-4B
    uint32_t dst_addr = 0;
    uint32_t dst_offset = 0;
    uint32_t dst_sram_bank_size = 0;
    int addr_align_bits = 0; //unify address align bits

    out << "lock " << module_id << std::endl;
    for (int i = 0; i < cmd_times; i++) {
        int shuffle_datatype = rand() % 3; // 0:int8; 1:int16; 2:fp32[l2d/w -> l1e]
        if (shuffle_datatype == 2) {
#ifdef ENABLE_L2_WB
            shuffle_shamt = (rand() % 2 + 1) * 4; // float path must be L2D/L2W -> L1E or L2 Write Back
#else
            shuffle_shamt = 4; // float path must be L2D/L2W -> L1E
#endif
        } else {
#ifdef ENABLE_L2_WB
            shuffle_shamt = rand() % 3 + 1; // 1:to l1d; 2:to l1w 3:l2 wb
            shuffle_shamt = (shuffle_shamt == 3) ? 8 : shuffle_shamt; // l2 write back
#else
            shuffle_shamt = rand() % 2 + 1; // 1:to l1d; 2:to l1w
#endif
        }
        out << "ds_cfg " << "rs=0" << " rt=0"
            << " rd=" << shuffle_datatype << " shamt=" << ds_cfg_type << std::endl;
        switch (shuffle_datatype) {
            case 0:
                // int8
                //data_type_size = 1;
                data_type_size = 2; // int8 should use int16 space because of RTL restriction
                break;
            case 1:
                // int16
                data_type_size = 2;
                break;
            case 2:
                // fp32
                data_type_size = 4;
                dst_addr_align = 4;
                break;
            default:
                break;
        }
        int dst_bank_id = rand() % 16; // dst bank id: 0 ~ 15
        switch (shuffle_shamt) {
            case 1:
                // to l1d
                dst_sram_bank_size = kL1DBankSize / 2;
                addr_align_bits = 1;
                break;
            case 2:
                // to l1w
                dst_sram_bank_size = kL1WBankSize / 2;
                addr_align_bits = 1;
                break;
            case 4:
                // to l1e
                dst_sram_bank_size = kL1EBankSize / 2;
                addr_align_bits = 2;
                break;
            case 8:
                // to l2
                dst_sram_bank_size = kL2WDBankSize / 2;
                addr_align_bits = 1;
                break;
            default:
                break;
        }

        if (shuffle_shamt == 8) { // Write Back
            out << "xfence" << std::endl;
            src_addr = rand() % (kL2WDBankSize / 4 - data_type_size);
            max_len = std::min((unsigned int)((kL2WDBankSize / 2 - src_addr) / data_type_size), (unsigned int)(dst_sram_bank_size / data_type_size - 1));
            assert(max_len != 0);
            len = rand() % max_len + 1;
            src_addr = (src_addr + len * data_type_size) > kL2WDBankSize / 2 ? 0 : src_addr;

            assert(len * data_type_size != dst_sram_bank_size);
            if (((dst_sram_bank_size - len * data_type_size) / dst_addr_align) == 0) {
                dst_offset = kL2WDBankSize / 2 + 1;
            } else {
                dst_offset = kL2WDBankSize / 2 + (rand() % ((dst_sram_bank_size - len * data_type_size) / dst_addr_align)) * dst_addr_align;
            }
            dst_offset = ((dst_offset + len * data_type_size) > dst_sram_bank_size * 2) ? 0 : dst_offset;
            dst_addr = dst_offset;
        } else {
            // rand base should not be 0 !!!
            assert(data_type_size != kL2WDBankSize);
            src_addr = rand() % (kL2WDBankSize - data_type_size);
            max_len = std::min((unsigned int)((kL2WDBankSize - src_addr) / data_type_size), (unsigned int)(dst_sram_bank_size / data_type_size - 1));
            assert(max_len != 0);
            len = rand() % max_len + 1;
            src_addr = (src_addr + len * data_type_size) > kL2WDBankSize ? 0 : src_addr;

            len = std::min(dst_sram_bank_size / data_type_size - 1, len);
            len = (len * data_type_size >= dst_sram_bank_size) ? 1 : len;
            assert(len * data_type_size < dst_sram_bank_size);
            if (((dst_sram_bank_size - len * data_type_size) / dst_addr_align) == 0) {
                dst_offset = core_id * dst_sram_bank_size;
            } else {
                dst_offset = core_id * dst_sram_bank_size + (rand() % ((dst_sram_bank_size - len * data_type_size) / dst_addr_align)) * dst_addr_align;
            }
            dst_offset = dst_offset > 0 ? dst_offset : 0;
            dst_offset = ((dst_offset + len * data_type_size) > (core_id + 1) * dst_sram_bank_size) ? core_id * dst_sram_bank_size : dst_offset;
            dst_addr = unified_hw_addr(dst_bank_id, dst_offset, 4, addr_align_bits);
        }
        out << "shuffle " << "rs=" << src_addr << " rt=" << dst_addr
            << " rd=" << len << " shamt=" << shuffle_shamt << std::endl;
        if (shuffle_shamt == 8) { // Write Back
            out << "xfence" << std::endl;
        }

#ifdef DEBUG
        std::cout << "shuffle_shamt = " << shuffle_shamt << std::endl;
        std::cout << "dst_bank_id = " << dst_bank_id << std::endl;
        std::cout << "src_addr = " << src_addr << std::endl;
        std::cout << "len = " << len << std::endl;
        std::cout << "dst_addr_align= " << dst_addr_align << std::endl;
        std::cout << "shuffle_datatype = " << shuffle_datatype << std::endl;
        std::cout << "dst_offset = " << dst_offset << std::endl;
        std::cout << "dst_addr = " << dst_addr << std::endl;
#endif
    }
    out << "xfence" << std::endl;
    out << "unlock " << module_id << std::endl;
}

void gen_ds_shuffle_batch(std::ostream& out, int core_id) {
    int cmd_times = rand() % CMD_TIMES + 1;
    uint32_t module_id = core_id; // 0:DS0-L2W; 1:DS1-L2D
    uint32_t shuffle_batch_shamt = 1;
    uint32_t src_addr = 0;
    uint32_t src_end = 0;
    uint32_t len = 0;
    uint32_t dst_bank = 0;
    uint32_t dst_offset = 0;
    uint32_t dst_addr = 0;
    uint32_t max_shuffle_batch_blockx = kL2WDBankSize / (4 * 16 * 2); // should 16 parts float 4 Byte
    uint32_t data_type_size = 0;
    int dst_addr_align = 1; //if ALIGN: int8-32B int16-32B fp32-4B
    uint32_t addr_align_bits= 0;
    uint32_t dst_sram_bank_size = 0;
    uint32_t l2_wb_stride = 1;

    out << "lock " << module_id << std::endl;
    for (int i = 0; i < cmd_times; i++) {
        uint32_t shuffle_batch_datatype = rand() % 3; // 0:int8; 1:int16; 2:fp32[l2d/w -> l1e]
        if (shuffle_batch_datatype == 2) {
#ifdef ENABLE_L2_WB
            shuffle_batch_shamt = (rand() % 2 + 1) * 4; // 4:to l1e 8:l2 write back
#else
            shuffle_batch_shamt = 4; // 4:to l1e
#endif
        } else {
#ifdef ENABLE_L2_WB
            shuffle_batch_shamt = rand() % 3 + 1; // 1:to l1d; 2:to l1w;
            shuffle_batch_shamt = (shuffle_batch_shamt == 3) ? 8 : shuffle_batch_shamt; //8:l2 write back
#else
            shuffle_batch_shamt = rand() % 2 + 1; // 1:to l1d; 2:to l1w;
#endif
        }

        uint32_t ds_cfg_shamt = 0;
        out << "ds_cfg " << "rs=0" << " rt=0"
            << " rd=" << shuffle_batch_datatype << " shamt=" << ds_cfg_shamt << std::endl;
        ds_cfg_shamt = 1;
        uint32_t shuffle_batch_blockx = rand() % max_shuffle_batch_blockx + 1;
        out << "ds_cfg " << "rs=0" << " rt=0"
            << " rd=" << shuffle_batch_blockx << " shamt=" << ds_cfg_shamt << std::endl;
        switch (shuffle_batch_datatype) {
            case 0:
                // int8
                //data_type_size = 1;
                data_type_size = 2; // int8 should use int16 space because of RTL restriction
                break;
            case 1:
                // int16
                data_type_size = 2;
                break;
            case 2:
                // fp32
                data_type_size = 4;
                dst_addr_align = 4;
                break;
            default:
                break;
        }
        ds_cfg_shamt = 11;
        uint32_t shuffle_batch_output_bank_num = rand() % 16 + 1; //[1, 16]
        out << "ds_cfg " << "rs=0" << " rt=0"
            << " rd=" << shuffle_batch_output_bank_num << " shamt=" << ds_cfg_shamt << std::endl;

        switch (shuffle_batch_shamt) {
            case 1:
                // to l1d
                dst_sram_bank_size = kL1DBankSize / 2;
                addr_align_bits = 1;
                break;
            case 2:
                // to l1w
                dst_sram_bank_size = kL1WBankSize / 2;
                addr_align_bits = 1;
                break;
            case 4:
                // to l1e
                dst_sram_bank_size = kL1EBankSize / 2;
                addr_align_bits = 2;
                break;
            case 8:
                // to l2
                dst_sram_bank_size = kL2WDBankSize / 2;
                addr_align_bits = 1;
                break;
            default:
                break;
        }
        if (shuffle_batch_shamt == 8) {
            src_addr = rand() % ((kL2WDBankSize / 8) / shuffle_batch_output_bank_num);
            if ((shuffle_batch_output_bank_num * shuffle_batch_blockx * data_type_size) > (kL2WDBankSize / 2 - src_addr)){
                len = 1;
            } else {
                len = rand() % ((kL2WDBankSize / 2 - src_addr) / (shuffle_batch_output_bank_num * shuffle_batch_blockx * data_type_size)) + 1;
            }
            src_addr = (src_addr + shuffle_batch_output_bank_num * shuffle_batch_blockx * data_type_size) > (kL2WDBankSize / 2) ? 0 : src_addr;
            l2_wb_stride = shuffle_batch_blockx + rand() % 10;
            l2_wb_stride = std::max(len, l2_wb_stride);
            ds_cfg_shamt = 12;
            out << "ds_cfg " << "rs=0" << " rt=0"
                << " rd=" << l2_wb_stride << " shamt=" << ds_cfg_shamt << std::endl;
            src_end = src_addr + (shuffle_batch_output_bank_num - 1) * shuffle_batch_blockx * data_type_size + len * data_type_size;
            if (shuffle_batch_blockx < len) {
                src_end += len * data_type_size;
            }
            if ((kL2WDBankSize - src_end - shuffle_batch_output_bank_num * l2_wb_stride * data_type_size) != 0) {
                dst_offset = src_end + rand() % (kL2WDBankSize - src_end - shuffle_batch_output_bank_num * l2_wb_stride * data_type_size);
            } else {
                dst_offset = src_end;
            }
            dst_addr = dst_offset;
        } else {
            src_addr = rand() % (kL2WDBankSize / shuffle_batch_output_bank_num);
            if ((shuffle_batch_output_bank_num * shuffle_batch_blockx * data_type_size) > (kL2WDBankSize - src_addr)) {
                len = 1;
            } else {
                len = rand() % ((kL2WDBankSize - src_addr) / (shuffle_batch_output_bank_num * shuffle_batch_blockx * data_type_size)) + 1;
            }
            assert ((src_addr + len * data_type_size * shuffle_batch_blockx) <= kL2WDBankSize);
            src_addr = (src_addr + shuffle_batch_output_bank_num * shuffle_batch_blockx * data_type_size) > kL2WDBankSize ? 0 : src_addr;
            len = std::min(dst_sram_bank_size / data_type_size - 1, len);
            len = (len * data_type_size >= dst_sram_bank_size) ? 1 : len;
            dst_offset = core_id * dst_sram_bank_size + (rand() % ((dst_sram_bank_size - len * data_type_size) / dst_addr_align)) * dst_addr_align;
            dst_offset = ((dst_offset + len * data_type_size) > ((core_id + 1) * dst_sram_bank_size)) ? core_id * dst_sram_bank_size : dst_offset;
            uint32_t dst_bank_id = 0; // do not care dst_bank_id in shuffle batch cmd
            dst_addr = unified_hw_addr(dst_bank_id, dst_offset, 4, addr_align_bits);
        }
        out << "shuffle_batch " << "rs=" << src_addr << " rt=" << dst_addr
            << " rd=" << len << " shamt=" << shuffle_batch_shamt << std::endl;
        if (shuffle_batch_shamt == 8) { // Write Back
            out << "xfence" << std::endl;
        }

#ifdef DEBUG
        std::cout << "shuffle_batch_datatype is " << shuffle_batch_datatype << std::endl;
        std::cout << "shuffle_batch_shamt is " << shuffle_batch_shamt << std::endl;
        std::cout << "len is " << len << std::endl;
        std::cout << "shuffle_batch_output_bank_num is " << shuffle_batch_output_bank_num << std::endl;
        std::cout << "shuffle_batch_blockx is " << shuffle_batch_blockx << std::endl;
        std::cout << "dst_addr_align is " << dst_addr_align << std::endl;
        std::cout << "src_addr is " << src_addr << std::endl;
        std::cout << "dst_offset is " << dst_offset << std::endl;
        std::cout << "dst_addr is " << dst_addr << std::endl;
        std::cout << "kL2WDBankSize is " << kL2WDBankSize << std::endl;
#endif
    }
    out << "xfence" << std::endl;
    out << "unlock " << module_id << std::endl;
}

void gen_ds_shuffle_coa(std::ostream& out, int core_id) {
    int cmd_times = rand() % CMD_TIMES + 1;
    uint32_t module_id = core_id; // 0:DS0-L2W; 1:DS1-L2D
    uint32_t shuffle_coa_shamt = 1;
    uint32_t src_end = 0;
    uint32_t len = 0;
    uint32_t dst_bank = 0;
    uint32_t dst_offset = 0;
    uint32_t dst_addr = 0;
    uint32_t max_shuffle_coa_blockx = 1024;
    uint32_t data_type_size = 0;
    int dst_addr_align = 1; //if ALIGN: int8-32B int16-32B fp32-4B
    uint32_t addr_align_bits= 0;
    uint32_t dst_sram_bank_size = 0;
    uint32_t l2_wb_stride = 1;

    out << "lock " << module_id << std::endl;
    for (int i = 0; i < cmd_times; i++) {
        uint32_t shuffle_coa_datatype = rand() % 3; // 0:int8; 1:int16; 2:fp32[l2d/w -> l1e]
        if (shuffle_coa_datatype == 2) {
#ifdef ENABLE_L2_WB
            shuffle_coa_shamt = 4 << (rand() % 2); // 4:float path must be L2D/L2W -> L1E 8:L2 write back
#else
            shuffle_coa_shamt = 4;
#endif
        } else {
#ifdef ENABLE_L2_WB
            shuffle_coa_shamt = rand() % 3 + 1; // 1:to l1d; 2:to l1w;
            shuffle_coa_shamt = (shuffle_coa_shamt == 3) ? 8 : shuffle_coa_shamt; // 8:L2 write back
#else
            shuffle_coa_shamt = rand() % 2 + 1;
#endif
        }

        int ds_cfg_shamt = 0;
        out << "ds_cfg " << "rs=0" << " rt=0"
            << " rd=" << shuffle_coa_datatype << " shamt=" << ds_cfg_shamt << std::endl;
        ds_cfg_shamt = 1;
        uint32_t shuffle_coa_blockx = rand() % max_shuffle_coa_blockx + 1;
        shuffle_coa_blockx = std::max(16, (int)shuffle_coa_blockx);
        out << "ds_cfg " << "rs=0" << " rt=0"
            << " rd=" << shuffle_coa_blockx << " shamt=" << ds_cfg_shamt << std::endl;
        uint32_t src_addr = rand() % (kL2WDBankSize / 8);
        switch (shuffle_coa_datatype) {
            case 0:
                // int8
                //data_type_size = 1;
                data_type_size = 2; // int8 should use int16 space because of RTL restriction
                break;
            case 1:
                // int16
                data_type_size = 2;
                break;
            case 2:
                // fp32
                data_type_size = 4;
                dst_addr_align = 4;
                break;
            default:
                break;
        }

        ds_cfg_shamt = 11;
        uint32_t shuffle_coa_output_bank_num = rand() % 16 + 1; //[1, 16]
        out << "ds_cfg " << "rs=0" << " rt=0"
            << " rd=" << shuffle_coa_output_bank_num << " shamt=" << ds_cfg_shamt << std::endl;
        switch (shuffle_coa_shamt) {
            case 1:
                // to l1d
                dst_sram_bank_size = kL1DBankSize / 2;
                addr_align_bits = 1;
                break;
            case 2:
                // to l1w
                dst_sram_bank_size = kL1WBankSize / 2;
                addr_align_bits = 1;
                break;
            case 4:
                // to l1e
                dst_sram_bank_size = kL1EBankSize / 2;
                addr_align_bits = 2;
                break;
            case 8:
                // to l2d/w
                dst_sram_bank_size = kL2WDBankSize;
                addr_align_bits = 1;
                break;
            default:
                break;
        }
        if (shuffle_coa_shamt == 8) {
            if (((kL2WDBankSize / 4) / (shuffle_coa_blockx * data_type_size)) > 0) {
                len = rand() % ((kL2WDBankSize / 4) / (shuffle_coa_blockx * data_type_size)) + 1;
            } else {
                len = 1;
            }
            len = std::min(len, dst_sram_bank_size / 4 / data_type_size - 1);
            assert ((src_addr + len * data_type_size * shuffle_coa_blockx) <= kL2WDBankSize);
            src_addr = (src_addr + len * shuffle_coa_blockx * data_type_size) > kL2WDBankSize ? 0 : src_addr;

            ds_cfg_shamt = 12;
            l2_wb_stride = (len + (rand() % 1024 + 1));
            out << "ds_cfg " << "rs=0" << " rt=0"
                << " rd=" << l2_wb_stride << " shamt=" << ds_cfg_shamt << std::endl;

            src_end = src_addr + len * shuffle_coa_blockx * data_type_size;
            if ((dst_sram_bank_size - src_end - l2_wb_stride * shuffle_coa_output_bank_num * data_type_size) > 0) {
                dst_addr = src_end + rand() % (dst_sram_bank_size - src_end - l2_wb_stride * shuffle_coa_output_bank_num * data_type_size);
            } else {
                dst_addr = src_end;
            }
        } else {
            if (((kL2WDBankSize - src_addr) / (shuffle_coa_blockx * data_type_size)) != 0) {
                len = rand() % ((kL2WDBankSize - src_addr) / (shuffle_coa_blockx * data_type_size)) + 1;
            } else {
                len = 1;
            }
            len = std::min(len, dst_sram_bank_size / data_type_size - 1);
            assert ((src_addr + len * data_type_size * shuffle_coa_blockx) <= kL2WDBankSize);
            src_addr = (src_addr + len * shuffle_coa_blockx * data_type_size) > kL2WDBankSize ? 0 : src_addr;
            if (((dst_sram_bank_size - len * data_type_size) / dst_addr_align) != 0) {
                dst_offset = core_id * dst_sram_bank_size + (rand() % ((dst_sram_bank_size - len * data_type_size) / dst_addr_align)) * dst_addr_align;
            } else {
                dst_offset = core_id * dst_sram_bank_size;
            }
            dst_offset = ((dst_offset + len * data_type_size) > ((core_id + 1) * dst_sram_bank_size)) ? (core_id * dst_sram_bank_size) : dst_offset;
            assert(dst_offset + len * data_type_size < dst_sram_bank_size * (core_id + 1));
            uint32_t dst_bank_id = 0; // do not care dst_bank_id in shuffle coa cmd
            dst_addr = unified_hw_addr(dst_bank_id, dst_offset, 4, addr_align_bits);
        }
        out << "shuffle_coa " << "rs=" << src_addr << " rt=" << dst_addr
            << " rd=" << len << " shamt=" << shuffle_coa_shamt << std::endl;
        if (shuffle_coa_shamt == 8) { // Write Back
            out << "xfence" << std::endl;
        }
#ifdef DEBUG
        std::cout << "len is " << len << std::endl;
        std::cout << "dst_addr_align is " << dst_addr_align << std::endl;
        std::cout << "data_type_size is " << data_type_size << std::endl;
        std::cout << "src_addr is " << src_addr << std::endl;
        std::cout << "dst_offset is " << dst_offset << std::endl;
        std::cout << "dst_sram_bank_size is " << dst_sram_bank_size << std::endl;
        std::cout << "dst_addr is " << dst_addr << std::endl;
#endif
    }
    out << "xfence" << std::endl;
    out << "unlock " << module_id << std::endl;
}

void gen_ds_win2vec(std::ostream& out, int core_id) {
    int cmd_times = rand() % CMD_TIMES + 1;

    int module_id = core_id; //win2vec only support L2D->L1D int16 int8
    int filter_size = 1;
    int conv_win_x = filter_size;
    int conv_win_y = filter_size;
    int conv_blk_x = conv_win_x;
    int conv_blk_y = conv_win_y;
    int conv_block_distance = conv_blk_x * conv_blk_x;
    int conv_dilation = 1;
    int pad_w_size = 0;
    int conv_pad_left = pad_w_size;
    int conv_pad_right = pad_w_size;
    int pad_h_size = 0;
    int conv_pad_up = pad_h_size;
    int conv_pad_down = pad_h_size;
    int dst_offset = 0;
    uint32_t src_addr = 0;
    uint32_t dst_addr = 0;
    int win2vec_shamt = 0;
    int dst_addr_align = 1; //if ALIGN: int8-32B int16-32B fp32-4B
    int dst_sram_bank_size = 0;
    int addr_align_bits = 1;
    int l2_wb_stride = 0;

    out << "lock " << module_id << std::endl;

    for (int i = 0; i < cmd_times; i++) {
        int ds_cfg_shamt = 2;
        int conv_datatype = rand() % 2; // 0-int8 1-int16
        out << "ds_cfg " << "rs=0" << " rt=0"
            << " rd=" << conv_datatype << " shamt=" << ds_cfg_shamt << std::endl;


        ds_cfg_shamt = 7;
        out << "ds_cfg " << "rs=0" << " rt=0"
            << " rd=" << conv_dilation << " shamt=" << ds_cfg_shamt << std::endl;

        ds_cfg_shamt = 8;
        int conv_stride = rand() % 15 + 1;
        out << "ds_cfg " << "rs=0" << " rt=0"
            << " rd=" << conv_stride << " shamt=" << ds_cfg_shamt << std::endl;

        ds_cfg_shamt = 10;
        conv_pad_up = rand() % 64; // 0 ~ 63
        conv_pad_up = std::min(62, conv_pad_up);
        conv_pad_down = rand() % (63 - conv_pad_up);
        assert((conv_pad_up + conv_pad_down) < 63);
        out << "ds_cfg " << "rs=" << conv_pad_down << " rt=0"
            << " rd=" << conv_pad_up << " shamt=" << ds_cfg_shamt << std::endl;

        ds_cfg_shamt = 4;
        //filter_size = rand() % 8 + 1;
        if (conv_datatype == 0) {
            // int8
            conv_win_x = rand() % 64 + 1;
            conv_win_y = conv_pad_up + conv_pad_down + rand() % (63 - conv_pad_up - conv_pad_down) + 1; // 1 ~ 63
        } else if (conv_datatype == 1) {
            // int16
            conv_win_x = rand() % 32 + 1;
            conv_win_y = conv_pad_up + conv_pad_down + rand() % (63 - conv_pad_up - conv_pad_down) + 1; // 1 ~ 63
        }
        conv_win_y = std::min(conv_win_y, 63);
        conv_win_y = std::max(conv_win_y, conv_pad_up + conv_pad_down + 1);
        out << "ds_cfg " << "rs=" << conv_win_y << " rt=0"
            << " rd=" << conv_win_x << " shamt=" << ds_cfg_shamt << std::endl;

        ds_cfg_shamt = 9;
        //pad_w_size = rand() % 16;
        if (conv_datatype == 0) {
            // int8
            conv_pad_left = rand() % 16;
            conv_pad_right = rand() % 16;
        } else if (conv_datatype == 1) {
            // int16
            conv_pad_left = rand() % 8;
            conv_pad_right = rand() % 8;
        }
        if (conv_pad_left + conv_pad_right > conv_win_x) {
            conv_pad_left = conv_win_x / 2;
            conv_pad_right = conv_win_x / 2;
        }

        out << "ds_cfg " << "rs=" << conv_pad_right << " rt=0"
            << " rd=" << conv_pad_left << " shamt=" << ds_cfg_shamt << std::endl;

        ds_cfg_shamt = 5;
        conv_blk_x = rand() % 2 + conv_win_x;
        if (conv_datatype == 1) {
            // int16
            conv_blk_x = roundup(conv_blk_x, 2);
        }
        conv_blk_y = rand() % 2 + conv_win_y;
        out << "ds_cfg " << "rs=" << conv_blk_y << " rt=0"
            << " rd=" << conv_blk_x << " shamt=" << ds_cfg_shamt << std::endl;

        ds_cfg_shamt = 6;
        conv_block_distance = conv_blk_x * conv_blk_y;
        int data_type_size = (conv_datatype == 0) ? 1 : 2;
        out << "ds_cfg " << "rs=0" << " rt=0"
            << " rd=" << conv_block_distance * data_type_size << " shamt=" << ds_cfg_shamt << std::endl;

        switch (conv_datatype) {
            case 0:
                // int8
                //data_type_size = 1;
                data_type_size = 2; // int8 should use int16 space because of RTL restriction
                break;
            case 1:
                // int16
                data_type_size = 2;
                break;
            default:
                break;
        }

#ifdef ENABLE_L2_WB
        win2vec_shamt = rand() % 3 + 1; // 1:to l1d; 2:to l1w
        win2vec_shamt = (win2vec_shamt == 3) ? 8 : win2vec_shamt;
#else
        win2vec_shamt = rand() % 2 + 1; // 1:to l1d; 2:to l1w
#endif
        switch (win2vec_shamt) {
            case 1:
                // to l1d
                dst_sram_bank_size = kL1DBankSize / 2;
                addr_align_bits = 1;
                break;
            case 2:
                // to l1w
                dst_sram_bank_size = kL1WBankSize / 2;
                addr_align_bits = 1;
                break;
            case 8:
                // to l2
                dst_sram_bank_size = kL2WDBankSize / 2;
                addr_align_bits = 1;
                break;
            default:
                break;
        }

        int conv_channel = rand() % 65535 + 1;
        if ((conv_channel * conv_block_distance * data_type_size) > kL2WDBankSize) {
            conv_channel = kL2WDBankSize / (conv_block_distance * data_type_size) - 1;
            conv_channel = std::max(1, conv_channel);
        }
        if ((data_type_size * conv_win_x * conv_win_y * conv_channel) > dst_sram_bank_size) {
            conv_channel = dst_sram_bank_size / (data_type_size * conv_win_x * conv_win_y) - 1;
            conv_channel = std::max(1, conv_channel);
        }
        if (win2vec_shamt == 8) {
            if ((conv_channel * conv_block_distance * data_type_size) > (kL2WDBankSize / 2)) {
                conv_channel = (kL2WDBankSize / 2) / (conv_block_distance * data_type_size) - 1;
                conv_channel = std::max(1, conv_channel);
            }
            if (conv_channel * conv_win_x * conv_win_y * data_type_size * 16 > (kL2WDBankSize / 2)) {
                conv_channel = (kL2WDBankSize / 2) / (conv_win_x * conv_win_y * data_type_size * 16) - 1;
                conv_channel = std::max(1, conv_channel);
            }
        }

        // win2vec_len int16:1~32767 int8:1~65535
        // output_w = (win2vec_len + pad_left + pad_right - win_w) / stride + 1  [1, 16]
        // conv_win_x - conv_pad_left - conv_pad_right <= win2vec_len <= 15 * conv_stride + conv_win_x - conv_pad_left - conv_pad_right
        int win2vec_len = rand() % (conv_win_x + conv_stride * 15 - conv_pad_left - conv_pad_right) + 1;
        win2vec_len = (win2vec_len < (conv_win_x - conv_pad_left - conv_pad_right)) ? (conv_win_x - conv_pad_left - conv_pad_right) : win2vec_len;

        int nwin = (conv_pad_left + conv_pad_right + win2vec_len - conv_win_x) / conv_stride + 1;

        assert(nwin <= 16);
        assert(nwin >= 1);

        int src_data_size = (conv_channel - 1) * conv_block_distance * data_type_size
            + (conv_win_y - 1) * conv_blk_x * data_type_size + win2vec_len * data_type_size;

        if (win2vec_shamt == 8) {
            if ((int)(kL2WDBankSize / 2) < (src_data_size + 16)) {
                src_addr = 0;
                conv_channel = 1;
            } else {
                src_addr = rand() % ((kL2WDBankSize / 2 - src_data_size) / 16);
            }
            ds_cfg_shamt = 12;
            l2_wb_stride = conv_channel * conv_win_x * conv_win_y;
            out << "ds_cfg " << "rs=0" << " rt=0"
                << " rd=" << l2_wb_stride << " shamt=" << ds_cfg_shamt << std::endl;
            dst_offset = src_addr + (conv_channel - 1) * conv_block_distance * data_type_size
                + (conv_win_y - 1) * conv_blk_x * data_type_size + win2vec_len * data_type_size;

            assert (dst_offset + nwin * l2_wb_stride * data_type_size < kL2WDBankSize);
            dst_addr = dst_offset;
        } else {
            if ((int)kL2WDBankSize < (src_data_size + 16)) {
                src_addr = 0;
                conv_channel = 1;
            } else {
                src_addr = rand() % ((kL2WDBankSize - src_data_size) / 16);
            }

            dst_offset = core_id * dst_sram_bank_size + roundup((rand() % (dst_sram_bank_size - dst_addr_align - data_type_size * conv_win_x * conv_win_y * conv_channel)) ,dst_addr_align);
            dst_offset = (dst_offset + dst_addr_align + data_type_size * conv_win_x * conv_win_y * conv_channel) > ((core_id + 1) * dst_sram_bank_size) ? core_id * dst_sram_bank_size : dst_offset;
            int dst_bank_id = 0; // do not care dst_bank_id in win2vec cmd
            dst_addr = unified_hw_addr(dst_bank_id, dst_offset, 4, addr_align_bits);
        }

        ds_cfg_shamt = 3;
        out << "ds_cfg " << "rs=0" << " rt=0"
            << " rd=" << conv_channel << " shamt=" << ds_cfg_shamt << std::endl;

#ifdef DEBUG
        std::cout << std::endl;
        std::cout << "nwin is  " << nwin << std::endl;
        std::cout << "conv_datatype is  " << conv_datatype << std::endl;
        std::cout << "conv_channel is  " << conv_channel << std::endl;
        std::cout << "conv_win_x is  " << conv_win_x << std::endl;
        std::cout << "conv_win_y is  " << conv_win_y << std::endl;
        std::cout << "conv_blk_x is  " << conv_blk_x << std::endl;
        std::cout << "conv_blk_y is  " << conv_blk_y << std::endl;
        std::cout << "conv_dilation is  " << conv_dilation << std::endl;
        std::cout << "conv_stride is  " << conv_stride << std::endl;
        std::cout << "conv_pad_left is  " << conv_pad_left << std::endl;
        std::cout << "conv_pad_right is  " << conv_pad_right << std::endl;
        std::cout << "conv_pad_up is  " << conv_pad_up << std::endl;
        std::cout << "conv_pad_down is  " << conv_pad_down << std::endl;
        std::cout << "win2vec_len is  " << win2vec_len << std::endl;
        std::cout << "src_addr is " << src_addr << std::endl;
        std::cout << "dst_offset is " << dst_offset << std::endl;
        std::cout << "dst_addr is " << dst_addr << std::endl;
        std::cout << "l2_wb_stride is  " << l2_wb_stride << std::endl;
        std::cout << "conv_block_distance is  " << conv_block_distance << std::endl;
        std::cout << "conv_channel is  " << conv_channel << std::endl;
        std::cout << "data_type_size is  " << data_type_size << std::endl;
        std::cout << "dst_addr_align is  " << dst_addr_align << std::endl;
        std::cout << std::endl;
#endif

        out << "win2vec " << "rs=" << win2vec_len << " rt=" << dst_addr
            << " rd=" << src_addr << " shamt=" << win2vec_shamt << std::endl;
        if (win2vec_shamt == 8) {
            out << "xfence" << std::endl;
        }
    }

    out << "xfence" << std::endl;
    out << "unlock " << module_id << std::endl;
}

int main(int argc, char** argv) {
    google::ParseCommandLineFlags(&argc, &argv, true);

    srand(FLAGS_seed);

    std::ofstream cmd_seq;
    cmd_seq.open(FLAGS_output);

    if (FLAGS_func == "shuffle") {
        gen_ds_shuffle(cmd_seq, FLAGS_core_id);
    }
    else if (FLAGS_func == "shuffle_batch") {
        gen_ds_shuffle_batch(cmd_seq, FLAGS_core_id);
    }
    else if (FLAGS_func == "shuffle_coa") {
        gen_ds_shuffle_coa(cmd_seq, FLAGS_core_id);
    }
    else if (FLAGS_func == "win2vec") {
        gen_ds_win2vec(cmd_seq, FLAGS_core_id);
    }

    //cmd_seq << "xfence" << std::endl;
    cmd_seq.close();

    return 0;
}
