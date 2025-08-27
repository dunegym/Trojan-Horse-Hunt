#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <gflags/gflags.h>
#include <assert.h>
#include <math.h>
#include<iomanip>

//#define DEBUG 1

static constexpr std::size_t kFpBankDepth = 2048;
static constexpr std::size_t kFpBankNum = 16;

static constexpr std::size_t kL2WDBankSize = (2048 * 512 / 8); // L2D/L2W
static constexpr std::size_t kL2DBankSize = (2048 * 512 / 8);
static constexpr std::size_t kL2DBankNum = 1;
static constexpr std::size_t kL2WBankSize = (2048 * 512 / 8);
static constexpr std::size_t kL2WBankNum = 1;
static constexpr std::size_t kL2RBankSize = (kFpBankDepth * 32 / 8);
static constexpr std::size_t kL2RBankNum = kFpBankNum;
static constexpr std::size_t kL2RSize = kL2RBankSize * kL2RBankNum;
static constexpr unsigned int BANK_BITS = 4;
static constexpr unsigned int ALIGN_BITS = 1;
//(2048 * 64) equal to kL2WDBankSize
static constexpr unsigned int TestSize = 2048;
static constexpr unsigned int PARTITION = 64;
static constexpr unsigned int MaxTestSize = 64 * 1024; //64KByte
static constexpr unsigned int MaxOffsetSize = 64; //64Byte

//#define kGlobalMemorySize (4ull * 1024 * 1024 * 1024)
#define kGlobalMemorySize (4ull * 1024 * 1024)
#define CMD_TIMES 30

DEFINE_string(direction, "readhbm", "dma direction readhbm or readl2r or readl2wd");
DEFINE_string(output, "./sd_cdnn_dma_cmd_seq.dat", "dma_cmd_file");
DEFINE_int32(seed, 0, "random seed");

int roundup(int n, int k) {
    if (k != 0) {
        return ((n + k - 1) / k) * k;
    } else {
        return n;
    }
}

uint32_t unified_hw_addr(uint32_t bank, uint32_t offset, int bank_bits=BANK_BITS, int align_bits=ALIGN_BITS) {
    uint32_t off_lo = offset & ((1 << align_bits) - 1);
    uint32_t off_hi = (offset >> align_bits) << (bank_bits + align_bits);
    uint32_t bank_f = bank << align_bits;
    uint32_t addr = off_hi | bank_f | off_lo;
    return addr;
}

void gen_dma_readl2r(std::ostream& out) {
    int cmd_times= rand() % CMD_TIMES + 1;
    int module_id = 7; // 7:DMAO
    int source_id = 0;
    int dst_id = 0;
    int src_type = 0;
    int dst_type = 0;
    uint32_t hbm_src_high_addr = 0;
    uint32_t hbm_dst_high_addr = 0;
    uint32_t rd = 0;
    uint32_t rs = 0;
    uint32_t rt = 0;
    float max = 0;
    uint32_t dst_2d_offset = 0;
    uint32_t dst_addr_offset = 0;
    uint32_t src_2d_offset = 0;
    uint32_t loop_2d = 0;
    int dma_2d_enable = 0;
    int findmax_enable = 0;
    uint32_t i_num = 0;

    uint32_t src_offset_max = 0;
    uint32_t src_offset = 0;
    uint32_t src_bank = 0;
    uint32_t src_addr = 0;
    uint32_t dst_addr = 0;
    int src_data_size = 0;
    int dst_data_size = 0;
    uint32_t dst_size = 0;

    out << "lock " << module_id << std::endl;

    // src_type:
    // 3'b000:FLOAT32
    // 3'b001:FLOAT16
    // 3'b010:INT16
    // 3'b011:BFLOAT16
    // 3'b100:INT8
    // 3'b101:INT31
    // dst_type:
    // 3'b000:FLOAT32
    // 3'b001:FLOAT16
    // 3'b010:INT16
    // 3'b011:BFLOAT16
    // 3'b100:INT8
    // 3'b101:INT31
    src_type = 0; //L2R only support fp32
    switch (src_type) {
        case 0: //fp32
            src_data_size = 4;
            dst_type = rand() % 6; //0:fp32 1:fp16 2:int16 3:bfp16 4:int8 5:int31
            break;
        //not support now
        /*
        case 1: //fp16
        case 3: //bfp16
            src_data_size = 2;
            dst_type = rand() % 4 + 1; //1:fp16 2:int16 3:bfp16 4:int8
            break;
        case 2: //int16
            src_data_size = 2;
            dst_type = 2;
            break;
        case 4: //int8
            src_data_size = 1;
            dst_type = 4;
            break;
        */
        default:
            break;
    }
    switch (dst_type) {
        case 0:
            dst_data_size = 4;
            break;
        case 1:
        case 2:
        case 3:
            dst_data_size = 2;
            break;
        case 4:
            dst_data_size = 1;
            break;
        case 5:
            dst_data_size = 4;
            break;
        default:
            break;
    }

    source_id = 1; //source is L2R
    dst_id = rand() % 4; //0:HBM 1:RESERVED 2:L2W 3:L2D
    dst_id = (dst_id == 1) ? 0 : dst_id;
    switch (dst_id) {
        case 0:
            dst_size = (uint32_t)kGlobalMemorySize;
            break;
        case 2:
        case 3:
            dst_size = kL2WDBankSize;
            break;
        default:
            break;
    }
    assert(source_id != dst_id);

    hbm_src_high_addr = 0;
    hbm_dst_high_addr = 0;
    rd = hbm_dst_high_addr | (dst_type << 4) | (dst_id << 8);
    rs = hbm_src_high_addr | (src_type << 4) | (source_id << 8);
    max = rand() / (double)(RAND_MAX / 10000.0);
    uint32_t* max_ptr = reinterpret_cast<uint32_t*>(&max);
    rt = *max_ptr;
    out << "l2_mov_cfg " << "rs=" << rs << " rt=" << rt
        << " rd=" << rd << " shamt=0" << std::endl;

    findmax_enable = rand() % 3; //0:no findmax 1:sub findmax 2:all findmax
    dma_2d_enable = rand() % 2; //0:disable 1:enable
    i_num = (dma_2d_enable << 2) | findmax_enable;

    for (int i = 0; i < cmd_times; i++) {
        //kL2RSize == PARTITION * TestSize
        uint32_t len = (rand() % (PARTITION / 2) + 1) * (TestSize / 2);
        len = roundup(len, src_data_size);
        assert(len < kL2RSize);

        if (dst_type == 5) {
            if (src_data_size != 0) {
                dst_addr_offset = len / src_data_size * dst_data_size / 2;
            }
            dst_addr_offset = roundup(dst_addr_offset, 64);
            rd = (dst_addr_offset << 12) | hbm_dst_high_addr | (dst_type << 4) | (dst_id << 8);
            out << "l2_mov_cfg " << "rs=" << rs << " rt=" << rt
                << " rd=" << rd << " shamt=0" << std::endl;
        }

        if (dma_2d_enable) {
            src_2d_offset = len + rand() % (TestSize / 2);
            src_2d_offset = roundup(src_2d_offset, src_data_size);
            src_2d_offset = src_2d_offset >= kL2RSize ? len : src_2d_offset;
            if (kL2RSize / src_2d_offset != 0) {
                loop_2d = rand() % (kL2RSize / src_2d_offset) + 1;
            } else {
                loop_2d = 1;
            }
            loop_2d = std::max(1, (int)loop_2d);
            loop_2d = loop_2d * src_2d_offset >= kL2RSize ? 1 : loop_2d;

            if (dst_type == 5) {
                dst_2d_offset = 2 * dst_addr_offset + rand() % TestSize;
            } else {
                if (dst_data_size != 0) {
                    dst_2d_offset = len / dst_data_size * dst_data_size + rand() % TestSize;
                }
            }
            dst_2d_offset = roundup(dst_2d_offset, dst_data_size);
            if (dst_2d_offset != 0) {
                loop_2d = std::min(loop_2d, dst_size / dst_2d_offset);
            }
            loop_2d = loop_2d * dst_2d_offset >= dst_size ? 1 : loop_2d;

            out << "l2_mov_2d_cfg " << "rs=" << src_2d_offset << " rt=" << loop_2d
                << " rd=" << dst_2d_offset << " shamt=0" << std::endl;

            src_offset_max =  kL2RBankSize - roundup(loop_2d * src_2d_offset, 16) / 16;
            if (src_offset_max > 0) {
                src_offset = rand() % src_offset_max;
            } else {
                src_offset = 0;
            }
            src_offset = roundup(src_offset, src_data_size);
            src_offset = src_offset >= kL2RBankSize ? 0 : src_offset;
            src_bank = rand() % 16;
            src_addr = unified_hw_addr(src_bank, src_offset, 4, 2);

            if (dst_size > (loop_2d * src_2d_offset)) {
                dst_addr = rand() % (dst_size - loop_2d * src_2d_offset);
            } else {
                dst_addr = 0;
            }
            dst_addr = roundup(dst_addr, dst_data_size);
            dst_addr = (dst_addr + loop_2d * dst_2d_offset) >= dst_size ? 0 : dst_addr;

        } else {
            src_offset_max =  kL2RBankSize - roundup(len, 16) / 16;
            if (src_offset_max != 0) {
                src_offset = rand() % src_offset_max;
            } else {
                src_offset = 0;
            }
            src_offset = roundup(src_offset, src_data_size);
            src_offset = src_offset + (roundup(len, 16) / 16) >= kL2RBankSize ? 0 : src_offset;
            src_bank = rand() % 16;
            src_addr = unified_hw_addr(src_bank, src_offset, 4, 2);

            if (dst_type == 5) {
                if (dst_size > ( 2 * dst_addr_offset)) {
                    dst_addr = rand() % (dst_size - 2 * dst_addr_offset);
                } else {
                    dst_addr = 0;
                }
            } else {
                if ((src_data_size != 0) && (dst_size > (len / src_data_size * dst_data_size))) {
                    dst_addr = rand() % (dst_size - len / src_data_size * dst_data_size);
                } else {
                    dst_addr = 0;
                }
            }
            dst_addr = roundup(dst_addr, dst_data_size);
            if (src_data_size != 0) {
                dst_addr = (dst_addr + len / src_data_size * dst_data_size) >= dst_size ? 0 : dst_addr;
            }
        }
        out << "l2_mov " << "rs=" << src_addr << " rt=" << len
            << " rd=" << dst_addr << " shamt=" << i_num << std::endl;

    }

    out << "xfence" << std::endl;
    out << "unlock " << module_id << std::endl;

#if DEBUG
    std::cout << "module_id is " << module_id << std::endl;
    std::cout << "source_id is " << source_id << std::endl;
    std::cout << "dst_id is " << dst_id << std::endl;
    std::cout << "dst_type is " << dst_type << std::endl;
    std::cout << "src_type is " << src_type << std::endl;
    std::cout << "src_offset_max is " << src_offset_max << std::endl;
    std::cout << "src_offset is " << src_offset << std::endl;
    std::cout << "src_bank is " << src_bank << std::endl;
    std::cout << "src_data_size is " << src_data_size << std::endl;
    std::cout << "dst_data_size is " << dst_data_size << std::endl;
    std::cout << "findmax_enable is " << findmax_enable << std::endl;
    std::cout << "dma_2d_enable is " << dma_2d_enable << std::endl;
    std::cout << "src_2d_offset is " << src_2d_offset << std::endl;
    std::cout << "dst_2d_offset is " << dst_2d_offset << std::endl;
    std::cout << "loop_2d is " << loop_2d << std::endl;
    std::cout << "len is " << len << std::endl;
    std::cout << "src_addr is " << src_addr << std::endl;
    std::cout << "dst_addr is " << dst_addr << std::endl;
#endif

}

void gen_dma_readl2wd(std::ostream& out) {
    int cmd_times= rand() % CMD_TIMES + 1;
    int module_id = 7; // 7:DMAO
    int source_id = 0;
    int dst_id = 0;
    int src_type = 0;
    int dst_type = 0;
    uint32_t hbm_src_high_addr = 0;
    uint32_t hbm_dst_high_addr = 0;
    uint32_t rd = 0;
    uint32_t rs = 0;
    uint32_t rt = 0;
    float max = 0;
    uint32_t dst_2d_offset = 0;
    uint32_t src_2d_offset = 0;
    uint32_t dst_addr_offset = 0;
    uint32_t loop_2d = 0;
    int dma_2d_enable = 0;
    int findmax_enable = 0;
    uint32_t i_num = 0;

    uint32_t src_offset_max = 0;
    uint32_t src_offset = 0;
    uint32_t src_bank = 0;
    uint32_t src_addr = 0;
    uint32_t dst_addr = 0;
    int src_data_size = 0;
    int dst_data_size = 0;
    uint32_t dst_size = 0;
    uint32_t pre_dst_addr_start = kL2WDBankSize;
    uint32_t pre_dst_addr_end = kL2WDBankSize;

    out << "lock " << module_id << std::endl;

    source_id = rand() % 2 + 2; //2:L2W 3:L2D
    dst_id = rand() % 3; //0:HBM 2:L2W 3:L2D
    dst_id = (dst_id == 1) ? 3 : dst_id;
    // src_type:
    // 3'b000:FLOAT32
    // 3'b001:FLOAT16
    // 3'b010:INT16
    // 3'b011:BFLOAT16
    // 3'b100:INT8
    // dst_type:
    // 3'b000:FLOAT32
    // 3'b001:FLOAT16
    // 3'b010:INT16
    // 3'b011:BFLOAT16
    // 3'b100:INT8
    // 3'b101:INT31
    src_type = rand() % 5; //0:fp32 1:fp16 2:int16 3:bfp16 4:int8
    switch (src_type) {
        case 0: //fp32
            src_data_size = 4;
            dst_type = (rand() % 4) * 2; //0:fp32 2:int16 4:int8
            if (dst_type == 6) {
                dst_type = 5; // 5:int31
            }
            break;
        case 1: //fp16
            src_data_size = 2;
            dst_type = pow(2, rand() % 3); //1:fp16 2:int16 4:int8
            break;
        case 2: //int16
            src_data_size = 2;
            dst_type = 2; //2:int16
            break;
        case 3: //bfp16
            src_data_size = 2;
            dst_type = rand() % 3 + 2; //2:int16 3:bfp16 4:int8
            break;
        case 4: //int8
            src_data_size = 1;
            dst_type = 4; //4:int8
            break;
        default:
            break;
    }
    switch (dst_type) {
        case 0:
            dst_data_size = 4;
            break;
        case 1:
        case 2:
        case 3:
            dst_data_size = 2;
            break;
        case 4:
            dst_data_size = 1;
            break;
        case 5:
            dst_data_size = 4;
            break;
        default:
            break;
    }

    switch (dst_id) {
        case 0:
            dst_size = (uint32_t)kGlobalMemorySize;
            break;
        case 2:
        case 3:
            dst_size = (uint32_t)kL2WDBankSize;
            break;
        default:
            break;
    }

    hbm_src_high_addr = 0;
    hbm_dst_high_addr = 0;
    rd = hbm_dst_high_addr | (dst_type << 4) | (dst_id << 8);
    rs = hbm_src_high_addr | (src_type << 4) | (source_id << 8);
    max = rand() / (double)(RAND_MAX / 10000.0);
    uint32_t* max_ptr = reinterpret_cast<uint32_t*>(&max);
    rt = *max_ptr;
    out << "l2_mov_cfg " << "rs=" << rs << " rt=" << rt
        << " rd=" << rd << " shamt=0" << std::endl;

    findmax_enable = rand() % 3; //0:no findmax 1:sub findmax 2:all findmax
    dma_2d_enable = rand() % 2; //0:disable 1:enable
    i_num = (dma_2d_enable << 2) | findmax_enable;

    for (int i = 0; i < cmd_times; i++) {
        uint32_t len = (rand() % (PARTITION / 2) + 1) * (TestSize / 2);
        len = roundup(len, src_data_size);
        assert(len < kL2WDBankSize);
        if (dst_type == 5) {
            if (src_data_size != 0) {
                dst_addr_offset = len / src_data_size * dst_data_size / 2;
            }
            dst_addr_offset = roundup(dst_addr_offset, 64);
            rd = (dst_addr_offset << 12) | hbm_dst_high_addr | (dst_type << 4) | (dst_id << 8);
            out << "l2_mov_cfg " << "rs=" << rs << " rt=" << rt
                << " rd=" << rd << " shamt=0" << std::endl;
        }

        if (dma_2d_enable) {
            src_2d_offset = len + rand() % (TestSize / 2);
            src_2d_offset = roundup(src_2d_offset, src_data_size);
            src_2d_offset = src_2d_offset >= kL2WDBankSize ? len : src_2d_offset;
            if ((src_2d_offset != 0) && (kL2WDBankSize / src_2d_offset) > 0) {
                loop_2d = rand() % (kL2WDBankSize / src_2d_offset) + 1;
            } else {
                loop_2d = 1;
            }
            loop_2d = std::max(1, (int)loop_2d);
            loop_2d = loop_2d * src_2d_offset >= kL2WDBankSize ? 1 : loop_2d;

            if (src_data_size != 0) {
                dst_2d_offset = len / src_data_size * dst_data_size + rand() % (TestSize / 2);
            }
            dst_2d_offset = roundup(dst_2d_offset, dst_data_size);
            loop_2d = loop_2d * dst_2d_offset >= kL2WDBankSize ? 1 : loop_2d;

            if (source_id == dst_id) { // dst is L2W/L2D
                if (kL2WDBankSize / 2 > src_2d_offset) {
                    if (src_2d_offset != 0) {
                        loop_2d = rand() % ((kL2WDBankSize / 2) / src_2d_offset) + 1;
                    }
                } else {
                    src_2d_offset = len;
                    loop_2d = 1;
                }
                loop_2d = std::max(1, (int)loop_2d);
                dst_2d_offset = std::min(dst_2d_offset, src_2d_offset);
                src_offset_max =  kL2WDBankSize / 2 - loop_2d * src_2d_offset;
            } else {
                src_offset_max =  kL2WDBankSize - loop_2d * src_2d_offset;
            }

            out << "l2_mov_2d_cfg " << "rs=" << src_2d_offset << " rt=" << loop_2d
                << " rd=" << dst_2d_offset << " shamt=0" << std::endl;

            if (src_offset_max > 0) {
                src_offset = rand() % src_offset_max;
            } else {
                src_offset = 0;
            }
            src_offset = roundup(src_offset, src_data_size);
            src_offset = (src_offset + loop_2d * src_2d_offset) >= kL2WDBankSize ? 0 : src_offset;
            src_addr = src_offset;

            if (source_id == dst_id) { // dst is L2W/L2D
                if ((dst_size - src_offset - loop_2d * src_2d_offset) > (loop_2d * dst_2d_offset)) {
                    dst_addr = src_offset + loop_2d * src_2d_offset + rand() % (dst_size - src_offset - loop_2d * dst_2d_offset - loop_2d * src_2d_offset);
                } else {
                    if (src_offset != 0) {
                        dst_addr = rand() % src_offset;
                    } else {
                        dst_addr = 0;
                    }
                }
                dst_addr = roundup(dst_addr, dst_data_size);

                assert(dst_addr + loop_2d * dst_2d_offset < dst_size);
                assert(src_addr + loop_2d * src_2d_offset < dst_addr);
                if (dst_data_size != 0) {
                    dst_addr = (dst_addr + loop_2d * dst_2d_offset * src_data_size / dst_data_size) >= dst_size ? 0 : dst_addr;
                }
                if (((src_addr > pre_dst_addr_start) & (src_addr < pre_dst_addr_end)) ||
                    ((src_addr + loop_2d * src_2d_offset > pre_dst_addr_start) & (src_addr + loop_2d * src_2d_offset < pre_dst_addr_end))) {
                    out << "xfence" << std::endl;
                }
                pre_dst_addr_start = dst_addr;
                if (dst_data_size != 0) {
                    pre_dst_addr_end = dst_addr + loop_2d * dst_2d_offset * src_data_size / dst_data_size;
                }
            } else {
                assert(dst_size > (loop_2d * dst_2d_offset));
                if (dst_size > (loop_2d * dst_2d_offset + dst_data_size)) {
                    dst_addr = rand() % (dst_size - loop_2d * dst_2d_offset - dst_data_size);
                } else {
                    dst_addr = 0;
                }
                dst_addr = roundup(dst_addr, dst_data_size);
                assert(dst_addr + loop_2d * dst_2d_offset < dst_size);
                dst_addr = (dst_addr + loop_2d * dst_2d_offset) >= dst_size ? 0 : dst_addr;
            }

        } else {
            src_offset_max =  kL2WDBankSize - 2 * len;
            if (src_offset_max != 0) {
                src_offset = rand() % src_offset_max;
            } else {
                src_offset = 0;
            }
            src_offset = roundup(src_offset, src_data_size);
            src_offset = (src_offset + len) >= kL2WDBankSize ? 0 : src_offset;
            src_addr = src_offset;

            if (source_id == dst_id) { // dst is L2W/L2D
                if ((src_data_size != 0)&& (dst_size - src_offset - len) > (len / src_data_size * dst_data_size)) {
                    dst_addr = src_offset + len + rand() % (dst_size - src_offset - len - len / src_data_size * dst_data_size);
                } else {
                    if (src_offset != 0) {
                        dst_addr = rand() % src_offset;
                    } else {
                        dst_addr = 0;
                    }
                }
                dst_addr = roundup(dst_addr, dst_data_size);
                if (src_data_size != 0) {
                    dst_addr = (dst_addr + len / src_data_size * dst_data_size) >= dst_size ? 0 : dst_addr;
                }
                if (((src_addr > pre_dst_addr_start) & (src_addr < pre_dst_addr_end)) ||
                    ((src_addr + loop_2d * src_2d_offset > pre_dst_addr_start) & (src_addr + loop_2d * src_2d_offset < pre_dst_addr_end))) {
                    out << "xfence" << std::endl;
                }
                pre_dst_addr_start = dst_addr;
                if (src_data_size != 0) {
                    pre_dst_addr_end = dst_addr + len / src_data_size * dst_data_size;
                }
            } else {
                if ((src_data_size != 0) && (dst_size > (len / src_data_size * dst_data_size))) {
                    dst_addr = rand() % (dst_size - len / src_data_size * dst_data_size);
                } else {
                    dst_addr = 0;
                }
                dst_addr = roundup(dst_addr, dst_data_size);
                if (src_data_size != 0) {
                    dst_addr = (dst_addr + len / src_data_size * dst_data_size) >= dst_size ? 0 : dst_addr;
                }
            }
            if (src_data_size != 0) {
                assert(dst_addr + len / src_data_size * dst_data_size < dst_size);
            }

        }
        out << "l2_mov " << "rs=" << src_addr << " rt=" << len
            << " rd=" << dst_addr << " shamt=" << i_num << std::endl;
    }

    out << "xfence" << std::endl;
    out << "unlock " << module_id << std::endl;

#if DEBUG
    std::cout << "module_id is " << module_id << std::endl;
    std::cout << "source_id is " << source_id << std::endl;
    std::cout << "dst_id is " << dst_id << std::endl;
    std::cout << "dst_type is " << dst_type << std::endl;
    std::cout << "src_type is " << src_type << std::endl;
    std::cout << "src_offset_max is " << src_offset_max << std::endl;
    std::cout << "src_offset is " << src_offset << std::endl;
    std::cout << "src_bank is " << src_bank << std::endl;
    std::cout << "src_data_size is " << src_data_size << std::endl;
    std::cout << "dst_data_size is " << dst_data_size << std::endl;
    std::cout << "findmax_enable is " << findmax_enable << std::endl;
    std::cout << "dma_2d_enable is " << dma_2d_enable << std::endl;
    std::cout << "src_2d_offset is " << src_2d_offset << std::endl;
    std::cout << "dst_2d_offset is " << dst_2d_offset << std::endl;
    std::cout << "dst_addr_offset is " << dst_addr_offset << std::endl;
    std::cout << "loop_2d is " << loop_2d << std::endl;
    std::cout << "len is " << len << std::endl;
    std::cout << "src_addr is " << src_addr << std::endl;
    std::cout << "dst_addr is " << dst_addr << std::endl;
    std::cout << "kL2WDBankSize is " << kL2WDBankSize << std::endl;
#endif

}
void gen_dma_readhbm(std::ostream& out) {
    int module_id = 5; // require by ruanyuan 5:DMAI0-L2W ; 6:DMAI1-L2D
    out << "lock " << module_id << std::endl;
    for (int i = 0; i < 10; i++) {

        int cmd_times= rand() % CMD_TIMES + 1;
        int source_id = 0;
        int dst_id = 0;
        int src_type = 0;
        int dst_type = 0;
        uint32_t hbm_src_high_addr = 0;
        uint32_t hbm_dst_high_addr = 0;
        uint32_t rd = 0;
        uint32_t rs = 0;
        uint32_t rt = 0;
        float max = 0;
        uint32_t dst_2d_offset = 0;
        uint32_t dst_offset_end = 0;
        uint32_t dst_addr_offset = 0;
        uint32_t src_2d_offset = 0;
        uint32_t loop_2d = 0;

        uint32_t src_addr = 0;
        uint32_t dst_addr = 0;
        uint32_t len = 0;
        int src_data_size = 0;
        int dst_data_size = 0;

        // src_type:
        // 3'b000:FLOAT32
        // 3'b001:FLOAT16
        // 3'b010:INT16
        // 3'b011:BFLOAT16
        // 3'b100:INT8
        // 3'b101:INT31 should not as src
        // dst_type:
        // 3'b000:FLOAT32
        // 3'b001:FLOAT16
        // 3'b010:INT16
        // 3'b011:BFLOAT16
        // 3'b100:INT8
        // 3'b101:INT31
        src_type = rand() % 5;
        switch (src_type) {
            case 0: //fp32
                src_data_size = 4;
                dst_type = (rand() % 4) * 2; //0:fp32 2:int16 4:int8
                if (dst_type == 6) {
                    dst_type = 5; // 5:int31
                }
                break;
            case 1: //fp16
                src_data_size = 2;
                dst_type = pow(2, rand() % 3); //1:fp16 2:int16 4:int8
                break;
            case 2: //int16
                src_data_size = 2;
                dst_type = 2; //2:int16
                break;
            case 3: //bfp16
                src_data_size = 2;
                dst_type = rand() % 3 + 2; //2:int16 3:fp16 4:int8
                break;
            case 4: //int8
                src_data_size = 1;
                dst_type = 4; //4:int8
                break;
            default:
                break;
        }
        switch (dst_type) {
            case 0:
                dst_data_size = 4;
                break;
            case 1:
            case 2:
            case 3:
                dst_data_size = 2;
                break;
            case 4:
                dst_data_size = 1;
                break;
            case 5:
                dst_data_size = 4;
                break;
            default:
                break;
        }

        source_id = 0; //source is HBM
        if (module_id == 5) {
            dst_id = 2; //2:L2W
        } else if (module_id == 6) {
            dst_id = 3; //3:L2D
        }
        assert(source_id != dst_id);
        hbm_src_high_addr = 0;
        hbm_dst_high_addr = 0;
        rd = hbm_dst_high_addr | (dst_type << 4) | (dst_id << 8);
        rs = hbm_src_high_addr | (src_type << 4) | (source_id << 8);
        max = rand() / (double)(RAND_MAX / 10000.0);
        uint32_t* max_ptr = reinterpret_cast<uint32_t*>(&max);
        rt = *max_ptr;
        out << "l2_mov_cfg " << "rs=" << rs << " rt=" << rt
            << " rd=" << rd << " shamt=0" << std::endl;

        for (int i = 0; i < cmd_times; i++) {
            int findmax_enable = rand() % 3; //0:no findmax 1:sub findmax 2:all findmax
            int dma_2d_enable = rand() % 2; //0:disable 1:enable
            uint32_t i_num = (dma_2d_enable << 2) | findmax_enable;

            if (dma_2d_enable) {
                len = rand() % (MaxTestSize / PARTITION) + 1;
                len = roundup(len, src_data_size);

                if (dst_type == 5) {
                    if (src_data_size != 0) {
                        dst_addr_offset = len / src_data_size * dst_data_size / 2;
                    }
                    dst_addr_offset = roundup(dst_addr_offset, 64);
                    rd = (dst_addr_offset << 12) | hbm_dst_high_addr | (dst_type << 4) | (dst_id << 8);
                    out << "l2_mov_cfg " << "rs=" << rs << " rt=" << rt
                        << " rd=" << rd << " shamt=0" << std::endl;
                }

                src_2d_offset = len + rand() % MaxOffsetSize;
                src_2d_offset = roundup(src_2d_offset, src_data_size);
                if (dst_type == 5) {
                    dst_2d_offset = 2 * dst_addr_offset + rand() % MaxOffsetSize;
                } else {
                    if (src_data_size != 0) {
                        dst_2d_offset = len / src_data_size * dst_data_size  + rand() % MaxOffsetSize;
                    }
                }
                dst_2d_offset = roundup(dst_2d_offset, dst_data_size);
                if ((dst_2d_offset != 0) && ((kL2WDBankSize - dst_offset_end) / dst_2d_offset > 0)) {
                    loop_2d = rand() % ((kL2WDBankSize - dst_offset_end) / dst_2d_offset) + 1;
                } else {
                    loop_2d = 1;
                }
                loop_2d = std::max(1, (int)loop_2d);
                loop_2d = dst_offset_end + loop_2d * dst_2d_offset >= kL2WDBankSize ? 1 : loop_2d;
                dst_addr = rand() % MaxOffsetSize + dst_offset_end;
                dst_addr = roundup(dst_addr, dst_data_size);
                dst_addr = (dst_addr + loop_2d * dst_2d_offset >= kL2WDBankSize) ? dst_offset_end : dst_addr;
                dst_offset_end = dst_addr + loop_2d * dst_2d_offset;
                if (dst_offset_end > kL2WDBankSize) {
                    std::cout << "dst_offset_end > kL2WDBankSize break out..." << std::endl;
                    break;
                }
                out << "l2_mov_2d_cfg " << "rs=" << src_2d_offset << " rt=" << loop_2d
                    << " rd=" << dst_2d_offset << " shamt=0" << std::endl;

                if (kGlobalMemorySize > (loop_2d * src_2d_offset)) {
                    src_addr = rand() % (kGlobalMemorySize - loop_2d * src_2d_offset);
                } else {
                    src_addr = 0;
                }
                src_addr = roundup(src_addr, src_data_size);
                src_addr = (src_addr + loop_2d * src_2d_offset) >= kGlobalMemorySize ? 0 : src_addr;

                out << "l2_mov " << "rs=" << src_addr << " rt=" << len
                    << " rd=" << dst_addr << " shamt=" << i_num << std::endl;

            } else {
                len = rand() % MaxTestSize + 1;
                len = roundup(len, src_data_size);

                if (kGlobalMemorySize > len) {
                    src_addr = rand() % (kGlobalMemorySize - len);
                } else {
                    src_addr = 0;
                }
                src_addr = roundup(src_addr, src_data_size);

                dst_addr = rand() % MaxOffsetSize + dst_offset_end;
                dst_addr = roundup(dst_addr, dst_data_size);

                if (dst_type == 5) {
                    if (src_data_size != 0) {
                        dst_addr_offset = len / src_data_size * dst_data_size / 2;
                    }
                    dst_addr_offset = roundup(dst_addr_offset, 64);
                    rd = (dst_addr_offset << 12) | hbm_dst_high_addr | (dst_type << 4) | (dst_id << 8);
                    out << "l2_mov_cfg " << "rs=" << rs << " rt=" << rt
                        << " rd=" << rd << " shamt=0" << std::endl;
                    dst_offset_end = dst_addr + 2 * dst_addr_offset;
                } else {
                    if (src_data_size != 0) {
                        dst_offset_end = dst_addr + len / src_data_size * dst_data_size;
                    }
                }

                if (dst_offset_end > kL2WDBankSize) {
                    std::cout << "dst_offset_end > kL2WDBankSize break out..." << std::endl;
                    break;
                }
                src_addr = (src_addr + len) >= kGlobalMemorySize ? 0 : src_addr;
                out << "l2_mov " << "rs=" << src_addr << " rt=" << len
                    << " rd=" << dst_addr << " shamt=" << i_num << std::endl;
            }
        }

        out << "xfence" << std::endl;
    }

    out << "unlock " << module_id << std::endl;

#if DEBUG
    std::cout << "module_id is " << module_id << std::endl;
    std::cout << "source_id is " << source_id << std::endl;
    std::cout << "dst_id is " << dst_id << std::endl;
    std::cout << "dst_type is " << dst_type << std::endl;
    std::cout << "src_type is " << src_type << std::endl;
    std::cout << "src_data_size is " << src_data_size << std::endl;
    std::cout << "dst_data_size is " << dst_data_size << std::endl;
    std::cout << "findmax_enable is " << findmax_enable << std::endl;
    std::cout << "dma_2d_enable is " << dma_2d_enable << std::endl;
    std::cout << "src_2d_offset is " << src_2d_offset << std::endl;
    std::cout << "dst_2d_offset is " << dst_2d_offset << std::endl;
    std::cout << "loop_2d is " << loop_2d << std::endl;
    std::cout << "len is " << len << std::endl;
    std::cout << "src_addr is " << src_addr << std::endl;
    std::cout << "dst_addr is " << dst_addr << std::endl;
#endif
}

int main(int argc, char** argv) {
    google::ParseCommandLineFlags(&argc, &argv, true);

    srand(FLAGS_seed);

    std::ofstream cmd_seq;
    cmd_seq.open(FLAGS_output);

    if (FLAGS_direction == "readhbm") {
        //DMAI
        gen_dma_readhbm(cmd_seq);
    }
    else if (FLAGS_direction == "readl2r") {
        //DMAO
        gen_dma_readl2r(cmd_seq);
    }
    else if (FLAGS_direction == "readl2wd") {
        //DMAO
        gen_dma_readl2wd(cmd_seq);
    }

    //cmd_seq << "xfence" << std::endl;
    cmd_seq.close();

    return 0;
}
