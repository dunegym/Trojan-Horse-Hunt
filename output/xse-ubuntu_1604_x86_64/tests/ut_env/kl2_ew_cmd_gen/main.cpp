#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <gflags/gflags.h>
#include <assert.h>
#include "common/bfloat16.h"

using bf16 = tensorflow::bfloat16;

static constexpr std::size_t L1EBankNum = 32;
static constexpr std::size_t L1EBankNumBits = 5;
static constexpr std::size_t L1EBankDepth = 6272;
static constexpr std::size_t L1EBankWidthBits = 2; 
static constexpr std::size_t L1EBankVDepth_RAW = 16;
static constexpr std::size_t L2EBankNum = 32;
static constexpr std::size_t L2EBankNumBits = 5;
static constexpr std::size_t L2EBankDepth = 2048;
static constexpr std::size_t L2EBankWidthBits = 2; 
static constexpr unsigned int EWTableLengh = 512;
static constexpr unsigned int MAX_RANGE = 10000;
static constexpr unsigned int MAX_CMD_TIMES = 5;
static constexpr unsigned int CMD_TYPE_NUM = 13;
static constexpr unsigned int CHANGE_CFG_PROPO = 20;
static constexpr unsigned int CHANGE_TABLE_PROPO = 20;
static constexpr unsigned int CHANGE_COEFFCFG_PROPO = 15;
static constexpr unsigned int POOL_PAD_SIZE_MAX = 8;
static constexpr unsigned int POOL_STRIDE_SIZE_MAX = 22;
static constexpr unsigned int POOL_FILTER_SIZE_MAX = 22;
static constexpr unsigned int POOL_IN_SIZE_MAX = 667;
static constexpr unsigned int CORE_NUM_MAX_BF16 = 131072;//max allowed 131072
static constexpr unsigned int CORE_NUM_MAX_FP32 = 65536;//max allowed 65536

DEFINE_string(output, "./sd_cdnn_ew_cmd_seq.dat", "ew_cmd_file");
DEFINE_int32(seed, 0, "random seed");
DEFINE_string(l1e_file, "./l1e_sram_init.dat", "l1e sram image file");
DEFINE_int32(limit, MAX_RANGE, "abs_max of l1e data");
DEFINE_int32(data_format, 0, "data format: 0 for float, 1 for bf16, 2 for mix");
DEFINE_int32(hazard_check_mode, 0, "hazard check mode: 0 for always off, 1 for always on, 2 for mix");

void array_disorder(int *a, int len) {
    int rand_idx = 0;
    int tmp_buf;
    int disorder_times = 2;
    
    while (disorder_times--) {
        for (int i = 0; i < len; i++) {
            rand_idx = rand() % len;
            tmp_buf = a[i];
            a[i] = a[rand_idx];
            a[rand_idx] = tmp_buf;
        }
    }
}

//RAW:To guarantee no overlap between src_addr and the dst_addr of previous wr-l1e instr
//WAR:To guarantee no overlap between dst_addr and src_addr of the current instr, or dst_addr=src_addr
bool gen_ew_addr(uint32_t src0_len, uint32_t src1_len, uint32_t dst_len, uint32_t &offset_l1e_sram_history_start, 
        uint32_t &offset_l1e_sram_history_end, uint32_t &dst_sram_id, bool is_pool, bool &output_max_index, uint32_t &src0_addr, 
        uint32_t &src1_addr, uint32_t &dst_addr, uint32_t &max_index_addr, bool is_ss_instr, bool &discard_instr, uint32_t hazard_check, int length_mode) {

    //dst_sram_id: 0 for l1e, 1 for l2e, 2 for null
    uint32_t gen_src0addr = 0;
    uint32_t gen_src1addr = 0;
    uint32_t gen_dstaddr = 0;
    bool do_xflush = false;
    bool src1_lenone = (src1_len != 0) && (src1_len < src0_len);
    uint32_t l1e_bank_depth = (length_mode && (src0_len<=L1EBankVDepth_RAW)) ? L1EBankVDepth_RAW : L1EBankDepth;

    //for src_addr; src0_len >= src1_len
    if (offset_l1e_sram_history_start == L1EBankDepth && offset_l1e_sram_history_end == 0) {
        gen_src0addr = rand() % (l1e_bank_depth-src0_len+1);
        gen_src1addr = rand() % (l1e_bank_depth-src1_len+1);
    } else if (offset_l1e_sram_history_start >= src0_len) {
        gen_src0addr = rand() % (offset_l1e_sram_history_start - src0_len);
        gen_src1addr = rand() % (offset_l1e_sram_history_start - src1_len);
    } else if ( (L1EBankDepth-offset_l1e_sram_history_end) > src0_len) {
        gen_src0addr = rand() % (L1EBankDepth - offset_l1e_sram_history_end - src0_len) + 1;
        gen_src0addr = offset_l1e_sram_history_end + gen_src0addr;
        gen_src1addr = rand() % (L1EBankDepth - offset_l1e_sram_history_end - src1_len) + 1;
        gen_src1addr = offset_l1e_sram_history_end + gen_src1addr;
    } else {
        do_xflush = true;
        gen_src0addr = rand() % (L1EBankDepth-src0_len+1);
        gen_src1addr = rand() % (L1EBankDepth-src1_len+1);
        //reset the offset history
        offset_l1e_sram_history_start = L1EBankDepth;
        offset_l1e_sram_history_end = 0;
    }
    
    //for dst_addr
    uint32_t gen_dstaddr_l1e = 0;
    uint32_t gen_dstaddr_l2e = 0;
    uint32_t gen_maxindex_addr = 0;
    //if (is_pool)
    //    std::cout << "dst_sram_id "<< dst_sram_id << "; dst_len: "<<dst_len<<"; output_max_index: "<<output_max_index<<std::endl;//debug
    if (dst_sram_id == 2) {//no write
        gen_dstaddr = 0;
    } else if (is_pool && output_max_index) {//pooling with max_index
        uint32_t src_addr_start = std::min(gen_src0addr, gen_src1addr);
        uint32_t src_addr_end = std::max((gen_src0addr + src0_len - 1), (gen_src1addr + src1_len - 1));
        if (src_addr_end + dst_len >= l1e_bank_depth) {
            if (dst_len > src_addr_start) {//no enough space left in l1e
                output_max_index = false;
                dst_sram_id = 0;//change dst_sram_id
            } else {
                gen_dstaddr_l1e = rand() % (src_addr_start-dst_len+1);
            }
        } else {
            gen_dstaddr_l1e = rand() % (l1e_bank_depth-src_addr_end-dst_len) + 1 + src_addr_end;
        }

        if (dst_len > L2EBankDepth) {
            discard_instr = true;
            return false;
        }
        gen_dstaddr_l2e = rand() % (L2EBankDepth-dst_len+1);

        if (dst_sram_id == 0) {//dst to l2e; max_index to l1e
            gen_dstaddr = gen_dstaddr_l2e;
            gen_maxindex_addr = gen_dstaddr_l1e;
        } else {//dst to l1e; max_index to l2e
            gen_dstaddr = gen_dstaddr_l1e;
            gen_maxindex_addr = gen_dstaddr_l2e;
        }

        if (hazard_check) {
            if (output_max_index && (dst_sram_id == 0)) {
                offset_l1e_sram_history_start = std::min(gen_dstaddr_l1e, offset_l1e_sram_history_start);
                offset_l1e_sram_history_end   = std::max(gen_dstaddr_l1e+dst_len, offset_l1e_sram_history_end);
            }
        } else {
            if (output_max_index) {
                offset_l1e_sram_history_start = std::min(gen_dstaddr_l1e, offset_l1e_sram_history_start);
                offset_l1e_sram_history_end   = std::max(gen_dstaddr_l1e+dst_len, offset_l1e_sram_history_end);
            }
        }
    } else if (dst_sram_id == 1) {//write to l1e;
        uint32_t src_addr_start = is_ss_instr ? gen_src0addr : std::min(gen_src0addr, gen_src1addr);
        uint32_t src_addr_end = is_ss_instr ? (gen_src0addr + src0_len - 1) : std::max((gen_src0addr + src0_len - 1), (gen_src1addr + src1_len - 1));
        //std::cout<<"src_addr_start: "<<src_addr_start<<"; src_addr_end: "<<src_addr_end<<std::endl;
        //std::cout<<"dst_len: "<<dst_len<<std::endl;
        if ((src_addr_end + dst_len) >= l1e_bank_depth) {
            if ((is_pool || src1_lenone) && (dst_len > src_addr_start) && (dst_len > L2EBankDepth)) {
                discard_instr = true;
                return false;
            } else if ((is_pool || src1_lenone) && (dst_len > src_addr_start)) {
                dst_sram_id = 0;//change dst_sram_id
                gen_dstaddr = rand() % (L2EBankDepth-dst_len+1);
            } else if (is_pool || src1_lenone) {
                gen_dstaddr = rand() % (src_addr_start-dst_len+1);
                if (hazard_check==0) {
                    offset_l1e_sram_history_start = std::min(gen_dstaddr, offset_l1e_sram_history_start);
                    offset_l1e_sram_history_end   = std::max(gen_dstaddr+dst_len, offset_l1e_sram_history_end);
                }
            } else {
                gen_dstaddr = rand() % (src_addr_start+1);
                if (hazard_check==0) {
                    offset_l1e_sram_history_start = std::min(gen_dstaddr, offset_l1e_sram_history_start);
                    offset_l1e_sram_history_end   = std::max(gen_dstaddr+dst_len, offset_l1e_sram_history_end);
                }
            }
        } else {
            gen_dstaddr = rand() % (l1e_bank_depth-src_addr_end-dst_len)+1+src_addr_end;
            if (hazard_check==0) {
                offset_l1e_sram_history_start = std::min(gen_dstaddr, offset_l1e_sram_history_start);
                offset_l1e_sram_history_end   = std::max(gen_dstaddr+dst_len, offset_l1e_sram_history_end);
            }
        }
    } else if (dst_sram_id == 0) {//write to l2e
        if (dst_len > L2EBankDepth) {
            discard_instr = true;
            return false;
        }
        gen_dstaddr = rand() % (L2EBankDepth-dst_len+1);
    }

    gen_src0addr = gen_src0addr << (L1EBankNumBits + L1EBankWidthBits);
    gen_src1addr = gen_src1addr << (L1EBankNumBits + L1EBankWidthBits);
    if (dst_sram_id == 1) {
        gen_maxindex_addr = gen_maxindex_addr << (L2EBankNumBits + L2EBankWidthBits);
        gen_dstaddr = gen_dstaddr << (L1EBankNumBits + L1EBankWidthBits);
    } else {
        gen_maxindex_addr = gen_maxindex_addr << (L1EBankNumBits + L1EBankWidthBits);
        gen_dstaddr = gen_dstaddr << (L2EBankNumBits + L2EBankWidthBits);
    }
    src0_addr = gen_src0addr;
    src1_addr = gen_src1addr;
    dst_addr = gen_dstaddr;
    max_index_addr = gen_maxindex_addr;

    return do_xflush;
}

//lut_change_mode
//0: only change activ_type
//1: change lut_range, lut_outofrange_param and activ_type
//2: change vldpart, lut_mode, lut_range, lut_outofrange_param and activ_type
//3: change split_inter_mode, lut_id, vldpart, lut_mode, lut_range, lut_outofrange_param and activ_type
//lut_pack_info=table_id<<4 + lut_mode<<2 + lut_inter_enable<<1 + lut_split_enable
void gen_ew_lut_cfg(std::ostream &out, uint32_t &lut_pack_info, uint32_t data_format, int lut_change_mode)
{
    uint32_t table_id = lut_pack_info >> 4;
    uint32_t lut_mode = (lut_pack_info >> 2) & 0x3;
    uint32_t lut_inter_enable = (lut_pack_info >> 1) & 0x1;
    uint32_t lut_split_enable = lut_pack_info & 0x1;
    uint32_t lut_split_vldpart = 0;
    uint32_t lut_inter_vldpart = 0;
    uint32_t lut_max_range = 0;
    uint32_t lut_range = 0;
    float   lut_min;
    int     lut_min_sign = rand() % 2;
    uint32_t  lut_min_i32 = 0;
    uint32_t  lut_max_i32 = 0;
    uint32_t Kunder_i32 = 0;
    uint32_t Kover_i32  = 0;
    uint32_t active_type = 0;

    if (lut_change_mode == 3) {
        table_id = rand() % 5000 + 1;
        lut_split_enable = rand() % 2;//can be 0 or 1
        lut_inter_enable = rand() % 2;//can be 0 or 1
    }
    if (lut_change_mode == 2 || lut_change_mode == 3) {
        lut_split_vldpart = rand() % 2;//can be 0 or 1
        lut_inter_vldpart = rand() % 2;//can be 0 or 1
        lut_mode = rand() % 4;
    }
    if (lut_change_mode == 2 || lut_change_mode == 3 || lut_change_mode == 1) {
        //gen lut_range
        if (lut_split_enable==1) {
            lut_max_range = 8;
        } else {
            lut_max_range = 9;
        }
        lut_range = rand() % (lut_max_range + 1);//can be [0-lut_max_range]

        //calcu lut_min
        if (lut_mode == 3) { // normal mode
            lut_min = lut_min_sign ? (double) (rand() % -100) : (double) (rand() % 100);
        } else {
            lut_min = (double) (rand() % 100);
        }
        bool sign_lut_min = (lut_min < 0);
        uint32_t* lut_min_ptr = reinterpret_cast<uint32_t*>(&lut_min);
        lut_min_i32 = *lut_min_ptr;
        if (data_format == 1) {
            lut_min_i32 = (lut_min_i32 & 0xFFFF0000) >> 16;
            lut_min_i32 = sign_lut_min ? (lut_min_i32 | 0xFFFF0000) : lut_min_i32;
        }

        //calcu lut_max
        float    lut_max;
        if (lut_inter_enable) {
            double temp = (double) (lut_max_range - lut_range);
            lut_max = lut_min + (pow(2.0, lut_max_range) - 1.0) / pow(2.0, temp);
        } else {
            lut_max = 0.0;
        }
        bool sign_lut_max = (lut_max < 0);
        uint32_t* lut_max_ptr = reinterpret_cast<uint32_t*>(&lut_max);
        lut_max_i32 = *lut_max_ptr;
        if (data_format == 1) {
            lut_max_i32 = (lut_max_i32 & 0xFFFF0000) >> 16;
            lut_max_i32 = sign_lut_max ? (lut_max_i32 | 0xFFFF0000) : lut_max_i32;
        }

        //cfg Kunderflow and Koverflow for interpolation mode
        int   under_sign = rand() % 2;
        int   over_sign = rand() % 2;
        float Kunder = under_sign ? (double) (rand() % -1000) : (double) (rand() % 1000);
        float Kover  = over_sign  ? (double) (rand() % -1000) : (double) (rand() % 1000);

        uint32_t* Kunder_ptr = reinterpret_cast<uint32_t*>(&Kunder);
        uint32_t* Kover_ptr  = reinterpret_cast<uint32_t*>(&Kover);
        Kunder_i32 = *Kunder_ptr;
        Kover_i32 = *Kover_ptr;
        if (data_format == 1) {
            Kunder_i32 = (Kunder_i32 & 0xFFFF0000) >> 16;
            Kunder_i32 = (under_sign == 1) ? (Kunder_i32 | 0xFFFF0000) : Kunder_i32;
            Kover_i32  = (Kover_i32 & 0xFFFF0000) >> 16;
            Kover_i32  = (over_sign == 1) ? (Kover_i32 | 0xFFFF0000) : Kover_i32;
        }
    }
    if (lut_change_mode == 2 || lut_change_mode == 3 || lut_change_mode == 1 || lut_change_mode == 0) {
        if (lut_mode==0) {
            active_type = rand() % 4;//can be 0 1 2 3
        } else {
            active_type = rand() % 7 + 2;//can be 0 1 4 5 6 7 8
            if (active_type==2) {
                active_type = 0;
            }
            if (active_type==3) {
                active_type = 1;
            }
        }
    }

    int rand_lut_cfg_array[5] = {0, 1, 2, 3, 4};
    array_disorder(rand_lut_cfg_array, 5);
    //0:10, 1:11, 2:13, 3:14, 4:15
    for (int i = 0; i < 5; i++) {
        if (rand_lut_cfg_array[i] == 0) {//cfg_type 10
            out << "ew_cfg " << "rs=0" << " rt=0" << " rd=" << active_type << " xfunct=10" << std::endl;
        } else if (rand_lut_cfg_array[i] == 1) {//cfg_type 11
            if (lut_change_mode == 3) {
                out << "ew_cfg " << "rs=" << lut_inter_enable << " rt=" << table_id
                    << " rd=" << lut_split_enable << " xfunct=11" << std::endl;
            }
        } else if (rand_lut_cfg_array[i] == 2) {//cfg_type 13
            if (lut_change_mode == 3 || lut_change_mode == 2) {
                out << "ew_cfg " << "rs=" << lut_inter_vldpart << " rt=" << lut_mode
                    << " rd=" << lut_split_vldpart << " xfunct=13" << std::endl;
            }
        } else if (rand_lut_cfg_array[i] == 3) {//cfg_type 14
            if (lut_change_mode == 3 || lut_change_mode == 2 || lut_change_mode == 1) {
                out << "ew_cfg " << "rs=" << lut_range << " rt=" << lut_max_i32
                    << " rd=" << lut_min_i32 << " xfunct=14" << std::endl;
            }
        } else if (rand_lut_cfg_array[i] == 4) {//cfg_type 15
            if ((lut_change_mode == 3 || lut_change_mode == 2 || lut_change_mode == 1) && lut_inter_enable) {
                out << "ew_cfg " << "rs=" << Kover_i32 << " rt=0"
                    << " rd=" << Kunder_i32 << " xfunct=15" << std::endl;
            }
        } 
    }
    lut_pack_info = (table_id << 4) + (lut_mode << 2) + (lut_inter_enable << 1) + lut_split_enable;
}

void gen_ewtable_rand(std::ostream& out, uint32_t &offset_l1e_sram_history_start, uint32_t &offset_l1e_sram_history_end, uint32_t hazard_check, int length_mode) {
    uint32_t k_addr = 0;
    uint32_t b_addr = 0;
    bool do_xflush_k = false;
    bool do_xflush_b = false;
    uint32_t null_addr0 = 0;
    uint32_t null_addr1 = 0;
    uint32_t null_addr2 = 0;
    uint32_t dst_sram_id = 2;//2:no write
    bool null_false0 = false;//not use
    bool null_false1 = false;//not use

    do_xflush_k = gen_ew_addr(EWTableLengh, 0, 0, offset_l1e_sram_history_start, offset_l1e_sram_history_end, dst_sram_id, 
            false, null_false0, k_addr, null_addr0, null_addr1, null_addr2, false, null_false1, hazard_check, length_mode);
    do_xflush_b = gen_ew_addr(EWTableLengh, 0, 0, offset_l1e_sram_history_start, offset_l1e_sram_history_end, dst_sram_id, 
            false, null_false0, b_addr, null_addr0, null_addr1, null_addr2, false, null_false1, hazard_check, length_mode);
    if (do_xflush_k || do_xflush_b) {
        out << "xflush" << " rs=3" << " rt=0" << " rd=0" << " xfunct=0" << std::endl;
    }
    
    out << "ewtable_cfg " << "rs=" << k_addr << " rt=0"
        << " rd=0" << " xfunct=" << 0 << std::endl;
    out << "ewtable_cfg " << "rs=" << b_addr << " rt=0"
        << " rd=0" << " xfunct=" << 1 << std::endl;
}

void gen_ewcoeff_nlf_rand(std::ostream& out, uint32_t &offset_l1e_sram_history_start, uint32_t &offset_l1e_sram_history_end, 
        uint32_t data_format, uint32_t &vld_core_end, uint32_t hazard_check, int length_mode) {

    //for coeff_nlf_cfg instr, vld_core_end should not be larger than bank_num
    uint32_t bank_num = (data_format==1) ? 64 : 32;
    if (vld_core_end >= bank_num) {
        uint32_t vld_core_start = rand() % bank_num;
        vld_core_end = rand() % (bank_num - vld_core_start) + vld_core_start;
        out << "ew_cfg " << "rs=" << vld_core_end << " rt=0"
            << " rd=" << vld_core_start << " xfunct=2" << std::endl;
    }

    float coeff_s = rand() / (double)(RAND_MAX) * MAX_RANGE - (MAX_RANGE / 2);
    bool sign_coeff_s = (coeff_s<0);
    uint32_t* coeff_s_ptr = reinterpret_cast<uint32_t*>(&coeff_s);
    uint32_t coeff_s_i32 = *coeff_s_ptr;
    if (data_format==1) {
        coeff_s_i32 = (coeff_s_i32 & 0xFFFF0000) >> 16;
        coeff_s_i32 = sign_coeff_s ? (coeff_s_i32 | 0xFFFF0000) : coeff_s_i32;
    }
    uint32_t coeff_sv_type = rand() % 2; //0:scalar mode    1:vector mode
    uint32_t which_coeff   = rand() % 3; //0:cfg_coeff0    1:cfg_coeff1    2:cfg_coeff2
    uint32_t coeff_nlf_cfg_type = which_coeff | (coeff_sv_type << 2);

    uint32_t coeff_nlf_addr = 0;
    uint32_t null_addr0 = 0;
    uint32_t null_addr1 = 0;
    uint32_t null_addr2 = 0;
    bool null_false0 = false;//not use
    bool null_false1 = false;//not use
    bool do_xflush = false;
    uint32_t dst_sram_id = 2;//2:no write

    if (coeff_sv_type == 0) {
        coeff_nlf_addr = coeff_s_i32;
    } else if (coeff_sv_type == 1) {
        do_xflush = gen_ew_addr(1, 0, 0, offset_l1e_sram_history_start, offset_l1e_sram_history_end, dst_sram_id, false, 
                null_false0, coeff_nlf_addr, null_addr0, null_addr1, null_addr2, false, null_false1, hazard_check, length_mode);
    }

    if (do_xflush) {
        out << "xflush" << " rs=3" << " rt=0" << " rd=0" << " xfunct=0" << std::endl;
    }
    out << "ewcoeff_nlf_cfg " << "rs=0 rt=0 rd="
        << coeff_nlf_addr << " xfunct=" << coeff_nlf_cfg_type << std::endl;
}

void gen_ewcoeff_rand(std::ostream& out, uint32_t &offset_l1e_sram_history_start, uint32_t &offset_l1e_sram_history_end, 
        uint32_t data_format, uint32_t hazard_check, int length_mode) {

    float coeff0 = rand() / (double)(RAND_MAX) * MAX_RANGE - (MAX_RANGE / 2);
    float coeff1 = rand() / (double)(RAND_MAX) * MAX_RANGE - (MAX_RANGE / 2);
    float coeff2 = rand() / (double)(RAND_MAX) * MAX_RANGE - (MAX_RANGE / 2);
    bool sign_coeff0 = (coeff0<0);
    bool sign_coeff1 = (coeff1<0);
    bool sign_coeff2 = (coeff2<0);
    uint32_t* coeff0_ptr = reinterpret_cast<uint32_t*>(&coeff0);
    uint32_t* coeff1_ptr = reinterpret_cast<uint32_t*>(&coeff1);
    uint32_t* coeff2_ptr = reinterpret_cast<uint32_t*>(&coeff2);
    uint32_t coeff0_i32 = *coeff0_ptr;
    uint32_t coeff1_i32 = *coeff1_ptr;
    uint32_t coeff2_i32 = *coeff2_ptr;
    if (data_format==1) {
        coeff0_i32 = (coeff0_i32 & 0xFFFF0000) >> 16;
        coeff0_i32 = sign_coeff0 ? (coeff0_i32 | 0xFFFF0000) : coeff0_i32;
        coeff1_i32 = (coeff1_i32 & 0xFFFF0000) >> 16;
        coeff1_i32 = sign_coeff1 ? (coeff1_i32 | 0xFFFF0000) : coeff1_i32;
        coeff2_i32 = (coeff2_i32 & 0xFFFF0000) >> 16;
        coeff2_i32 = sign_coeff2 ? (coeff2_i32 | 0xFFFF0000) : coeff2_i32;
    }

    uint32_t coeff0_type = rand() % 2; //0:staitc scalar coeff   1:static scalar-vector coeff
    uint32_t coeff1_type = rand() % 2; //0:staitc scalar coeff   1:static scalar-vector coeff
    uint32_t coeff2_type = rand() % 2; //0:staitc scalar coeff   1:static scalar-vector coeff
    uint32_t ewcoeff_cfg_type = coeff0_type | (coeff1_type << 1) | (coeff2_type << 2); //coeff0 coeff1 coeff2

    uint32_t coeff0_vector_addr_sram = 0;
    uint32_t coeff1_vector_addr_sram = 0;
    uint32_t coeff2_vector_addr_sram = 0;
    uint32_t null_addr0 = 0;
    uint32_t null_addr1 = 0;
    uint32_t null_addr2 = 0;
    bool null_false0 = false;//not use
    bool null_false1 = false;//not use
    bool do_xflush_0 = false;
    bool do_xflush_1 = false;
    bool do_xflush_2 = false;
    uint32_t dst_sram_id = 2;//2:no write

    if (coeff0_type == 0) {
        coeff0_vector_addr_sram = coeff0_i32;
    } else if (coeff0_type == 1) {
        do_xflush_0 = gen_ew_addr(1, 0, 0, offset_l1e_sram_history_start, offset_l1e_sram_history_end, dst_sram_id, false, null_false0, coeff0_vector_addr_sram, null_addr0, null_addr1, null_addr2, false, null_false1, hazard_check, length_mode);
    }
    if (coeff1_type == 0) {
        coeff1_vector_addr_sram = coeff1_i32;
    } else if (coeff1_type == 1) {
        do_xflush_1 = gen_ew_addr(1, 0, 0, offset_l1e_sram_history_start, offset_l1e_sram_history_end, dst_sram_id, false, null_false0, coeff1_vector_addr_sram, null_addr0, null_addr1, null_addr2, false, null_false1, hazard_check, length_mode);
    }
    if (coeff2_type == 0) {
        coeff2_vector_addr_sram = coeff2_i32;
    } else if (coeff2_type == 1) {
        do_xflush_2 = gen_ew_addr(1, 0, 0, offset_l1e_sram_history_start, offset_l1e_sram_history_end, dst_sram_id, false, null_false0, coeff2_vector_addr_sram, null_addr0, null_addr1, null_addr2, false, null_false1, hazard_check, length_mode);
    }

    if (do_xflush_0 || do_xflush_1 || do_xflush_2) {
        out << "xflush" << " rs=3" << " rt=0" << " rd=0" << " xfunct=0" << std::endl;
    }
    out << "ewcoeff_cfg " << "rs=" << coeff1_vector_addr_sram << " rt=" << coeff2_vector_addr_sram
        << " rd=" << coeff0_vector_addr_sram << " xfunct=" << ewcoeff_cfg_type << std::endl;
}

void gen_poolsize_cfg_rand(std::ostream& out, uint32_t &pool_output_len, uint32_t &pool_input_len, uint32_t &data_format, uint32_t &vld_core_end, int length_mode) {
    uint32_t bank_num = data_format ? 64 : 32;
    uint32_t real_len = L1EBankDepth / (vld_core_end / bank_num + 1);
    uint32_t pool_size_common = rand() % 10;//0:can be any value within limit; >0:use common value

    uint32_t pool_input_w = 0;
    uint32_t pool_input_h = 0;
    //generate input size
    //0<pool_input_w<real_len; 0<pool_input_h<real_len
    //pool_input_w*pool_input_h<=real_len
    if (pool_size_common>0) {
        pool_input_w = rand() % POOL_IN_SIZE_MAX + 1;
        pool_input_w = std::min((rand() % real_len + 1), pool_input_w);
        uint32_t pool_input_h_limit = std::min(POOL_IN_SIZE_MAX, (real_len/pool_input_w));
        pool_input_h = rand() % pool_input_h_limit + 1;
    } else {
        pool_input_w = rand() % real_len + 1;//1~read_len
        pool_input_h = rand() % (real_len / pool_input_w) + 1;
    }

    //generate pad size
    //pool_pad_left>=0, pool_pad_right>=0
    //pool_pad_up>=0, pool_pad_down>=0
    //(pool_pad_left+pool_pad_right+pool_input_w)<=2*real_len
    //(pool_pad_up+pool_pad_down+pool_input_h)<=2*real_len
    //(pool_pad_left+pool_pad_right+pool_input_w)*(pool_pad_up+pool_pad_down+pool_input_h)<=2*real_len
    uint32_t sum_left_right = rand() % ((2 * real_len / pool_input_h) - pool_input_w + 1);//0~(2*real_len-pool_input_w)
    uint32_t pool_pad_left  = rand() % (sum_left_right + 1);//0~sum_left_right
    uint32_t pool_pad_right = sum_left_right - pool_pad_left;
    uint32_t sum_up_down    = rand() % (2*real_len / (pool_pad_left+pool_pad_right+pool_input_w) - pool_input_h + 1);
    uint32_t pool_pad_up    = rand() % (sum_up_down + 1);
    uint32_t pool_pad_down  = sum_up_down - pool_pad_up;
    if (pool_size_common>0) {
        pool_pad_left   = std::min((rand() % POOL_PAD_SIZE_MAX), pool_pad_left);
        pool_pad_right  = std::min((rand() % POOL_PAD_SIZE_MAX), pool_pad_right);
        pool_pad_up     = std::min((rand() % POOL_PAD_SIZE_MAX), pool_pad_up);
        pool_pad_down   = std::min((rand() % POOL_PAD_SIZE_MAX), pool_pad_down);
    }

    //generate filter size
    //0<pool_filter_w<=(pool_pad_left+pool_pad_right+pool_input_w)
    //0<pool_filter_h<=(pool_pad_up+pool_pad_down+pool_input_h)
    uint32_t pool_filter_w = 0;
    uint32_t pool_filter_h = 0;
    pool_filter_w = rand() % (pool_input_w+pool_pad_left+pool_pad_right) + 1;
    pool_filter_h = rand() % (pool_input_h+pool_pad_up+pool_pad_down) + 1;
    if (pool_size_common>0) {
        pool_filter_w = std::min((rand() % POOL_FILTER_SIZE_MAX), pool_filter_w);
        pool_filter_h = std::min((rand() % POOL_FILTER_SIZE_MAX), pool_filter_h);
        pool_filter_w = std::max(pool_filter_w, pool_pad_left);//is it necessary? TODO check
        pool_filter_w = std::max(pool_filter_w, pool_pad_right);
        pool_filter_h = std::max(pool_filter_h, pool_pad_up);
        pool_filter_h = std::max(pool_filter_h, pool_pad_down);
        pool_filter_w = std::max(pool_filter_w, (uint32_t)1);
        pool_filter_h = std::max(pool_filter_h, (uint32_t)1);
    }

    //generate stride size
    //0<pool_stride_w<=(pool_pad_left+pool_pad_right+pool_input_w)
    //0<pool_stride_h<=(pool_pad_up+pool_pad_down+pool_input_h)
    //pool_stride_h*(pool_pad_left+pool_pad_right+pool_input_w)<=2*real_len
    uint32_t pool_stride_w = 0;
    uint32_t pool_stride_h = 0;
    pool_stride_w = rand() % (pool_input_w+pool_pad_left+pool_pad_right) + 1;
    pool_stride_h = rand() % (pool_input_h+pool_pad_up+pool_pad_down) + 1;
    pool_stride_h = std::min((2*real_len/(pool_pad_left+pool_pad_right+pool_input_w)), pool_stride_h);
    if (pool_size_common>0) {
        pool_stride_w = std::min((rand() % POOL_STRIDE_SIZE_MAX), pool_stride_w);
        pool_stride_h = std::min((rand() % POOL_STRIDE_SIZE_MAX), pool_stride_h);
        pool_stride_w = std::max(pool_stride_w, (uint32_t)1);//avoid value 0
        pool_stride_h = std::max(pool_stride_h, (uint32_t)1);//avoid value 0
    }
    //pool_stride_w = std::min((pool_input_w+pool_pad_left+pool_pad_right-pool_filter_w), pool_stride_w);
    //pool_stride_h = std::min((pool_input_h+pool_pad_up+pool_pad_down-pool_filter_h), pool_stride_h);
    //pool_stride_w = std::max(pool_stride_w, (uint32_t)1);//avoid value 0

    //consider padding, add some random margin
    uint32_t pool_output_w = 0;
    uint32_t pool_output_h = 0;
    pool_output_w = (pool_input_w + pool_pad_left + pool_pad_right - pool_filter_w) / pool_stride_w + 1;
    pool_output_h = (pool_input_h + pool_pad_up + pool_pad_down - pool_filter_h) / pool_stride_h + 1;
    pool_input_len  = pool_input_w * pool_input_h;
    pool_output_len = pool_output_w * pool_output_h;

    //if length_mode=1, length will be short to make more raw hazard
    if (length_mode) {
        pool_input_w = 2;
        pool_input_h = 2;
        pool_pad_left   = 0;
        pool_pad_right  = 0;
        pool_pad_up     = 0;
        pool_pad_down   = 0;
        pool_filter_w   = 2;
        pool_filter_h   = 2;
        pool_stride_w   = 1;
        pool_stride_h   = 1;
        pool_output_w   = 1;
        pool_output_h   = 1;
    }

    //print: randonmize the print order
    int pool_cfg_array[6] = {0, 1, 2, 3, 4, 5};
    array_disorder(pool_cfg_array, 6);
    for (int i=0; i<6; i++) {
        if (pool_cfg_array[i] == 0) {
            out << "ew_cfg " << "rs=" << pool_pad_right << " rt=0"
                << " rd=" << pool_pad_left << " xfunct=8" << std::endl;
        } else if (pool_cfg_array[i] == 1) {
            out << "ew_cfg " << "rs=" << pool_pad_down << " rt=0"
                << " rd=" << pool_pad_up << " xfunct=9" << std::endl;
        } else if (pool_cfg_array[i] == 2) {
            out << "ew_cfg " << "rs=" << pool_input_h << " rt=0"
                << " rd=" << pool_input_w << " xfunct=5"  << std::endl;
        } else if (pool_cfg_array[i] == 3) {
            out << "ew_cfg " << "rs=" << pool_filter_h << " rt=0"
                << " rd=" << pool_filter_w << " xfunct=6"  << std::endl;
        } else if (pool_cfg_array[i] == 4) {
            out << "ew_cfg " << "rs=" << pool_stride_h << " rt=0"
                << " rd=" << pool_stride_w << " xfunct=7"  << std::endl;
        } else if (pool_cfg_array[i] == 5) {
            out << "ew_cfg " << "rs=" << pool_output_h << " rt=0"
                << " rd=" << pool_output_w << " xfunct=4"  << std::endl;
        }
    }
}

//Necessary cfg cmd include findmaxvld_core_num, stream_len, activ_type, pool_input/output/filter/stride_size,table_cfg, select_cfg and coeff_cfg
void gen_ew_common_cfg_rand(std::ostream& out, uint32_t &offset_l1e_sram_history_start, uint32_t &offset_l1e_sram_history_end, uint32_t &stream_len, 
        uint32_t &pool_output_len, uint32_t &pool_input_len, int data_format_mode, uint32_t &data_format, uint32_t &vld_core_end, 
        uint32_t &lut_pack_info, int hazard_check_mode, uint32_t &hazard_check, int length_mode) {

    //data format type_mode
    if (data_format_mode==2) 
        data_format = rand() % 2; //0~1
    out << "ew_cfg " << "rs=0" << " rt=0"
        << " rd=" << data_format << " xfunct=1" << std::endl;

    int common_cfg_array[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    array_disorder(common_cfg_array, 8);
    for (int i=0; i<8; i++) {
        if (common_cfg_array[i] == 0) {//findmax cfg
            out << "ew_cfg " << "rs=0" << " rt=1"
                << " rd=0" << " xfunct=0" << std::endl;
        } else if (common_cfg_array[i] == 1) {//hazard check mode
            if (hazard_check_mode==2)
                hazard_check = rand() % 2; //0~1
            out << "ew_cfg " << "rs=0" << " rt=0"
                << " rd=" << hazard_check << " xfunct=16" << std::endl;
        } else if (common_cfg_array[i] == 2) {//vld_core_num & stream_len cfg
            uint32_t core_range_choose = rand() % 10;//0~9
            uint32_t core_range_fp32 = (core_range_choose==0) ? CORE_NUM_MAX_FP32 : ((core_range_choose==1) ?
                                                                                    4096 : 256);
            uint32_t core_range_bf16 = (core_range_choose==0) ? CORE_NUM_MAX_BF16 : ((core_range_choose==1) ?
                                                                                    8192 : 512);
            std::size_t core_start_range = (data_format==1) ? 64 : 32;
            std::size_t core_end_range = (data_format==1) ? core_range_bf16 : core_range_fp32;
            uint32_t vld_core_start = rand() % core_start_range; //0~(core_start_range-1)
            vld_core_end = rand() % (core_end_range-vld_core_start) + vld_core_start;
            stream_len = rand() % (L2EBankDepth / (vld_core_end / core_start_range + 1)) + 1; //use L2E sram size
            if (length_mode) {
                vld_core_end = rand() % (core_start_range - vld_core_start) + vld_core_start;//vld_core_start~(core_start_range-1)
                stream_len = 1;
            }
            out << "ew_cfg " << "rs=0" << " rt=0"
                << " rd=" << stream_len << " xfunct=3" << std::endl;
            out << "ew_cfg " << "rs=" << vld_core_end << " rt=0"
                << " rd=" << vld_core_start << " xfunct=2" << std::endl;

        } else if (common_cfg_array[i] == 3) {//generate ewcoeff_cfg cmd
            gen_ewcoeff_rand(out, offset_l1e_sram_history_start, offset_l1e_sram_history_end, data_format, hazard_check, length_mode);
        } else if (common_cfg_array[i] == 4) {//generate ewtable_cfg cmd
            gen_ewtable_rand(out, offset_l1e_sram_history_start, offset_l1e_sram_history_end, hazard_check, length_mode);
        } else if (common_cfg_array[i] == 5) {//lut cfg
            gen_ew_lut_cfg(out, lut_pack_info, data_format, 3);
        } else if (common_cfg_array[i] == 6) {//select_param cfg
            float select_min = rand() / (double)(RAND_MAX) * MAX_RANGE - (MAX_RANGE / 2);
            float select_max = rand() / (double)(RAND_MAX) * MAX_RANGE - (MAX_RANGE / 2);
            while (select_max < select_min) {
                select_max = select_max + 100.0;
            }
            bool sign_select_min = (select_min<0);
            bool sign_select_max = (select_max<0);
            uint32_t* select_min_ptr = reinterpret_cast<uint32_t*>(&select_min);
            uint32_t* select_max_ptr = reinterpret_cast<uint32_t*>(&select_max);
            uint32_t select_min_i32 = *select_min_ptr;
            uint32_t select_max_i32 = *select_max_ptr;
            if (data_format)
            {
                select_min_i32 = (select_min_i32 & 0xFFFF0000) >> 16;
                select_min_i32 = sign_select_min ? (select_min_i32 | 0xFFFF0000) : select_min_i32;
                select_max_i32 = (select_max_i32 & 0xFFFF0000) >> 16;
                select_max_i32 = sign_select_max ? (select_max_i32 | 0xFFFF0000) : select_max_i32;
            }
            out << "ew_cfg " << "rs=" << select_max_i32 << " rt=0"
                << " rd=" << select_min_i32 << " xfunct=12" << std::endl;
        } else if (common_cfg_array[i] == 7) {//pooling size cfg
            gen_poolsize_cfg_rand(out, pool_output_len, pool_input_len, data_format, vld_core_end, length_mode);
        }
    }
}

void add_common_randomcfg_before_compute_instr(std::ostream& out, uint32_t &offset_l1e_sram_history_start, uint32_t &offset_l1e_sram_history_end, 
        uint32_t &stream_len, int data_format_mode, uint32_t &data_format, uint32_t &vld_core_end, uint32_t &lut_pack_info,
        int hazard_check_mode, uint32_t &hazard_check, int length_mode) {

    uint32_t re_set_cfg = 0;
    re_set_cfg = rand() % CHANGE_CFG_PROPO;
    if ((re_set_cfg == 0) && (data_format_mode == 2)) {
        data_format = rand() % 2; //0~1
        out << "ew_cfg " << "rs=0" << " rt=0"
        << " rd=" << data_format << " xfunct=1" << std::endl;
    }

    //random change cfg settings
    int compute_change_cfg_array[6] = {0, 1, 2, 3, 4, 5};
    array_disorder(compute_change_cfg_array, 6);
    for (int i=0; i<6; i++) {
        if (compute_change_cfg_array[i] == 0) {//change the value of vld_core_num and stream_len
            uint32_t core_range_choose = rand() % 10;//0~9
            uint32_t core_range_fp32 = (core_range_choose==0) ? CORE_NUM_MAX_FP32 : ((core_range_choose==1) ?
                                                                                    4096 : 256);
            uint32_t core_range_bf16 = (core_range_choose==0) ? CORE_NUM_MAX_BF16 : ((core_range_choose==1) ?
                                                                                    8192 : 512);
            std::size_t core_start_range = (data_format==1) ? 64 : 32;
            std::size_t core_end_range = (data_format==1) ? core_range_bf16 : core_range_fp32;
            re_set_cfg = rand() % CHANGE_CFG_PROPO;
            if (re_set_cfg == 0 || (data_format == 0 && vld_core_end >= CORE_NUM_MAX_FP32)) {
                uint32_t vld_core_start = rand() % core_start_range; //0~(core_start_range-1)
                vld_core_end = rand() % (core_end_range - vld_core_start) + vld_core_start;
                stream_len = rand() % (L1EBankDepth / (vld_core_end / core_start_range + 1)) + 1;
                if (length_mode) {
                    vld_core_end = rand() % (core_start_range - vld_core_start) + vld_core_start;//vld_core_start~(core_start_range-1)
                    stream_len = 1;
                }
                out << "ew_cfg " << "rs=" << vld_core_end << " rt=0"
                    << " rd=" << vld_core_start << " xfunct=2" << std::endl;
                out << "ew_cfg " << "rs=0" << " rt=0"
                    << " rd=" << stream_len << " xfunct=3" << std::endl;
            }
        } else if (compute_change_cfg_array[i] == 1) {//change hazard_check mode
            re_set_cfg = rand() % CHANGE_CFG_PROPO;
            if ((re_set_cfg == 0) && (hazard_check_mode == 2)) {
                hazard_check = rand() % 2; //0~1
                out << "xflush" << " rs=3" << " rt=0" << " rd=0" << " xfunct=0" << std::endl;
                out << "ew_cfg " << "rs=0" << " rt=0"
                    << " rd=" << hazard_check << " xfunct=16" << std::endl;
            }
        } else if (compute_change_cfg_array[i] == 2) {//change lut_mode & lut_cfg
            re_set_cfg = rand() % CHANGE_CFG_PROPO;
            if (re_set_cfg == 0) {
                int lut_change_mode = rand() % 4;
                gen_ew_lut_cfg(out, lut_pack_info, data_format, lut_change_mode);
            }
        } else if (compute_change_cfg_array[i] == 3) {// change findmax cfg
            re_set_cfg = rand() % CHANGE_CFG_PROPO;
            if (re_set_cfg == 0) {
                uint32_t do_findmax = rand() % 2;
                uint32_t init_max = rand() % 2;
                float max_value = rand() / (double)(RAND_MAX) * MAX_RANGE - (MAX_RANGE / 2);
                max_value = (max_value < 0.0) ? (-max_value) : max_value;
                uint32_t* max_value_ptr = reinterpret_cast<uint32_t*>(&max_value);
                uint32_t  max_value_i32 = *max_value_ptr;
                if (data_format==1) {
                    max_value_i32 = (max_value_i32 & 0xFFFF0000) >> 16;
                }
                out << "ew_cfg " << "rs=" << max_value_i32 << " rt=" << init_max
                    << " rd=" << do_findmax << " xfunct=0" << std::endl;
            }
        } else if (compute_change_cfg_array[i] == 4) {//generate ewcoeff_cfg cmd
            re_set_cfg = rand() % CHANGE_COEFFCFG_PROPO;
            if (re_set_cfg == 0) {
                gen_ewcoeff_rand(out, offset_l1e_sram_history_start, offset_l1e_sram_history_end, data_format, hazard_check, length_mode);
            }
        } else if (compute_change_cfg_array[i] == 5) {//generate ewtable_cfg cmd
            re_set_cfg = rand() % CHANGE_TABLE_PROPO;
            if (re_set_cfg == 0) {
                gen_ewtable_rand(out, offset_l1e_sram_history_start, offset_l1e_sram_history_end, hazard_check, length_mode);
            }
        }
    }
}

void gen_ew_dsmadd_rand(std::ostream& out, uint32_t &offset_l1e_sram_history_start, uint32_t &offset_l1e_sram_history_end, 
        uint32_t &stream_len, int data_format_mode, uint32_t &data_format, uint32_t &vld_core_end, uint32_t &lut_pack_info, 
        int hazard_check_mode, uint32_t &hazard_check, int length_mode) {

    add_common_randomcfg_before_compute_instr(out, offset_l1e_sram_history_start, offset_l1e_sram_history_end, stream_len,
            data_format_mode, data_format, vld_core_end, lut_pack_info, hazard_check_mode, hazard_check, length_mode);
    uint32_t dst_sram_id = 0;
    uint32_t bank_num = data_format ? 64 : 32;
    uint32_t real_stream_len = stream_len * (vld_core_end / bank_num + 1);
    if (real_stream_len > L2EBankDepth) {
        dst_sram_id = 1; // 0:l2-e-sram 1:l1-e-sram 
    } else {
        dst_sram_id = rand() % 2; // 0:l2-e-sram 1:l1-e-sram
    }
    
    uint32_t stream_lenone = rand() % 2;//0: src1_len=stream_len; 1: src1_len=1
    uint32_t src1_len = stream_lenone ? (vld_core_end / bank_num + 1) : real_stream_len;

    uint32_t src0_addr = 0;
    uint32_t src1_addr = 0;
    uint32_t dst_addr = 0;
    uint32_t null_addr = 0;
    bool do_xflush = false;
    bool null_false = false;
    bool discard_instr = false;
    do_xflush = gen_ew_addr(real_stream_len, src1_len, real_stream_len, offset_l1e_sram_history_start, offset_l1e_sram_history_end, 
            dst_sram_id, false, null_false, src0_addr, src1_addr, dst_addr, null_addr, false, discard_instr, hazard_check, length_mode);
    uint32_t xfunct = (stream_lenone << 3) | dst_sram_id;
    if (do_xflush) {
        out << "xflush" << " rs=3" << " rt=0" << " rd=0" << " xfunct=0" << std::endl;
    }
    if (discard_instr == false) {
        out << "dsmadd " << "rs=" << src0_addr << " rt=" << src1_addr
            << " rd=" << dst_addr << " xfunct=" << xfunct << std::endl;
    }
}

void gen_ew_dsmul_rand(std::ostream& out, uint32_t &offset_l1e_sram_history_start, uint32_t &offset_l1e_sram_history_end, 
        uint32_t &stream_len, int data_format_mode, uint32_t &data_format, uint32_t &vld_core_end, uint32_t &lut_pack_info,
        int hazard_check_mode, uint32_t &hazard_check, int length_mode) {

    add_common_randomcfg_before_compute_instr(out, offset_l1e_sram_history_start, offset_l1e_sram_history_end, 
            stream_len, data_format_mode, data_format, vld_core_end, lut_pack_info, hazard_check_mode, hazard_check, length_mode);
    uint32_t dst_sram_id = 0;
    uint32_t bank_num = data_format ? 64 : 32;
    uint32_t real_stream_len = stream_len * (vld_core_end / bank_num + 1);
    if (real_stream_len > L2EBankDepth) {
        dst_sram_id = 1; // 0:l2-e-sram 1:l1-e-sram 
    } else {
        dst_sram_id = rand() % 2; // 0:l2-e-sram 1:l1-e-sram
    }

    uint32_t stream_lenone = rand() % 2;//0: src1_len=stream_len; 1: src1_len=1
    uint32_t src1_len = stream_lenone ? (vld_core_end / bank_num + 1) : real_stream_len;

    uint32_t src0_addr = 0;
    uint32_t src1_addr = 0;
    uint32_t dst_addr = 0;
    uint32_t null_addr = 0;
    bool null_false = false;
    bool do_xflush = false;
    bool discard_instr = false;
    do_xflush = gen_ew_addr(real_stream_len, src1_len, real_stream_len, offset_l1e_sram_history_start, offset_l1e_sram_history_end, 
            dst_sram_id, false, null_false, src0_addr, src1_addr, dst_addr, null_addr, false, discard_instr, hazard_check, length_mode);
    uint32_t xfunct = (stream_lenone << 3) | dst_sram_id;
    if (do_xflush) {
        out << "xflush" << " rs=3" << " rt=0" << " rd=0" << " xfunct=0" << std::endl;
    }
    if (discard_instr == false) {
        out << "dsmul " << "rs=" << src0_addr << " rt=" << src1_addr
            << " rd=" << dst_addr << " xfunct=" << xfunct << std::endl;
    }
}

void gen_ew_sslut_rand(std::ostream& out, uint32_t &offset_l1e_sram_history_start, uint32_t &offset_l1e_sram_history_end, 
        uint32_t &stream_len, int data_format_mode, uint32_t &data_format, uint32_t &vld_core_end, uint32_t &lut_pack_info,
        int hazard_check_mode, uint32_t &hazard_check, int length_mode) {

    add_common_randomcfg_before_compute_instr(out, offset_l1e_sram_history_start, offset_l1e_sram_history_end, 
            stream_len, data_format_mode, data_format, vld_core_end, lut_pack_info, hazard_check_mode, hazard_check, length_mode);
    uint32_t dst_sram_id = 0;
    uint32_t bank_num = data_format ? 64 : 32;
    uint32_t real_stream_len = stream_len * (vld_core_end / bank_num + 1);
    if (real_stream_len > L2EBankDepth) {
        dst_sram_id = 1; // 0:l2-e-sram 1:l1-e-sram 
    } else {
        dst_sram_id = rand() % 2; // 0:l2-e-sram 1:l1-e-sram
    }

    uint32_t src0_addr = 0;
    uint32_t src1_addr = 0;
    uint32_t dst_addr = 0;
    uint32_t null_addr = 0;
    bool null_false0 = false;
    bool null_false1 = false;
    bool do_xflush = false;
    do_xflush = gen_ew_addr(real_stream_len, 0, real_stream_len, offset_l1e_sram_history_start, offset_l1e_sram_history_end, 
            dst_sram_id, false, null_false0, src0_addr, src1_addr, dst_addr, null_addr, true, null_false1, hazard_check, length_mode);
    if (do_xflush) {
        out << "xflush" << " rs=3" << " rt=0" << " rd=0" << " xfunct=0" << std::endl;
    }
    out << "sslut " << "rs=" << src0_addr << " rt=0"
        << " rd=" << dst_addr << " xfunct=" << dst_sram_id << std::endl;
}

void gen_ew_dscmpnsel_rand(std::ostream& out, uint32_t &offset_l1e_sram_history_start, uint32_t &offset_l1e_sram_history_end, 
        uint32_t &stream_len, int data_format_mode, uint32_t &data_format, uint32_t &vld_core_end, uint32_t &lut_pack_info,
        int hazard_check_mode, uint32_t &hazard_check, int length_mode) {

    add_common_randomcfg_before_compute_instr(out, offset_l1e_sram_history_start, offset_l1e_sram_history_end, 
            stream_len, data_format_mode, data_format, vld_core_end, lut_pack_info, hazard_check_mode, hazard_check, length_mode);
    uint32_t dst_sram_id = 0;
    uint32_t bank_num = data_format ? 64 : 32;
    uint32_t real_stream_len = stream_len * (vld_core_end / bank_num + 1);
    if (real_stream_len > L2EBankDepth) {
        dst_sram_id = 1; // 0:l2-e-sram 1:l1-e-sram 
    } else {
        dst_sram_id = rand() % 2; // 0:l2-e-sram 1:l1-e-sram
    }
    uint32_t cmp_type = rand() % 2; // 0:max 1:min
    uint32_t data_structure = rand() % 2; // 0:sv 1:vv
    uint32_t stream_lenone = (data_structure == 1) ? rand() % 2 : 0;//0: src1_len=stream_len; 1: src1_len=1
    uint32_t src1_len = stream_lenone ? (vld_core_end / bank_num + 1) : real_stream_len;

    uint32_t src0_addr = 0;
    uint32_t src1_addr = 0;
    uint32_t dst_addr = 0;
    uint32_t null_addr = 0;
    bool null_false = false;
    bool do_xflush = false;
    bool discard_instr = false;
    bool is_ss_instr = (data_structure==1) ? false : true;
    do_xflush = gen_ew_addr(real_stream_len, src1_len, real_stream_len, offset_l1e_sram_history_start, offset_l1e_sram_history_end, 
            dst_sram_id, false, null_false, src0_addr, src1_addr, dst_addr, null_addr, is_ss_instr, discard_instr, hazard_check, length_mode);
    if (do_xflush) {
        out << "xflush" << " rs=3" << " rt=0" << " rd=0" << " xfunct=0" << std::endl;
    }
    uint32_t dscmpnsel_xfunct = (stream_lenone << 3) | (cmp_type << 2) | (data_structure << 1) | dst_sram_id;
    if (discard_instr == false) {
        out << "dscmpnsel " << "rs=" << src0_addr << " rt=" << src1_addr
            << " rd=" << dst_addr << " xfunct=" << dscmpnsel_xfunct << std::endl;
    }
}

void gen_ew_dsselect_rand(std::ostream& out, uint32_t &offset_l1e_sram_history_start, uint32_t &offset_l1e_sram_history_end, 
        uint32_t &stream_len, int data_format_mode, uint32_t &data_format, uint32_t &vld_core_end, uint32_t &lut_pack_info,
        int hazard_check_mode, uint32_t &hazard_check, int length_mode) {

    add_common_randomcfg_before_compute_instr(out, offset_l1e_sram_history_start, offset_l1e_sram_history_end, 
            stream_len, data_format_mode, data_format, vld_core_end, lut_pack_info, hazard_check_mode, hazard_check, length_mode);
    uint32_t dst_sram_id = 0;
    uint32_t bank_num = data_format ? 64 : 32;
    uint32_t real_stream_len = stream_len * (vld_core_end / bank_num + 1);
    if (real_stream_len > L2EBankDepth) {
        dst_sram_id = 1; // 0:l2-e-sram 1:l1-e-sram 
    } else {
        dst_sram_id = rand() % 2; // 0:l2-e-sram 1:l1-e-sram
    }

    uint32_t cmp_type = rand() % 4;
    uint32_t src_type = rand() % 7;
    uint32_t src0_addr = 0;
    uint32_t src1_addr = 0;
    uint32_t dst_addr = 0;
    uint32_t null_addr = 0;
    bool null_false0 = false;
    bool null_false1 = false;
    bool do_xflush = false;
    do_xflush = gen_ew_addr(real_stream_len, real_stream_len, real_stream_len, offset_l1e_sram_history_start, offset_l1e_sram_history_end, 
            dst_sram_id, false, null_false0, src0_addr, src1_addr, dst_addr, null_addr, false, null_false1, hazard_check, length_mode);
    if (do_xflush) {
        out << "xflush" << " rs=3" << " rt=0" << " rd=0" << " xfunct=0" << std::endl;
    }
    uint32_t dsselect_xfunct = (cmp_type << 4) | (src_type << 1) | dst_sram_id;
    out << "dsselect " << "rs=" << src0_addr << " rt=" << src1_addr
        << " rd=" << dst_addr << " xfunct=" << dsselect_xfunct << std::endl;
}

void gen_ew_ssreduce_rand(std::ostream& out, uint32_t &offset_l1e_sram_history_start, uint32_t &offset_l1e_sram_history_end, 
        uint32_t &stream_len, int data_format_mode, uint32_t &data_format, uint32_t &vld_core_end, uint32_t &lut_pack_info,
        int hazard_check_mode, uint32_t &hazard_check, int length_mode) {

    add_common_randomcfg_before_compute_instr(out, offset_l1e_sram_history_start, offset_l1e_sram_history_end, 
            stream_len, data_format_mode, data_format, vld_core_end, lut_pack_info, hazard_check_mode, hazard_check, length_mode);
    uint32_t dst_sram_id = rand() % 2; // 0:l2-e-sram 1:l1-e-sram 
    uint32_t reduce_type = rand() % 7; // 0:sum 1:mean 2:max 3:max_abs 4:min 5:min_abs 6:prod
    uint32_t bank_num = data_format ? 64 : 32;
    uint32_t real_stream_len = stream_len * (vld_core_end / bank_num + 1);
    uint32_t real_dst_len = 1 * (vld_core_end / bank_num + 1);
    uint32_t src0_addr = 0;
    uint32_t src1_addr = 0;
    uint32_t dst_addr = 0;
    uint32_t null_addr = 0;
    bool null_false0 = false;
    bool null_false1 = false;
    bool do_xflush = false;
    do_xflush = gen_ew_addr(real_stream_len, 0, real_dst_len, offset_l1e_sram_history_start, offset_l1e_sram_history_end, 
            dst_sram_id, false, null_false0, src0_addr, src1_addr, dst_addr, null_addr, true, null_false1, hazard_check, length_mode);
    if (do_xflush) {
        out << "xflush" << " rs=3" << " rt=0" << " rd=0" << " xfunct=0" << std::endl;
    }
    uint32_t ssreduce_xfunct = (reduce_type << 1) | dst_sram_id;
    out << "ssreduce " << "rs=" << src0_addr << " rt=0"
        << " rd=" << dst_addr << " xfunct=" << ssreduce_xfunct << std::endl;
}

void gen_ew_sspooling_rand(std::ostream& out, uint32_t &offset_l1e_sram_history_start, uint32_t &offset_l1e_sram_history_end, 
        uint32_t &stream_len, uint32_t &pool_output_len, uint32_t &pool_input_len, int data_format_mode, uint32_t &data_format, 
        uint32_t &vld_core_end, uint32_t &lut_pack_info, int hazard_check_mode, uint32_t &hazard_check, int length_mode) {

    //random change cfg settings
    add_common_randomcfg_before_compute_instr(out, offset_l1e_sram_history_start, offset_l1e_sram_history_end, 
            stream_len, data_format_mode, data_format, vld_core_end, lut_pack_info, hazard_check_mode, hazard_check, length_mode);
    uint32_t bank_num = data_format ? 64 : 32;
    uint32_t bank_loop_num = vld_core_end / bank_num + 1;
    uint32_t re_set_cfg = rand() % CHANGE_CFG_PROPO;
    uint32_t l1e_bank_depth = length_mode ? L1EBankVDepth_RAW : L1EBankDepth;
    //std::cout<<"pool_input_len: "<<pool_input_len<<"; pool_output_len: "<<pool_output_len<<"; bank_loop_num: "<<bank_loop_num<<std::endl;
    //std::cout<<"offset_l1e_sram_history_start: "<<offset_l1e_sram_history_start<<"; offset_l1e_sram_history_end: "<<offset_l1e_sram_history_end<<std::endl;
    if ((re_set_cfg == 0) || (bank_loop_num * pool_input_len > l1e_bank_depth) || (bank_loop_num * pool_output_len > L2EBankDepth)) {//change pool_size cfg
        gen_poolsize_cfg_rand(out, pool_output_len, pool_input_len, data_format, vld_core_end, length_mode);
    }

    uint32_t real_pool_input_len = pool_input_len * (vld_core_end / bank_num + 1);
    uint32_t real_pool_output_len = pool_output_len * (vld_core_end / bank_num + 1);
    uint32_t dst_sram_id = 0;
    if (real_pool_output_len + real_pool_input_len > L2EBankDepth) {
        dst_sram_id = 1;
    } else {
        dst_sram_id = rand() % 2; // 0:l2-e-sram 1:l1-e-sram
    }

    uint32_t pool_type = rand() % 2; // 0:max_pool 1:ave_pool
    uint32_t max_index = rand() % 2; // 0:null 1:output max_index
    uint32_t pool_pad = rand() % 2; // 0:include pad 1:exclude pad
    uint32_t src0_addr = 0;
    uint32_t src1_addr = 0;
    uint32_t max_index_addr = 0;
    uint32_t dst_addr = 0;
    bool do_xflush = false;
    bool discard_instr = false;
    bool output_maxindex = (max_index == 1);
    //std::cout<<"real_pool_input_len: "<<real_pool_input_len<<"; real_pool_output_len: "<<real_pool_output_len<<std::endl;
    //std::cout<<"offset_l1e_sram_history_start: "<<offset_l1e_sram_history_start<<"; offset_l1e_sram_history_end: "<<offset_l1e_sram_history_end<<std::endl;
    do_xflush = gen_ew_addr(real_pool_input_len, 0, real_pool_output_len, offset_l1e_sram_history_start, offset_l1e_sram_history_end, 
            dst_sram_id, true, output_maxindex, src0_addr, src1_addr, dst_addr, max_index_addr, true, discard_instr, hazard_check, length_mode);
    if (do_xflush) {
        out << "xflush" << " rs=3" << " rt=0" << " rd=0" << " xfunct=0" << std::endl;
    }
    max_index = output_maxindex ? 1 : 0;
    uint32_t sspooling_xfunct = (pool_pad << 4) | (pool_type << 3) | (max_index << 2) | dst_sram_id;
    max_index_addr = (pool_type == 0 && max_index == 1) ? max_index_addr : 0;
    if (discard_instr == false) {
        out << "sspooling " << "rs=" << src0_addr << " rt=" << max_index_addr
            << " rd=" << dst_addr << " xfunct=" << sspooling_xfunct << std::endl;
    }
}

void gen_ew_ldsd_rand(std::ostream& out, uint32_t data_format, uint32_t &vld_core_end) {

    uint32_t wb_addr = rand() % 32;// cluster has 32 common registers
    int ldsd_type = rand() % 8;//0~7

    if (ldsd_type <= 5) {
        out << "ld_sd " << "rs=3" << " rt=0" << " rd=0" << " xfunct=0" << " wb_address=" << wb_addr << std::endl;
    } else if (ldsd_type == 6) {
        out << "ld_sd " << "rs=3" << " rt=1" << " rd=0" << " xfunct=0" << " wb_address=" << wb_addr << std::endl;
    } else {//ldsd_type==7
        //change vld_core_id
        uint32_t bank_num = (data_format==1) ? 64 : 32;
        uint32_t vld_core_start = rand() % bank_num;
        uint32_t vld_core_end = vld_core_start;
        out << "ew_cfg " << "rs=" << vld_core_end << " rt=0"
            << " rd=" << vld_core_start << " xfunct=2" << std::endl;
        //stream length
        out << "ew_cfg rs=0 rt=0 rd=1 xfunct=3" << std::endl;
        //calcu sram addr
        uint32_t sram_access_addr = rand() % L1EBankDepth;
        sram_access_addr = sram_access_addr << (L1EBankNumBits + L1EBankWidthBits);
        //change findmax cfg
        out << "ew_cfg " << "rs=0" << " rt=0"
            << " rd=2" << " xfunct=0" << std::endl;
        out << "ew_cfg rs=0 rt=0 rd=0 xfunct=10" << std::endl;
        out << "sslut rs=0 rt=0 rd=0 xfunct=1" << std::endl;
        out << "ld_sd " << "rs=4" << " rt=" << sram_access_addr << " rd=0" << " xfunct=1" << " wb_address=" << wb_addr << std::endl;
        out << "ew_cfg " << "rs=0" << " rt=0"
            << " rd=1" << " xfunct=0" << std::endl;
    }
}

int main(int argc, char** argv) {
    google::ParseCommandLineFlags(&argc, &argv, true);
    srand(FLAGS_seed);
    std::ofstream cmd_seq;
    cmd_seq.open(FLAGS_output);

    int data_format_mode = FLAGS_data_format;
    int hazard_check_mode = FLAGS_hazard_check_mode;
    uint32_t data_format = (data_format_mode==2||data_format_mode==0) ? 0 : 1;
    uint32_t hazard_check = (hazard_check_mode==2||hazard_check_mode==0) ? 0 : 1;
    //0: normal random stream len; 1: short stream len to get more raw case
    int  length_mode = (hazard_check_mode==1) ? (rand() % 2) : 0;

    uint32_t module_id = 3;
    cmd_seq << "lock " <<  module_id << std::endl;

    //use these 5 param to avoid RAW in L1E_SRAM and insert xflush if needed
    uint32_t offset_l1e_sram_history_end = 0;
    uint32_t offset_l1e_sram_history_start = L1EBankDepth;
    uint32_t stream_len = 0;
    uint32_t vld_core_end = 0;
    uint32_t pool_output_len = 0;
    uint32_t pool_input_len = 0;
    uint32_t lut_pack_info = 0;

    /*Necessary cfg cmd include vld_core_num, stream_len, pool_input/output/filter/stride_size,
     * table_cfg, select_cfg and coeff_cfg
     */
    gen_ew_common_cfg_rand(cmd_seq, offset_l1e_sram_history_start, offset_l1e_sram_history_end, stream_len, pool_output_len, pool_input_len, 
            data_format_mode, data_format, vld_core_end, lut_pack_info, hazard_check_mode, hazard_check, length_mode);

    int func_type_rand = 0;
    int cmd_times_rand = rand() % MAX_CMD_TIMES + 1;

    //each time, use one random type computing instruction
    for (int i=0; i<cmd_times_rand; i++) {
        func_type_rand = rand() % CMD_TYPE_NUM; // ld_sd + 7 compute types = 8
        if (func_type_rand == 0) { //dsmadd
            gen_ew_dsmadd_rand(cmd_seq, offset_l1e_sram_history_start, offset_l1e_sram_history_end, stream_len, data_format_mode, data_format, vld_core_end, lut_pack_info, hazard_check_mode, hazard_check, length_mode);
        } else if (func_type_rand == 1) { //dsmul
            gen_ew_dsmul_rand(cmd_seq, offset_l1e_sram_history_start, offset_l1e_sram_history_end, stream_len, data_format_mode, data_format, vld_core_end, lut_pack_info, hazard_check_mode, hazard_check, length_mode);
        } else if (func_type_rand == 2) { //dscmpnsel
            gen_ew_dscmpnsel_rand(cmd_seq, offset_l1e_sram_history_start, offset_l1e_sram_history_end, stream_len, data_format_mode, data_format, vld_core_end, lut_pack_info, hazard_check_mode, hazard_check, length_mode);
        } else if (func_type_rand == 3) { // ssreduce
            gen_ew_ssreduce_rand(cmd_seq, offset_l1e_sram_history_start, offset_l1e_sram_history_end, stream_len, data_format_mode, data_format, vld_core_end, lut_pack_info, hazard_check_mode, hazard_check, length_mode);
        } else if (func_type_rand == 4) { // sspooling
            gen_ew_sspooling_rand(cmd_seq, offset_l1e_sram_history_start, offset_l1e_sram_history_end, stream_len, pool_output_len, pool_input_len, data_format_mode, data_format, vld_core_end, lut_pack_info, hazard_check_mode, hazard_check, length_mode);
        } else if (func_type_rand == 5) { // sslut
            gen_ew_sslut_rand(cmd_seq, offset_l1e_sram_history_start, offset_l1e_sram_history_end, stream_len, data_format_mode, data_format, vld_core_end, lut_pack_info, hazard_check_mode, hazard_check, length_mode);
        } else if (func_type_rand == 6) { //dsselect
            gen_ew_dsselect_rand(cmd_seq, offset_l1e_sram_history_start, offset_l1e_sram_history_end, stream_len, data_format_mode, data_format, vld_core_end, lut_pack_info, hazard_check_mode, hazard_check, length_mode);
        } else if (func_type_rand == 7) { //coeff_nlf
            gen_ewcoeff_nlf_rand(cmd_seq, offset_l1e_sram_history_start, offset_l1e_sram_history_end, data_format, vld_core_end, hazard_check, length_mode);
        } else if (func_type_rand == 8) { //xflag
            cmd_seq << "xflag " << "rs=" << module_id << " rt=0" << " rd=0" << " xfunct=0" << std::endl;
        } else if (func_type_rand == 9) { // xflush
            cmd_seq << "xflush" << " rs=3" << " rt=0" << " rd=0" << " xfunct=0" << std::endl;
            offset_l1e_sram_history_end = 0;
            offset_l1e_sram_history_start = L1EBankDepth;
        } else if (func_type_rand >= 10) { //ld_sd
            gen_ew_ldsd_rand(cmd_seq, data_format, vld_core_end);
        }
    }

    cmd_seq << "xflag " << "rs=" << module_id << " rt=0" << " rd=0" << " xfunct=0" << std::endl;
    cmd_seq << "unlock " << module_id << std::endl;
    cmd_seq.close();

    return 0;

}

