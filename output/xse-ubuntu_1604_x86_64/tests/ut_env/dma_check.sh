#!/bin/bash

function rand(){
    min=$1
    max=$(($2 - $min +1))
    num=$(date +%s%N)
    echo $(($num%$max+$min))
}

function result_cmp
{
    if [ -e $1 ];then
        if [ -e $2 ];then
            cmp_len=$(cat $1 | wc -l)
            echo -e "TOTAL Lane is $cmp_len \n"
            head -n ${cmp_len} $2 > $2.tmp
            diff $1 $2.tmp > /dev/null
            if [ $? -eq 0 ];then
                echo -e "\033[32m OK.\033[0m"
            else
                echo -e "\033[31m FAIL.\033[0m"
            fi
        else
            echo -e "\033[33m $2 is not found, skip.\033[0m"
        fi
    else
        echo -e "\033[33m $1 is not found, skip.\033[0m"
    fi
}

direction=("readhbm" "readl2r" "readl2wd")
direction_file=("i" "o" "o")

for ((i = 0; i < 1; i++))
do
    seed=$(rand 0 10000)
    direction_id=$(rand 0 2)

    rm -rf ./dma_i_cmd_seq*.dat
    rm -rf ./dma_o_cmd_seq*.dat
    rm -rf ./*_init.dat*
    ./data_gen/build/sd_cdnn_data_gen -seed=${seed}

    echo -en "===================================================================================\n"
    echo -en "RUN TEST \n"
    echo -en "===================================================================================\n"
    echo -en "seed=${seed}\n"
    echo -en "direction=${direction[direction_id]}\n"

    ./dma_cmd_gen/build/dma_cmd -direction ${direction[direction_id]} -output "./dma_${direction_file[direction_id]}_cmd_seq.dat" -seed=${seed}
    echo -en "RUN TEST DONE...\n"

    ./output/bin/xcdnn  \
        -t ./dma_${direction_file[direction_id]}_cmd_seq.dat  \
        --dump-cdnn-instr=dma_cmd_output.cdnn \
        --init-sram=L2D --init-file=./l2d_sram_init.dat  \
        --init-sram=L2W --init-file=./l2w_sram_init.dat  \
        --init-sram=L2R --init-file=./l2r_sram_init.dat  \
        --init-global=hbm_init.dat \
        --dump-sram=L2D --dump-file=./l2d_sram_sim.dat \
        --dump-sram=L2W --dump-file=./l2w_sram_sim.dat \
        --dump-global=hbm_sim.dat \
        --dump-global-params=0x0:0x00400000

    #make ro

    #result_cmp ./l2w_sram_sim.dat  ./l2w_sram_rst.dat
    #result_cmp ./l2d_sram_sim.dat  ./l2d_sram_rst.dat

    echo -en "===================================================================================\n"
    echo -en "END TEST \n"
    echo -en "===================================================================================\n"

done
