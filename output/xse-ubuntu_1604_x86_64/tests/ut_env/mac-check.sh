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

mode=("int16" "int8" "random")
data_type=("16bit" "8bit" "8bit")

for ((i = 0; i < 10000; i++))
do
    seed=$(rand 0 10000)
    mode_id=$(rand 0 2)
    # seed=3040
    # mode_id=1

    rm -rf ./mac_cmd_seq*.dat

    echo -en "===================================================================================\n"
    echo -en "RUN TEST \n"
    echo -en "===================================================================================\n"
    echo -en "seed = ${seed}\n"
    echo -en "mode=${mode[mode_id]}\n"

    n=$(($i % 25))
    if [ "$n" -eq "0" ]; then
        echo -en "generate SRAM data whith seed = ${seed}, mode=${mode[mode_id]}\n"
        ./data_gen/build/sd_cdnn_data_gen -seed=${seed} -data_type=${data_type[mode_id]}
    fi

    ./mac_cmd/build/mac_cmd -output=./mac_cmd_seq_0.dat -mode=${mode[mode_id]} -seed=${seed}

    # echo -en "[CORE 0]\n" >> ./mac_cmd_seq.dat
    # cat ./mac_cmd_seq_0.dat  >> ./mac_cmd_seq.dat
    # echo -en "[CORE 1]\n" >> ./mac_cmd_seq.dat
    # echo -en "[CORE 2]\n" >> ./mac_cmd_seq.dat
    # echo -en "[CORE 3]\n" >> ./mac_cmd_seq.dat

    ./simulator/output/bin/xcdnn  \
        -t ./mac_cmd_seq_0.dat  \
        --init-sram=L1D --init-file=./l1d_sram_init.dat  \
        --init-sram=L1W --init-file=./l1w_sram_init.dat  \
        --init-sram=L1E --init-file=./l1e_sram_init.dat  \
        --dump-sram=L1E --dump-file=./l1e_sram_sim.dat

    # TODO run HW sim of MAC
    # make ro
    ./hwemu_mac/simv -l vcs_log_output/sim_run.log +TESTID=0

    result_cmp ./l1e_sram_sim.dat  ./l1e_sram_rst.dat

    echo -en "===================================================================================\n"
    echo -en "END TEST \n"
    echo -en "===================================================================================\n"

done
