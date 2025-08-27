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
                return 1
            else
                echo -e "\033[31m FAIL.\033[0m"
                return 0
            fi
        else
            echo -e "\033[33m $2 is not found, skip.\033[0m"
        fi
    else
        echo -e "\033[33m $1 is not found, skip.\033[0m"
    fi
}

func=("shuffle" "shuffle_coa" "win2vec" "shuffle_batch")

j=0
loop=$1

rm -rf ./failed_case/*

for ((i = 0; i < $loop ; i++))
do
    seed=$(rand 0 10000)
    func_id=$(rand 0 3)

    rm -rf ./ds_cmd_seq*.dat
    rm -rf ./*_init.dat*
    ./data_gen/build/sd_cdnn_data_gen -seed=${seed}

    echo -en "===================================================================================\n"
    echo -en "RUN TEST $i\n"
    echo -en "===================================================================================\n"
    echo -en "seed=${seed}\n"
    echo -en "func_id=${func_id}\n"

    ./ds_cmd_gen/build/ds_cmd -func ${func[func_id]} -core_id 0  -output "./ds_cmd_seq_0.dat" -seed=${seed}
    ./ds_cmd_gen/build/ds_cmd -func ${func[func_id]} -core_id 1  -output "./ds_cmd_seq_1.dat" -seed=${seed}

    echo -en "RUN TEST DONE...\n"

    echo -en "[CORE 0]\n" >> ./ds_cmd_seq.dat
    cat ./ds_cmd_seq_0.dat  >> ./ds_cmd_seq.dat
    echo -en "[CORE 1]\n" >> ./ds_cmd_seq.dat
    cat ./ds_cmd_seq_1.dat  >> ./ds_cmd_seq.dat

    ./output/bin/xcdnn  \
        -t ./ds_cmd_seq.dat  \
        --init-sram=L2D --init-file=./l2d_sram_init.dat \
        --init-sram=L2W --init-file=./l2w_sram_init.dat \
        --init-sram=L1D --init-file=./l1d_sram_init.dat \
        --init-sram=L1W --init-file=./l1w_sram_init.dat \
        --init-sram=L1E --init-file=./l1e_sram_init.dat \
        --dump-sram=L1D --dump-file=./l1d_sram_sim.dat \
        --dump-sram=L1W --dump-file=./l1w_sram_sim.dat \
        --dump-sram=L1E --dump-file=./l1e_sram_sim.dat

    #make ro
    ./simv -l log/sim_run.log +TESTID=0

    result_cmp ./l1d_sram_sim.dat  ./l1d_sram_rst.dat
    l1d_cmp_result=$?
    result_cmp ./l1w_sram_sim.dat  ./l1w_sram_rst.dat
    l1w_cmp_result=$?
    result_cmp ./l1e_sram_sim.dat  ./l1e_sram_rst.dat
    l1e_cmp_result=$?

    if [[ $l1d_cmp_result -eq 0 || $l1w_cmp_result -eq 0 || $l1e_cmp_result -eq 0 ]];then
        mkdir ./failed_case/$j
        cp ./ds_0_cmd_seq.dat ./failed_case/$j/ds_0_cmd_seq.dat -f
        echo -e "func_id = ${func_id}   seed = ${seed}" > ./failed_case/$j/seed.log
        j=`expr $j + 1`
    fi

    echo -en "===================================================================================\n"
    echo -en "END TEST $i\n"
    echo -en "===================================================================================\n"

    echo -e "$j failed"

done

echo -e "Total $loop cases, $j failed"
