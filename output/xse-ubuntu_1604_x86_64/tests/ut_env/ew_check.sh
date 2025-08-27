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
                echo -e "\033[32m OK.\033[0m" >> diff.log
            else
                echo -e "\033[31m FAIL.\033[0m" >> diff.log
            fi
        else
            echo -e "\033[33m $2 is not found, skip.\033[0m" >> diff.log
        fi
    else
        echo -e "\033[33m $1 is not found, skip.\033[0m" >> diff.log
    fi
}

func=("dsmadd" "dsmul" "dscmpnsel" "ssvsum" "sspooling" "dsdiv" "ssresize" "mix" "fussion")
rm -rf ./diff.log

for ((i = 0; i < 1; i++))
do
    seed=$(rand 0 10000)
    func_id=$(rand 0 8)

    rm -rf ./ew_cmd_seq*.dat
    rm -rf ./*_init.dat*
    ./data_gen/build/sd_cdnn_data_gen -seed=${seed}
    echo "**************CASE${i}****************" >> diff.log
    echo "seed=${seed}" >> diff.log
    echo "func_id=${func_id}" >> diff.log

    echo -en "===================================================================================\n"
    echo -en "RUN TEST \n"
    echo -en "===================================================================================\n"
    echo -en "seed = ${seed}\n"
    echo -en "func_id= ${func_id}\n"

    ./ew_cmd_gen/build/ew_cmd -func ${func[func_id]} -output "./ew_cmd_seq.dat" -seed=${seed}
    if [ $? -ne 0 ];then
        echo -en "ew_cmd_gen failed...\n"
        continue
    fi
    echo -en "RUN TEST DONE...\n"

    ./output/bin/xcdnn  \
        -t ./ew_cmd_seq.dat  \
        --dump-cdnn-instr=ew_cmd_output.cdnn \
        --init-sram=L1E --init-file=./l1e_sram_init.dat  \
        --init-sram=L2E --init-file=./l2e_sram_init.dat  \
        --dump-sram=L2E --dump-file=./S_L2E_dump.dat \
        --dump-sram=L1E --dump-file=./S_L1E_dump.dat \
        --dump-ew-max ew_max.dat

    #make ro
    ./simv -l log/sim_run.log +TESTID=0

    result_cmp ./S_L1E_dump.dat  ./H_L1E_dump.dat
    result_cmp ./S_L2E_dump.dat  ./H_L2E_dump.dat
    result_cmp ./ew_max-cluster0-core0.dat ./H_ew_max.dat

    echo -en "===================================================================================\n"
    echo -en "END TEST \n"
    echo -en "===================================================================================\n"

done
