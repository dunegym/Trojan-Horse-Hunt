#!/bin/bash
#COVERAGE_OPTION="-cm line+cond+tgl+branch"
COVERAGE_OPTION=""
#FSDB_OPTION="-ucli -i fsdb.do +fsdb+functions +fsdb+autoflush"
FSDB_OPTION=""
SIM_RUN_FLAGS=${COVERAGE_OPTION}${FSDB_OPTION}

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

smode=("single" "batch" "complex")
sfunc=("rs_row" "rs_col")

data_seed=$(rand 0 1000000)
echo -en "data_seed = ${data_seed}\n"
./data_gen/build/sd_cdnn_data_gen -seed=${data_seed}

for ((i = 0; i < 10000; i++))
do
    seed=$(rand 0 1000000)
    #core_id=$(rand 0 3)
    sfunc_id=$(rand 0 1)
    smode_id=$(rand 0 2)

    #seed=0
    core_id=0
    #sfunc_id=1
    #smode_id=2
    case_name=${seed}_${smode[smode_id]}_${sfunc[sfunc_id]}

    rm -rf ./rs_cmd_seq*.dat
    
    echo -en "===================================================================================\n"
    echo -en "RUN TEST \n"
    echo -en "===================================================================================\n"
    echo -en "seed = ${seed}\n"
    echo -en "core_id = ${core_id}\n"
    echo -en "mode=${smode[smode_id]}\n"
    echo -en "func=${sfunc[sfunc_id]}\n"
    
    
    ./rs_cmd/build/reshape_cmd -core_id 0 -func ${sfunc[sfunc_id]} -mode ${smode[smode_id]}  -output "./rs_cmd_seq_0.dat" -seed=${seed}
    
    
    ./simulator/output/bin/xcdnn  \
        -t ./rs_cmd_seq_0.dat  \
        --init-sram=L2E --init-file=./l2e_sram_init.dat  \
        --init-sram=L2R --init-file=./l2r_sram_init.dat  \
        --dump-sram=L2R --dump-file=./l2r_sram_sim.dat  > ./rs-sim.log 2>&1

	./hwemu_rs/simv ${SIM_RUN_FLAGS}  -cm_name ${case_name}    -l vcs_log_output/sim_run.log

    result_cmp ./l2r_sram_sim.dat  ./l2r_sram_rst.dat
    
    echo -en "===================================================================================\n"
    echo -en "END TEST \n"
    echo -en "===================================================================================\n"

done
