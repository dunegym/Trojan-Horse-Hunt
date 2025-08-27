#!/usr/bin/env python3

import warnings
warnings.filterwarnings('ignore')

import os
import argparse
import subprocess
import multiprocessing as mp
import time
import shutil
import pandas as pd
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn
)

# define variables to use in processes
def sync_variable():
    globals()["result_condition"] = mp.Condition()
    globals()["finished"] = mp.Value('L', 0)

# this function use to run test in one process
def run_test(filter_str, args, api_path):

    # create test folder
    if os.path.exists(f"perf_{args.xpu_type}/{filter_str}"):
        shutil.rmtree(f"perf_{args.xpu_type}/{filter_str}")
    os.makedirs(f"perf_{args.xpu_type}/{filter_str}")

    old_cwd = os.getcwd()
    os.chdir(f"perf_{args.xpu_type}/{filter_str}")

    if args.set_seed:
        command_str = f"../../{api_path}/output/test/test_refactor --gtest_filter={filter_str} -seed={args.seed} >> run.log 2>&1"
    if args.xml_output:
        command_str = f"../../{api_path}/output/test/test_refactor --gtest_filter={filter_str} --gtest_output=xml:{args.xml_path}/{filter_str}.xml >> run.log 2>&1"
    else:
        command_str = f"../../{api_path}/output/test/test_refactor --gtest_filter={filter_str} >> run.log 2>&1"

    if args.perf_test:
        command_str = f"LD_PRELOAD={args.gperftools_path} CPUPROFILE=perf.prof " + command_str

    if args.time_out != -1:
        timeout = args.time_out
    else:
        timeout = None

    try:
        with open(f"run.log", 'w') as log_file:
            print(command_str, file=log_file)

        last_time = 0
        if args.time_test:
            start_time = time.time()

        #  run test shell command
        completed_process = subprocess.run(args=command_str, shell=True, timeout=timeout)

        if args.time_test:
            end_time = time.time()
            last_time = end_time - start_time

        if completed_process.returncode == 0:
            if args.perf_test:
                # transfer perf data to text and svg
                ret = subprocess.run(f"pprof --text ../../{api_path}/output/test/test_refactor perf.prof > perf.text 2> /dev/null",
                                    shell=True)
                if ret.returncode != 0:
                    with result_condition: # pylint:disable=undefined-variable
                        finished.value += 1 # pylint:disable=undefined-variable
                    os.chdir(old_cwd)
                    return [-3302, last_time]

                ret = subprocess.run(f"pprof --svg ../../{api_path}/output/test/test_refactor perf.prof > perf.svg 2> /dev/null",
                                    shell=True)
                if ret.returncode != 0:
                    with result_condition: # pylint:disable=undefined-variable
                            finished.value += 1 # pylint:disable=undefined-variable
                    os.chdir(old_cwd)
                    return [-3303, last_time]

            with result_condition: # pylint:disable=undefined-variable
                finished.value += 1 # pylint:disable=undefined-variable
            os.chdir(old_cwd)
            return [0, last_time]

        else:
            # command failed
            with result_condition: # pylint:disable=undefined-variable
                finished.value += 1 # pylint:disable=undefined-variable  
            os.chdir(old_cwd)         
            return [completed_process.returncode, last_time]

    except subprocess.TimeoutExpired:
        os.system(f"echo timeout error !!! >> run.log")
        with result_condition: # pylint:disable=undefined-variable
            finished.value += 1 # pylint:disable=undefined-variable    
        os.chdir(old_cwd)
        return [-3301, last_time]

# argument parser
def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-xt', '--xpu_type', type=str, default='', help='run tests in which xpu, xpu1/xpu2/xpu3')

    parser.add_argument('-st', '--show_test', action="store_true", help='show selected or all tests')
    parser.add_argument('-rt', '--run_test', action="store_true", help='run selected tests')
    parser.add_argument('-tt', '--time_test', action="store_true", help='run selected tests and count run last time')
    parser.add_argument('-pt', '--perf_test', action="store_true", help='run selected tests and analyse performance')
    parser.add_argument('-cl', '--classify', action="store_true", help='judge api use cluster or sdnn')

    parser.add_argument('-tg', '--together', action="store_true", help='run all cases with gtest_filter, default filter_string is *xpu_type')
    parser.add_argument('-gf', '--gtest_filter', type=str, default="", help="filter string used for gtest with --together")
    parser.add_argument('-ssd', '--set_seed', action="store_true", help='if set gtest cases seed or not')
    parser.add_argument('-sd', '--seed', type=int, default=1, help="gtest cases seed, default 1")  
    parser.add_argument('-tn', '--thread_num', type=int, default=1, help="run tests in how many threads")   
    parser.add_argument('-c', '--cases', type=str, default="", help="select cases which name has these strings, this can be string or file")
    parser.add_argument('-to', '--time_out', type=int, default=-1, help="if one thread run time greater than this, stop and fail")  
    parser.add_argument('-ap', '--api_path', type=str, default="../../../api", help="set api code path to find api test cases, default ../../../api ") 
    parser.add_argument('-rp', '--runtime_path', type=str, default="../../../runtime_output/output/so/", help="set runtime so path to use runtime, default ../../../runtime_output/output/so/ ") 
    parser.add_argument('-sp', '--simulator_path', type=str, default="../../output/so/", help="set simulator so path to use simulator, default ../../output/so/ ") 
    parser.add_argument('-pp', '--gperftools_path', type=str, default="/usr/local/lib/libprofiler.so", help="google-perftools so path.")
    parser.add_argument('-xml', '--xml_output', action="store_true", help="if save gtest output xml or not")
    parser.add_argument('-xp', '--xml_path', type=str, default="${WORKSPACE}/test_result_xml/", help="api gtest output xml path.")
    parser.add_argument('-nd', '--not_daily', action="store_true", help="if test function_xpu*_daily case or not.")
    parser.add_argument('-nw', '--not_weekly', action="store_true", help="if test function_xpu*_weekly case or not.")
    parser.add_argument('-sc', '--skip_case', type=str, default="", help="only test endswith function_xpu* case.")
    return parser.parse_args()

# get test list
def list_tests(api_path, args):

    # get tests list from gtest
    list_cmd = f'{api_path}/output/test/test_refactor --gtest_list_tests'
    list_str = subprocess.getoutput(list_cmd)
    list_str_l = list_str.splitlines() 

    # get selected cases string
    sp_case_strs = list()
    if args.cases:
        if os.path.exists(args.cases.strip()):
            with open(args.cases.strip(), 'r') as f:
                for test in f.readlines():
                    sp_case_strs.append(test.strip())                
        else:
            sp_case_strs = args.cases.strip().split(",")
    # analyse gtest lists out
    test_dict = dict()
    test_sum = 0
    for test_str in list_str_l:
        if test_str.startswith("XPU") or test_str.startswith("[WARN][XPURT][xpu_set_device:177]"):
            continue
        if test_str.startswith("[") or test_str.startswith("Warning:") or test_str.startswith("In file"):
            continue
        if args.not_daily and test_str.endswith("daily"):
            continue
        if args.not_weekly and test_str.endswith("weekly"):
            continue
        test_str = test_str.strip()
        if not test_str:
            continue        
        if test_str.endswith('.'): 
            test_type = test_str.strip()
        else:
            # judge if in selected cases
            if sp_case_strs:
                selected = False
                # need select
                for sp_case_str in sp_case_strs:
                    if sp_case_str in f"{test_type}{test_str.strip()}":
                        selected = True
                        break
            else:
                selected = True

            if selected:
                # add test into test_dict
                if not args.xpu_type:
                    # no sepecific xpu
                    if test_type not in test_dict.keys():
                        test_dict[test_type] = list()
                    test_dict[test_type].append(test_str.strip())
                    test_sum = test_sum + 1
                else:
                    # find xpu type tests
                    if args.xpu_type in test_str:
                        if test_type not in test_dict.keys():
                            test_dict[test_type] = list()                    
                        test_dict[test_type].append(test_str.strip())
                        test_sum = test_sum + 1
    #filter skip case
    if args.skip_case:
        skip_case_list = list(filter(None, args.skip_case.split(";")))
        for sc in skip_case_list:
            casename = sc.split(".")[0] + "."
            suffixes = sc.split(".")[1]
            for case_item in list(test_dict.keys()):
                if case_item in casename and suffixes in test_dict[case_item]:
                    print("skip case : " + sc)
                    if len(test_dict[case_item]) == 1:
                        del test_dict[case_item]
                    else:
                        test_dict[case_item].remove(suffixes)

    return [test_dict, test_sum]

# print test_dict
def show_test(test_dict):
    for test_type, test_list in test_dict.items():
            print(test_type + " : " + ",".join(test_list))

def process_bar_setup( args, total_num ):
    # progress bar configurations
    progress = Progress(
        TextColumn("[bold blue]{task.fields[name]}"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "case_sum:",
        TextColumn("[bold red]{task.total}"),
        "elapsed:",
        TimeElapsedColumn(),
        "remaining:",
        TimeRemainingColumn()
    )

    progress.start()
    task_name = "run_test"
    if args.classify:
        task_name = "classify"
    elif args.time_test:
        task_name = "time_test"
    elif args.perf_test:
        task_name = "perf_test"

    task_id = progress.add_task(task_name, name = task_name, total = total_num, start=True) 

    return [progress, task_id]

# use to update progress bar when a process finished
def runner_callback(progress, task_id, completed):
    progress.update( task_id, completed = completed )

def analyze_perf(success_tests, args):
    print("analyse perf results...")
    data = list()
    func_num = 0
    perf_data = dict()
    for test in success_tests:
        print(".", end="")
        data_test = [test]
        perf_data[test] = dict()

        with open(f"perf_{args.xpu_type}/{test}/perf.text", "r") as file:
            
            # get sample number in perf this test
            samples = int(file.readline().split(" ")[1])
            data_test.append(samples)
            perf_data[test]["samples"] = samples

            # get first 80% function
            while True:
                # parse the line
                perf_strs = file.readline().strip().split()
                samples_func = int(perf_strs[0])
                percent_func_str = perf_strs[1]
                percent_func = float(perf_strs[1].strip("%"))/100
                percent_func_cum = float(perf_strs[2].strip("%"))/100
                func = perf_strs[5]

                data_test.append(f"{func}({percent_func_str})")
                perf_data[test][func] = samples_func

                # when find function occupy 80%, stop reading
                if percent_func_cum > 0.8:
                    break
            
            if func_num < (len(data_test) - 2):
                func_num = len(data_test) - 2
            data.append(data_test)

    # prepare column index
    columns = ["test", "samples"]
    for i in range(func_num):
        columns.append(str(i))

    # create seperated test data
    df = pd.DataFrame(data, columns=columns)
    df.to_excel("seperated-perf.xlsx", sheet_name="Sheet1", index=False)
    print()
    print("saved seperated-perf.xlsx")

    # get perf data together of all cases by function
    all_func_dict = dict()
    all_samples = 0
    for test, data_dict in perf_data.items():
        print(".", end="")
        all_samples = all_samples + data_dict["samples"]
        for func, sample_num in data_dict.items():
            if func == "samples":
                continue
            if func not in all_func_dict.keys():
                all_func_dict[func] = sample_num
            else:
                all_func_dict[func] = all_func_dict[func] + sample_num

    # transfer sample number to percent
    all_func_list = list()
    for func, sample_num in all_func_dict.items():
        all_func_list.append([func, sample_num / all_samples * 100])

    df2 = pd.DataFrame(all_func_list, columns=["function", "percentage(%)"])
    df2.sort_values(by="percentage(%)", ascending=False, inplace=True)
    df2.to_excel("all-perf.xlsx", sheet_name="Sheet1", index=False)
    print()
    print("saved all-perf.xlsx")

def run_tests(test_dict, args, api_path, test_sum):
    if not args.xpu_type:
        print("run tests need know xpu type!!!")
        return 

    # set environment variable
    with open(f"{api_path}/script/env/sim_kl{args.xpu_type[-1]}.sh", "r") as file:
        for env_str in file.readlines():
            if "LD_LIBRARY_PATH" in env_str:
                continue
            elif env_str.startswith("#"):
                continue
            else:
                env_str = env_str.strip().split(" ")[1]
                env_var = env_str.split("=")[0]
                env_val = env_str.split("=")[1]
                os.environ[env_var] = env_val
    os.environ["LD_LIBRARY_PATH"] = os.path.abspath(args.runtime_path)
    os.environ["LD_LIBRARY_PATH"] = os.getenv("LD_LIBRARY_PATH") + ":" + os.path.abspath(args.simulator_path)
    
    if args.classify:
        # will disable cluster and cdnn, just print "XPUSIM_TEST_CLASSIFY: xxxx"
        os.environ["XPUSIM_TEST_CLASSIFY"] = "10"
        os.environ["XPUSIM_SKIP_RUN"] = "1"
    
    if args.time_test:
        # to print cluster or cdnn in time.xlsx
        os.environ["XPUSIM_TEST_CLASSIFY"] = "10"

    # just for debug
    # print(os.getenv("XPU_SIMULATOR_MODE"))
    # print(os.getenv("XPUSIM_DEVICE_MODEL"))

    if not os.path.exists(f"perf_{args.xpu_type}"):
        os.makedirs(f"perf_{args.xpu_type}")

    [progress, task_id] = process_bar_setup(args, test_sum)

    sync_variable()

    res_dict = dict()
    # use a process pool to run test
    with mp.Pool(processes=args.thread_num) as pool:
        if not args.together:
            for test_type, test_list in test_dict.items():
                for test_case in test_list:
                    # use a process for one test
                    filter_str = test_type + test_case
                    res_dict[filter_str] = pool.apply_async(run_test, (filter_str, args, api_path),
                    callback=lambda _: runner_callback( progress, task_id, finished.value))
        else:
            filter_str = args.gtest_filter
            res_dict[filter_str] = pool.apply_async(run_test, (filter_str, args, api_path),
                    callback=lambda _: runner_callback( progress, task_id, finished.value))

        # wait test finished
        success_tests = list()
        failed_tests = list()
        time_dict = dict()
        cluster_list = list()
        sdnn_list = list()
        cluster_sdnn_list = list()
        other_list = list()
        if not os.path.exists("log"):
            os.makedirs("log")
        with open("log/run.log", "wt") as file:
            # wait test finished and get return code
            for test, res in res_dict.items():
                [ret, last_time] = res.get()
                # print(test + " : " + str(ret))

                # print result into log/run.log
                if ret == 0:
                    print(f"PASS-{test}", file=file)
                    success_tests.append(test)
                else:
                    failed_tests.append(test)
                    last_time = 0
                    if ret == -3301:
                        print(f"FAIL_TIMEOUT-{test}", file=file)
                    elif ret == -3302:
                        print(f"FAIL_PROFTEXT-{test}", file=file)
                    elif ret == -3303:
                        print(f"FAIL_PROFSVG-{test}", file=file)
                    else:
                        print(f"FAIL-{test}", file=file)

                if args.classify or args.time_test:
                    # try to find XPUSIM_TEST_CLASSIFY and classify tests
                    with open(f"perf_{args.xpu_type}/{test}/run.log", "r") as f:
                        is_cluster = False
                        is_sdnn = False
                        for r_str in f.readlines():
                            if "XPUSIM_TEST_CLASSIFY: cluster" in r_str.strip():
                                is_cluster = True
                            if "XPUSIM_TEST_CLASSIFY: sdnn" in r_str.strip():
                                is_sdnn = True
                        if is_cluster and is_sdnn:
                            cluster_sdnn_list.append(test)
                            time_dict[test] = [last_time, "cluster_sdnn"]
                        elif is_cluster and not is_sdnn:
                            cluster_list.append(test)
                            time_dict[test] = [last_time, "cluster"]
                        elif is_sdnn and not is_cluster:
                            sdnn_list.append(test)
                            time_dict[test] = [last_time, "sdnn"]
                        else:
                            other_list.append(test)
                            time_dict[test] = [last_time, "others"]
    
    progress.stop()
        
    if args.classify:
        # print classify tests into log/classify.log
        with open("log/classify.log", "wt") as file:
            print("cluster cases:", file=file)
            for case in cluster_list:
                print("  " + case, file=file)
            print("sdnn cases:", file=file)
            for case in sdnn_list:
                print("  " + case, file=file)
            print("cluster_sdnn cases:", file=file)
            for case in cluster_sdnn_list:
                print("  " + case, file=file)
            print("other cases:", file=file)
            for case in other_list:
                print("  " + case, file=file)  
        print("saved classify result in log/classify.log")                                                 

    with open("log/pass.log", "wt") as file:
        for test in success_tests:
            print(test, file=file)
    with open("log/failed.log", "wt") as file:
        for test in failed_tests:
            print(test, file=file)        

    if test_sum - len(success_tests) == 0:
        print(f"{len(success_tests)} cases ran successfully, more info in log/run.log ")
    else:
        print(f"{len(success_tests)} cases ran successfully, {test_sum - len(success_tests)} cases failed, more info in log/run.log ")
    
    if args.time_test:
        print("saving time data...")
        time_data = list()
        time_sum = 0
        for test, test_info in time_dict.items():
            time_data.append([test, test_info[0], test_info[1]])
            time_sum += test_info[0]
        time_df = pd.DataFrame(time_data, columns=["test", "last_time/s", "use cluster or sdnn"])
        time_df.sort_values(by="last_time/s", ascending=False, inplace=True)
        time_df.loc[len(time_df.index)] = ["sum", time_sum, ""]
        time_df.to_excel("time.xlsx", sheet_name="Sheet1", index=False)
        print("saved time.xlsx")

    if args.perf_test:
        analyze_perf(success_tests, args)
        
def main():

    args = arg_parser()

    api_path = args.api_path

    if not os.path.exists(api_path) :
        print(f"{api_path} does not exist, please check!")
        return -1

    if not os.path.exists(f"{api_path}/output/test/test_refactor") :
        print(f"{api_path}/output/test/test_refactor does not exist, please check!")
        return -1

    if not args.together:
        [test_dict, test_sum] = list_tests(api_path, args)

        if args.show_test:
            show_test(test_dict)
    else:
        # use gtest_filter to run tests in one thread
        test_dict = dict()
        test_sum = 1
        if not args.gtest_filter:
            args.gtest_filter = "*" + args.xpu_type
        print(f"use {args.gtest_filter} to filter gtest cases in one thread.")

    if args.run_test or args.time_test or args.perf_test or args.classify or args.together:

        if not os.path.exists(f"{api_path}/script/env/sim_kl{args.xpu_type[-1]}.sh") :
            print(f"{api_path}/script/env/sim_kl{args.xpu_type[-1]}.sh does not exist, please check!")
            return -1   

        if not os.path.exists(args.runtime_path) :
            print(f"{args.runtime_path} does not exist, please check!")
            return -1

        if not os.path.exists(args.simulator_path) :
            print(f"{args.simulator_path} does not exist, please check!")
            return -1                       

        run_tests(test_dict, args, api_path, test_sum)

if __name__ == "__main__":
    main()












