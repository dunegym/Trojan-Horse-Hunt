#!/usr/bin/env python3

import os
import argparse
import shutil
import yaml
import glob
import multiprocessing as mp
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn
)
import shutil
import subprocess
import filecmp

# argument parser
def arg_parser():
    """
    analyse parameters to get argument
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--env_yaml', type=str, default="", 
        help="run tests with which environment, including simulator and commands.")

    parser.add_argument('-cp', "--case_path", type=str, default="../cases",
        help="search cases in which file path(be folder or file)")

    parser.add_argument('-c', '--cases', type=str, default="", 
        help="select cases which name has these strings, this can be string or file")

    parser.add_argument('-tn', "--thread_num", type=int, default=1,
        help="use how many threads to run cases parallelly")

    parser.add_argument('-to', '--time_out', type=int, default=-1, 
        help="if one simulator run one case time longer than this, stop and fail")


    # parser.add_argument('-st', '--show_test', action="store_true", help='show selected or all tests')  
    
    return parser.parse_args()

def analyse_env(env_yaml):
    """
    analyse env yaml file to get simulator info, run_command, env yaml file path
    """

    # parse env yaml and get dict
    with open(env_yaml, 'r') as f:
        env_dict = yaml.load(f, Loader=yaml.FullLoader)
    # print(env_dict)

    if "simulator" not in env_dict.keys():
        print("no simulator in env yaml, please check!")
        exit(-1)

    simulator_dict = env_dict["simulator"]

    # get run rule to use later
    run_rule_dict = dict()
    for simulator, config in env_dict["simulator"].items():
        if "run_rule" not in config.keys():
            print(f"no run_rule in {simulator} config, please check!")
        else:
            for run_rule in config["run_rule"].keys():
                if run_rule not in run_rule_dict.keys():
                    run_rule_dict[run_rule] = list()

                run_rule_dict[run_rule].append(simulator)

    # get env yaml path
    env_path = os.path.dirname(os.path.abspath(env_yaml))

    return [simulator_dict, run_rule_dict, env_path]

def get_cases(path, run_rule_dict, cases_string):
    """
    analyse case test.yaml file to get test cases to test later,
    the cases' run rule need be found in run_rule_dict from env yaml
    """
    case_list = list()

    # get selected cases string
    sp_case_strs = list()
    if cases_string:
        if os.path.exists(cases_string.strip()):
            with open(cases_string.strip(), 'r') as f:
                for test in f.readlines():
                    sp_case_strs.append(test.strip())                
        else:
            sp_case_strs = cases_string.strip().split(",")

    old_cwd = os.getcwd()
    os.chdir(path)
    # print(os.getcwd())
    # find file whose name ends with test.yaml to get cases
    for file in glob.glob("**/*test.yaml", recursive=True):
        # print(file)  
        with open(f"{file}", 'r') as f:
            tests_dict = yaml.load(f, Loader=yaml.FullLoader)
            # print(tests_dict)

        if tests_dict["type"] == 'machine':
            # we just run cases whose run rule in run_rule_dict from env yaml
            if "run1" in tests_dict["run_rule"]:
                run_flag = True
                for run_no in range(1, len(tests_dict["run_rule"].keys())+1):
                    run_key = "run" + str(run_no)                    
                    if tests_dict["run_rule"][run_key]["type"] not in run_rule_dict.keys():
                        run_flag = False
                        break
            else:
                if tests_dict["run_rule"]["type"] in run_rule_dict.keys():
                    run_flag = True
                else:
                    run_flag = False
                
            if run_flag:
                case_name = os.path.dirname(os.path.abspath(file)).split("/")[-1]
                # judge if this is the selected case
                if sp_case_strs:
                    selected = False
                    for sp_case_str in sp_case_strs:
                        if sp_case_str in case_name:
                            selected = True
                else:
                    selected = True

                if selected:
                    case_list.append(tests_dict)
                    case_list[-1]["path"] = os.path.dirname(os.path.abspath(file))
                    # print(case_list[-1]["path"])
                    if "name" not in case_list[-1].keys():
                        case_list[-1]["name"] = case_name


    os.chdir(old_cwd)  
    return case_list

def progress_sync_variable():
    """
    create global variable to use in different threads
    for progress bar
    """
    globals()["result_condition"] = mp.Condition()
    globals()["finished"] = mp.Value('L', 0)

def process_bar_setup( total_num ):
    """
    create the process bar used to show test cases running progress
    """
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
    task_name = "simulator test"

    task_id = progress.add_task(task_name, name = task_name, total = total_num, start=True) 

    return [progress, task_id]

def runner_callback(progress, task_id, completed):
    """
    use to update progress bar when a process finished
    """
    progress.update( task_id, completed = completed )

def diff_data_file(path_dict, check_dict, dump_dict):
    if not isinstance(check_dict["ref"], dict):
        print("check ref need be a dict")
        return False

    if "ref" not in check_dict.keys():
        print("can't find ref of check")
        return False   

    run_path = path_dict["run_path"]
    log_file = f"{run_path}/{run_path.split('/')[-1]}.log"
    if path_dict["copy_flag"]:
        check_path = run_path
    else:
        check_path = path_dict["case_path"]

    with open(log_file, mode="a+") as f:
        for type, file in check_dict["ref"].items():
            if type not in dump_dict.keys():
                print(f"can't find {type} in dump", file=f)
                return False
            else:
                if isinstance(dump_dict[type], dict):
                    if "file" not in dump_dict[type].keys():
                        print(f"can't find file in {type} of dump", file=f)
                        return False  
                    else:
                        if not filecmp.cmp(f"{run_path}/{dump_dict[type]['file']}",
                                f"{check_path}/{file}"):
                            print(f"{run_path}/{dump_dict[type]['file']}-{check_path}/{file}-check failed", file=f)
                            return False
                        else:
                            print(f"{run_path}/{dump_dict[type]['file']}-{check_path}/{file}-check pass", file=f)
                else:
                    if not filecmp.cmp(f"{run_path}/{dump_dict[type]}",
                            f"{check_path}/{file}"):
                        print(f"{run_path}/{dump_dict[type]}-{check_path}/{file}-check failed", file=f)
                        return False
                    else:
                        print(f"{run_path}/{dump_dict[type]}-{check_path}/{file}-check pass", file=f)
    
    return True

def diff_data_file_bypass_csr0(path_dict, check_dict, dump_dict):
    if not isinstance(check_dict["ref"], dict):
        print("check ref need be a dict")
        return False

    if "ref" not in check_dict.keys():
        print("can't find ref of check")
        return False   

    run_path = path_dict["run_path"]
    log_file = f"{run_path}/{run_path.split('/')[-1]}.log"
    if path_dict["copy_flag"]:
        check_path = run_path
    else:
        check_path = path_dict["case_path"]

    with open(log_file, mode="a+") as f:
        for type, file in check_dict["ref"].items():
            if type not in dump_dict.keys():
                print(f"can't find {type} in dump", file=f)
                return False
            else:
                if isinstance(dump_dict[type], dict):
                    if "file" not in dump_dict[type].keys():
                        print(f"can't find file in {type} of dump", file=f)
                        return False  
                    else:
                        # bypass csr0 for 8csrs
                        if "reg_" in f"{check_path}/{file}":
                            with open(f"{run_path}/{dump_dict[type]['file']}", 'r') as dump_f:
                                with open(f"{check_path}/{file}", 'r') as check_f:
                                    dump_line = dump_f.readline()
                                    csr_flag = False
                                    csr_no = 0
                                    # read all lines in dump file
                                    while dump_line:
                                        # read one line in ref file to compare
                                        check_line = check_f.readline()

                                        # if csr reg, bypass csr0
                                        if csr_flag and len(check_line) == 9: # include \n 
                                            if csr_no == 0:
                                                csr_no += 1
                                                if csr_no == 8:
                                                    csr_no = 0
                                                dump_line = dump_f.readline()
                                                continue
                                            else:
                                                csr_no += 1
                                                if csr_no == 8:
                                                    csr_no = 0                                            

                                        if check_line != dump_line:
                                            print(f"{run_path}/{dump_dict[type]['file']}-{check_path}/{file}-check failed", file=f)
                                            print(f"{dump_line} != {check_line}", file=f)
                                            return False
                                        
                                        # csr reg in end of reg dump file, seperate with scalar reg                                    
                                        if not csr_flag and len(check_line) > 9:
                                            csr_flag = True

                                        dump_line = dump_f.readline()
                                    
                                    # in case ref file has more lines
                                    check_line = check_f.readline()
                                    if check_line:
                                        print(f"{run_path}/{dump_dict[type]['file']}-{check_path}/{file}-check failed", file=f)
                                        print(f"{check_line} in ref file, but dump file end.", file=f)
                                        return False
                                    else:
                                        print(f"{run_path}/{dump_dict[type]['file']}-{check_path}/{file}-check pass", file=f) 

                        else:
                            if not filecmp.cmp(f"{run_path}/{dump_dict[type]['file']}",
                                    f"{check_path}/{file}"):
                                print(f"{run_path}/{dump_dict[type]['file']}-{check_path}/{file}-check failed", file=f)
                                return False
                            else:
                                print(f"{run_path}/{dump_dict[type]['file']}-{check_path}/{file}-check pass", file=f)
                else:
                    # bypass csr0 for 8csrs
                    if "reg_" in f"{check_path}/{file}":
                        with open(f"{run_path}/{dump_dict[type]}", 'r') as dump_f:
                            with open(f"{check_path}/{file}", 'r') as check_f:
                                dump_line = dump_f.readline()
                                csr_flag = False
                                csr_no = 0
                                # read all lines in dump file
                                while dump_line:
                                    # read one line in ref file to compare
                                    check_line = check_f.readline()

                                    # if csr reg, bypass csr0
                                    if csr_flag and len(check_line) == 9:  # include \n 
                                        if csr_no == 0:
                                            csr_no += 1
                                            if csr_no == 8:
                                                csr_no = 0
                                            dump_line = dump_f.readline()
                                            continue
                                        else:
                                            csr_no += 1
                                            if csr_no == 8:
                                                csr_no = 0                                            

                                    if check_line != dump_line:
                                        print(f"{run_path}/{dump_dict[type]}-{check_path}/{file}-check failed", file=f)
                                        print(f"{dump_line} != {check_line}", file=f)                                                                               
                                        return False
                                    
                                    # csr reg in end of reg dump file, seperate with scalar reg                                    
                                    if not csr_flag and len(check_line) > 9:
                                        csr_flag = True

                                    dump_line = dump_f.readline()
                                
                                # in case ref file has more lines
                                check_line = check_f.readline()
                                if check_line:
                                    print(f"{run_path}/{dump_dict[type]}-{check_path}/{file}-check failed", file=f)
                                    print(f"{check_line} in ref file, but dump file end.", file=f)
                                    return False
                                else:
                                    print(f"{run_path}/{dump_dict[type]}-{check_path}/{file}-check pass", file=f)                                

                    else:
                        if not filecmp.cmp(f"{run_path}/{dump_dict[type]}",
                                f"{check_path}/{file}"):
                            print(f"{run_path}/{dump_dict[type]}-{check_path}/{file}-check failed", file=f)
                            return False
                        else:
                            print(f"{run_path}/{dump_dict[type]}-{check_path}/{file}-check pass", file=f)
    
    return True

def run_case(case, args, simulator_dict, env_path):
    """
    thread function to run case
    """

    if args.time_out != -1:
        timeout = args.time_out
    else:
        timeout = None

    if case["type"] == "machine":
        # create log folder
        log_path = "log" + case["path"].split("cases")[-1]
        log_path = os.path.abspath(log_path)
        if not os.path.exists(log_path):
            # shutil.rmtree(log_path)
            os.makedirs(log_path)

        if "run1" not in case["run_rule"].keys():
            case_run_dict = dict()
            case_run_dict["run1"] = case["run_rule"]
            case_check_dict = dict()
            case_check_dict["run1"] = case["check_rule"]            
        else:            
            case_run_dict = case["run_rule"]
            case_check_dict = case["check_rule"]

        # may run many times, so use loop to run many times which 
        # correspond to test.yaml
        for run_no in range(1, len(case_run_dict.keys())+1):
            run_key = "run" + str(run_no)
            run_dict = case_run_dict[run_key]

            # create run command and run 
            for si_name, config in simulator_dict.items():
                if run_dict["type"] not in config["run_rule"].keys():
                    continue

                command = ""

                # put environment variable in front
                if "environ" in config:
                    for name, value in config["environ"].items():
                        if "../"  in str(value) or "./" in str(value):
                            value = env_path + '/' + value
                        command += f" {name}={value} "
                
                # dict of specific run_rule
                command_dict = config["run_rule"][run_dict["type"]]

                if "environ" in command_dict.keys():
                    for name, value in command_dict["environ"].items():
                        if "../"  in str(value) or "./" in str(value):
                            value = env_path + '/' + value
                        command += f" {name}={value} "                

                # get command tool 
                if "../" in command_dict["command"] or "./" in command_dict["command"]:
                    command += env_path + "/" + command_dict["command"]
                else:
                    command += command_dict["command"]

                if run_no == 1:
                    if "copy" in command_dict.keys():
                        copy_flag = command_dict["copy"]
                    else:
                        copy_flag = False

                    if os.path.exists(f"{log_path}/{si_name}"):
                        shutil.rmtree(f"{log_path}/{si_name}")                        
                    if copy_flag:
                        shutil.copytree(case["path"], f"{log_path}/{si_name}")
                    else:
                        os.mkdir(f"{log_path}/{si_name}")  

                # add code option
                if "code" in run_dict.keys():
                    case_code_dict = run_dict["code"]
                    if "code" in command_dict["option"].keys():
                        env_code_dict = command_dict["option"]["code"]
                    else:
                        env_code_dict = ''
                    if isinstance(case_code_dict, dict):
                        for key, value in case_code_dict.items():
                            if isinstance(env_code_dict, dict):
                                if key in env_code_dict.keys():
                                    if key == "file" and not copy_flag:
                                        command += f" {env_code_dict[key]}={case['path']}/{value} "
                                    else:
                                        command += f" {env_code_dict[key]}={value} "
                            else:
                                if env_code_dict != "":
                                    # case and env need correspond
                                    print("case and env need correspond of code")
                                    with result_condition:
                                        finished.value +=1
                                    return -1                            
                    else:
                        if isinstance(env_code_dict, dict):
                            # don't know using which key to set case option
                            print("don't know using which key to set case option of code")
                            with result_condition:
                                finished.value +=1
                            return -2                          
                        else:
                            if env_code_dict != "":
                                if copy_flag:
                                    command += f" {env_code_dict}={case_code_dict} "
                                else:
                                    command += f" {env_code_dict}={case['path']}/{case_code_dict} "

                # add data option
                if "data" in run_dict.keys():
                    case_data_dict = run_dict["data"]
                    if "data" in command_dict["option"].keys():
                        env_data_dict = command_dict["option"]["data"]
                    else:
                        env_data_dict = ''
                    if isinstance(case_data_dict, dict):
                        for key, value in case_data_dict.items():
                            if isinstance(env_data_dict, dict):
                                if key in env_data_dict.keys():

                                    # value may be a dict
                                    if isinstance(value, dict):
                                        for op, va in value.items():
                                            if isinstance(env_data_dict[key], dict):
                                                if  op in env_data_dict[key].keys():
                                                    if "$run_path$" in str(va):
                                                        command += f" {env_data_dict[key][op]}={va} ".replace(
                                                                "$run_path$", f"{log_path}/{si_name}")
                                                    elif op == "file" and not copy_flag:
                                                        command += f" {env_data_dict[key][op]}={case['path']}/{va} "
                                                    else:
                                                        command += f" {env_data_dict[key][op]}={va} "
                                            else:
                                                # case and env need correspond
                                                print(f"case and env need correspond for {key} of data")
                                                with result_condition:
                                                    finished.value +=1
                                                return -3
                                    else:
                                        if isinstance(env_data_dict[key], dict):
                                            # don't know using which key to set case option
                                            print(f"don't know using which key to set case option for {key} of data")
                                            with result_condition:
                                                finished.value +=1
                                            return -4                                          
                                        else:
                                            if copy_flag:
                                                command += f" {env_data_dict[key]}={value} "
                                            elif "$run_path$" in value:
                                                command += f" {env_data_dict[key]}={value} ".replace(
                                                        "$run_path$", f"{log_path}/{si_name}")
                                            else:
                                                command += f" {env_data_dict[key]}={case['path']}/{value} "
                            else:
                                if env_data_dict != "":
                                    # case and env need correspond
                                    print("case and env need correspond of data")
                                    with result_condition:
                                        finished.value +=1
                                    return -5                            
                    else:
                        if isinstance(env_data_dict, dict):
                            # don't know using which key to set case option
                            print("don't know using which key to set case option of data")
                            with result_condition:
                                finished.value +=1
                            return -6                         
                        else:
                            if env_data_dict != "":
                                if copy_flag:
                                    command += f" {env_data_dict}={case_data_dict} " 
                                elif "$run_path$" in case_data_dict:
                                    command += f" {env_data_dict}={case_data_dict} ".replace(
                                            "$run_path$", f"{log_path}/{si_name}")                                    
                                else:
                                    command += f" {env_data_dict}={case['path']}/{case_data_dict} " 

                # add dump option
                if "dump" in run_dict.keys():
                    case_dump_dict = run_dict["dump"]
                    if "dump" in command_dict["option"].keys():
                        env_dump_dict = command_dict["option"]["dump"]
                    else:
                        env_dump_dict = ''
                    if isinstance(case_dump_dict, dict):
                        for key, value in case_dump_dict.items():
                            if isinstance(env_dump_dict, dict):
                                if key in env_dump_dict.keys():

                                    # value may be a dict
                                    if isinstance(value, dict):
                                        for op, va in value.items():
                                            if isinstance(env_dump_dict[key], dict):
                                                if  op in env_dump_dict[key].keys():
                                                    command += f" {env_dump_dict[key][op]}={va} "
                                            else:
                                                # case and env need correspond
                                                print(f"case and env need correspond for {key} of dump")
                                                with result_condition:
                                                    finished.value +=1
                                                return -7
                                    else:
                                        command += f" {env_dump_dict[key]}={value} "
                                    
                            else:
                                if env_dump_dict != "":
                                    # case and env need correspond
                                    print("case and env need correspond of dump")
                                    with result_condition:
                                        finished.value +=1
                                    return -8                            
                    else:
                        if isinstance(env_dump_dict, dict):
                            # don't know using which key to set case option
                            print("don't know using which key to set case option of dump")
                            with result_condition:
                                finished.value +=1
                            return -9                          
                        else:
                            if env_dump_dict != "":
                                command += f" {env_dump_dict}={case_dump_dict} "   

                # add other option
                if "other" in command_dict["option"].keys():
                    if "$case_path$" in command_dict['option']['other']:
                        other_str = command_dict['option']['other'].replace('$case_path$', f"{case['path']}")
                        command += other_str
                    else:
                        command += f" {command_dict['option']['other']} "
                
                # print(command)

                # print command to log_file
                log_file = f"{log_path}/{si_name}/{si_name}.log"
                os.system(f"echo {run_key} >> {log_file}") 
                os.system(f"echo {command} >> {log_file}")  
                
                try:
                    command += f" >> {log_file} 2>&1"
                    #  run test shell command
                    completed_process = subprocess.run(args=command, shell=True, timeout=timeout, cwd=f"{log_path}/{si_name}", stderr=subprocess.STDOUT)

                    if completed_process.returncode != 0:
                        # command failed
                        with result_condition: 
                            finished.value += 1            
                        return completed_process.returncode

                except subprocess.TimeoutExpired:
                    os.system(f"echo timeout error !!! >> {log_file}")
                    with result_condition: 
                        finished.value += 1     
                    return -3301

            # check result
            check_dict = case_check_dict[run_key]
            dump_dict = run_dict["dump"]
            path_dict = dict()
            path_dict["copy_flag"] = copy_flag
            path_dict["run_path"] = f"{log_path}/{si_name}"
            path_dict["case_path"] = case["path"]            
            if not eval(f"{check_dict['type']}(path_dict, check_dict, dump_dict)"):
                with result_condition: 
                    finished.value += 1 
                return -2000

        with result_condition: 
            finished.value += 1                 
        return 0
    else:
        # don't support type
        with result_condition:
            finished.value +=1
        return -1000

def run_cases(case_list, args, simulator_dict, env_path):

    progress_sync_variable()

    test_sum = len(case_list)
    [progress, task_id] = process_bar_setup(test_sum)

    res_dict = dict()
    # use thread pool to run cases in parallel
    with mp.Pool(processes=args.thread_num) as pool:
        if not os.path.exists("log"):
            os.makedirs("log")

        for case in case_list:
            res_dict[case["name"]] = pool.apply_async(run_case, (case, args, simulator_dict, env_path),
                callback=lambda _: runner_callback( progress, task_id, finished.value))

        # wait test finished
        success_tests = list()
        failed_tests = list()  

        with open("log/run.log", "wt") as file:
            # wait test finished and get return code
            for test, res in res_dict.items():
                ret = res.get()
                # print(test + " : " + str(ret))

                # print result into log/run.log
                if ret == 0:
                    print(f"PASS-{test}", file=file)
                    success_tests.append(test)
                else:
                    failed_tests.append(test)
                    if ret == -3301:
                        print(f"FAIL_TIMEOUT-{test}", file=file)
                    elif ret == -1000:
                        print(f"FAIL_DONT_SUPPROT_TEST_TYPE-{test}", file=file)
                    elif ret == -2000:
                        print(f"FAIL_CHECK_FAILED-{test}", file=file)                        
                    else:
                        print(f"FAIL-{test}", file=file)
    
    progress.stop()

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

def main():
    args = arg_parser()

    simulator_dict, run_rule_dict, env_path = analyse_env(args.env_yaml)
    # print(simulator_dict)
    # print(run_rule_dict)

    case_list = get_cases(args.case_path, run_rule_dict, args.cases)
    # print(case_list)

    run_cases(case_list, args, simulator_dict, env_path)

    


if __name__ == '__main__':
    main()