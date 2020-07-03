import sys, os, shutil, re, argparse, json, subprocess

# errors and passing flags
class err():
    no_err = 0
    compile_err = 1
    runtime_err = 2
    mismatch_err = 3

class pass_test():
    none = 0
    public = 1
    both = 2


#################################################
# Compilation (simple ver)

def compile_code(ARGS, code, probid, subid, compile_only=False):
    """
    Write the code to [probid]-[subid].cpp and compile it.
    Return None if the compilation succeeds.
    Otherwise, return the compiler message as a string.
    """
    unique_id = probid + "-" + subid
    if probid.startswith("prob"): #deepfix
        with open(unique_id + ".c", "w") as src_file:
            src_file.write(code)
        command = "gcc -w -std=c99 -pedantic %s.c -lm -o %s" %(unique_id, unique_id)
    else:
        with open(unique_id + ".cpp", "w") as src_file:
            src_file.write(code)
        if not compile_only:
            command = "timeout {} g++ {}.cpp -o {}".format(ARGS.gcc_timeout+1, unique_id, unique_id) ##-std=gnu++11
        else:
            command = "timeout {} g++ {}.cpp -c".format(ARGS.gcc_timeout+1, unique_id)

    try:
        process = subprocess.run(command, shell=True, timeout=ARGS.gcc_timeout, stderr=subprocess.PIPE)
    except subprocess.TimeoutExpired:
        return "g++ timeout!"

    if process.returncode == 0:
        return None
    else:
        return process.stderr.decode('utf8', 'backslashreplace')


def cleanup_simple(objfile):
    if os.path.exists(objfile):
        os.remove(objfile)


def compile_check(ARGS, code, probid, subid, iter_count):
    """
    Compile the code
    Return (pass_test code, err code, extra_info).
    """
    unique_id = probid + "-" + subid
    # generate c++
    compile_errors = compile_code(ARGS, code, probid, subid)
    cleanup_simple(unique_id)
    if compile_errors is not None:
        return pass_test.none, err.compile_err, compile_errors
    return pass_test.none, err.no_err, ""




#################################################
# Compilation & Testcases (used for SPoC evaluation)

BIG_FILE_THRESHOLD = 1000000

def compare_files(outfile, prgfile):
    # Don't read big files
    if (
        os.path.getsize(prgfile) > BIG_FILE_THRESHOLD
        and os.path.getsize(outfile) < BIG_FILE_THRESHOLD
    ):
        return False
    with open(outfile, 'br') as fin:
        outdata = fin.read()
    with open(prgfile, 'br') as fin:
        return fin.read() == outdata

def read_outprgfile(prgfile):
    cmd = "cat {}".format(prgfile)
    process = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    prg_text = process.stdout.decode('utf8', 'backslashreplace')
    prg_text = prg_text[:1000]
    return prg_text

def get_diff_text(outfile, prgfile):
    diff_cmd = "diff --text {} {}".format(outfile, prgfile)
    diff_process = subprocess.run(diff_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    diff_text = diff_process.stdout.decode('utf8', 'backslashreplace')
    # diff_text = diff_text.replace('\r\n', ' ')
    diff_text = diff_text[:1000]
    return diff_text


def run_tests_all_popen(ARGS, code, probid, subid, test_name):
    """
    Run the code on test cases.
    Assume that the code is already compiled to [probid]-[subid].

    Return the error code (no_err, runtime_err, or mismatch_err) and extra info.

    Note: Does not clean up the files. Need to run cleanup afterwards.
    """
    unique_id = probid + "-" + subid
    objfile = unique_id

    testcases = "{}/{}/{}_{}.txt".format(
            ARGS.prog_dir, probid, probid, test_name)
    with open(testcases) as f:
        contents = f.readlines()

    error_code = err.no_err
    error_info = None
    num_test = 0
    counter = 0
    pass_all = 1
    results = []
    processes = []

    cur_inpfile = objfile + "_inp{}.txt".format(num_test)
    cur_outfile = objfile + "_out{}.txt".format(num_test)
    cur_prgfile = objfile + "_prg{}.txt".format(num_test)
    cur_input_file = open(cur_inpfile, "w")
    cur_output_file = open(cur_outfile, "w")
    popen_count = 0
    for line in contents:
        if line == "###ENDINPUT###\n":
            counter = 1
            continue
        if line == "###ENDOUTPUT###\n":
            cur_input_file.close()
            cur_output_file.close()
            command = "timeout {} ./{} < {} > {}".format(
                    ARGS.timeout, objfile, cur_inpfile, cur_prgfile)
            process = subprocess.Popen(command, shell=True, stderr=subprocess.PIPE)
            popen_count += 1
            processes.append(process)
            #reset
            error_code = err.no_err
            error_info = None
            num_test += 1
            counter = 0
            cur_inpfile = objfile + "_inp{}.txt".format(num_test)
            cur_outfile = objfile + "_out{}.txt".format(num_test)
            cur_prgfile = objfile + "_prg{}.txt".format(num_test)
            cur_input_file = open(cur_inpfile, "w")
            cur_output_file = open(cur_outfile, "w")
            # if popen_count % 10 == 0: time.sleep(1)
            continue
        if counter == 0:
            cur_input_file.write(line)
        if counter == 1:
            cur_output_file.write(line)

    for (num_test, process) in enumerate(processes):
        process.wait()
        cur_inpfile = objfile + "_inp{}.txt".format(num_test)
        cur_outfile = objfile + "_out{}.txt".format(num_test)
        cur_prgfile = objfile + "_prg{}.txt".format(num_test)
        with open(cur_inpfile, "r") as ff: inp_text = ff.read()
        out_text = read_outprgfile(cur_outfile)
        prg_text = None
        if process.returncode != 0:
            pass_all = 0
            error_code = err.runtime_err
            if process.returncode == 124:
                error_info = 'Timeout {}'
            else:
                outs, errs = process.communicate()
                error_info = errs.decode('utf8', 'backslashreplace')
        elif not compare_files(cur_outfile, cur_prgfile):
            pass_all = 0
            error_code = err.mismatch_err
            # error_info = 'Mismatch {}'.format(num_test)
            if os.path.getsize(cur_prgfile) > BIG_FILE_THRESHOLD:
                prg_text  = "TOO_BIG"
                error_info = "TOO_BIG"
            else:
                prg_text = read_outprgfile(cur_prgfile)
                error_info = get_diff_text(cur_outfile, cur_prgfile)
        else: #passed test case
            prg_text = read_outprgfile(cur_prgfile)

        results.append({"inp_text": inp_text, "out_text": out_text, "prg_text": prg_text, "error_code": error_code, "error_info": error_info})
        error_code = err.no_err
        error_info = None
    cur_input_file.close()
    cur_output_file.close()
    return pass_all, results #error_code, error_info


def cleanup(objfile):
    if os.path.exists(objfile):
        os.remove(objfile)
    subprocess.call("rm %s* 2> /dev/null" % (objfile + '_inp'), shell=True)
    subprocess.call("rm %s* 2> /dev/null" % (objfile + '_out'), shell=True)
    subprocess.call("rm %s* 2> /dev/null" % (objfile + '_prg'), shell=True)


def compile_and_run_tests_all(ARGS, code, probid, subid, iter_count, skip_hidden_test=False):
    """
    Compile the code, run on public and hidden test cases, then clean up.
    Return (pass_test code, err code, extra_info).
    """
    unique_id = probid + "-" + subid
    if ARGS.verbose:
        with open('verbose-{:05d}'.format(iter_count), 'w') as fout:
            fout.write(code)
            fout.write('###############################\n')
    # generate c++
    compile_errors = compile_code(ARGS, code, probid, subid)
    if compile_errors is not None:
        if ARGS.verbose:
            print('{}: Compilation fails!'.format(iter_count))
            with open('verbose-{:05d}'.format(iter_count), 'a') as fout:
                fout.write('\n\n@@@ {} {}\n'.format(pass_test.none, err.compile_err))
                fout.write('\n################ COMPILE ERROR ################\n')
                fout.write(compile_errors)
        cleanup(unique_id)
        return pass_test.none, err.compile_err, compile_errors
    ### run public test cases
    pass_all_public, test_results_public = run_tests_all_popen(ARGS, code, probid, subid, 'testcases_public')
    ### run hidden test cases
    if skip_hidden_test:
        pass_all_hidden = 1; test_results_hidden = None
    else:
        pass_all_hidden, test_results_hidden = run_tests_all_popen(ARGS, code, probid, subid, 'testcases_hidden')

    cleanup(unique_id)
    if pass_all_public == 0:
        status1 = pass_test.none
        status2 = err.mismatch_err
    elif pass_all_public == 1 and pass_all_hidden == 0:
        status1 = pass_test.public
        status2 = err.mismatch_err
    elif pass_all_public == 1 and pass_all_hidden == 1:
        status1 = pass_test.both
        status2 = err.no_err
    return status1, status2, {"test_results_public": test_results_public, "test_results_hidden": test_results_hidden}
