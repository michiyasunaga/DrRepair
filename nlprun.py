#!/u/nlp/anaconda/main/anaconda3/bin/python

from __future__ import print_function

import argparse
import math
import random
import re
import subprocess
import sys

from datetime import datetime

# slurm constants
SBATCH_COMMAND = "sbatch"
SRUN_COMMAND = "srun"

# load machine info
MACHINE_INFO_PATH = "/u/nlp/machine-info"
ALL_MACHINE_NAMES = open(MACHINE_INFO_PATH+"/slurm_machines.txt", "r").read().split("\n")[:-1]

# function for loading machine info from yaml file
mem_line_regex = re.compile(":memtot: ([0-9]+)")
cores_line_regex = re.compile(":cores: ([0-9]+)")


# load info about machines from yaml
def load_machine_info(machine_yaml_file_path):
    machine_info = {}
    info_lines = open(machine_yaml_file_path, "r").read().split("\n")
    for info_line in info_lines:
        if mem_line_regex.match(info_line):
            machine_info["mem"] = float(mem_line_regex.match(info_line).group(1))/float(1000)
        if cores_line_regex.match(info_line):
            machine_info["cores"] = float(cores_line_regex.match(info_line).group(1))
    return machine_info


# build up machine info dictionary
MACHINE_INFO = {}
for machine_name in ALL_MACHINE_NAMES:
    MACHINE_INFO[machine_name] = load_machine_info(MACHINE_INFO_PATH+"/"+machine_name+".yaml")
# load gpu counts
gpu_count_lines = open(MACHINE_INFO_PATH+"/gpu_counts.yaml", "r").read().split("\n")[1:-1]
for (machine_name, gpu_count) in [gc.split(":") for gc in gpu_count_lines]:
    MACHINE_INFO[machine_name]["gpu_count"] = gpu_count


# function for mapping nlprun args to slurm args
# this is where specific defaults and policies are generally enforced as well
# e.g. you can't specify a specific machine with --nodelist when submitting to jag-hi
def map_nlprun_args_to_slurm_args(cl_args):
    slurm_args = {}
    # translate nlprun settings to slurm settings
    # determine queue from machine type and priority
    queue_to_use = cl_args.queue
    # jag is default if no machine name or queue is specified
    # if no queue is specified but a john machine is requested, set queue appropriately
    if (queue_to_use is None and cl_args.machine_name and cl_args.machine_name[:4] == "john") or queue_to_use == "john":
        queue_to_use = "john"
    else:
        queue_to_use = "jag"
    # map "urgent", "high" or "low" to "jag-urgent", "jag-hi" or "jag-lo"
    if queue_to_use == "jag":
        if cl_args.priority == "urgent":
            queue_to_use += "-urgent"
        elif cl_args.priority == "high":
            queue_to_use += "-hi"
        elif cl_args.priority == "standard":
            queue_to_use += '-standard'
        else:
            queue_to_use += "-lo"
    # set partition value
    slurm_args["partition"] = queue_to_use
    # set gpu count to 0 if requesting a john machine
    # set gpu count to at least 1 if requesting a jagupard machine
    # TO DO: is that an unwise policy ?
    if queue_to_use == "john":
        cl_args.gpu_count = "0"
    elif queue_to_use in ["jag-hi", "jag-urgent", "jag-lo"]:
        cl_args.gpu_count = str(max(int(cl_args.gpu_count), 1))
    # determine gres setting
    gpu_gres_value = "gpu"
    if cl_args.gpu_type is not None:
        gpu_gres_value += (":"+cl_args.gpu_type)
    gpu_gres_value += (":"+cl_args.gpu_count)
    # set gres value
    slurm_args["gres"] = gpu_gres_value
    # block targeting a specific machine if requesting "jag-hi" or "jag-urgent"
    if queue_to_use in ["jag-hi"]:
        cl_args.machine_name = None
    # set machine name
    if cl_args.machine_name is not None:
        slurm_args["nodelist"] = cl_args.machine_name
    # set the exclude list
    if cl_args.exclude is not None:
        slurm_args["exclude"] = cl_args.exclude
    # set cpu and ram
    # handle case where cpu count was not specified by user
    # None means nothing was specified
    # for john queue, just set to 1
    # for jagupards try to pick a reasonable default
    if queue_to_use == "john" and cl_args.cpu_count is None:
        slurm_args["cpus-per-task"] = "1"
    elif queue_to_use in ["jag-hi", "jag-urgent", "jag-standard", "jag-lo"] and cl_args.cpu_count is None:
        if cl_args.machine_name is None:
            # go with 3 cpu cores by default
            slurm_args["cpus-per-task"] = "3"
        else:
            # if a specific machine is requested, allocate percent gpu's requested of resources
            total_cores = MACHINE_INFO[cl_args.machine_name]["cores"]
            machine_gpu_count = MACHINE_INFO[cl_args.machine_name]["gpu_count"]
            percent_of_gpus_requested = float(cl_args.gpu_count)/float(machine_gpu_count)
            slurm_args["cpus-per-task"] = str(int(math.floor(percent_of_gpus_requested * total_cores)))
    else:
        slurm_args["cpus-per-task"] = cl_args.cpu_count
    # set ram
    # handle case where ram was not specified
    if queue_to_use == "john" and cl_args.memory is None:
        slurm_args["mem"] = "8G"
    elif queue_to_use in ["jag-hi", "jag-urgent", "jag-standard", "jag-lo"] and cl_args.memory is None:
        if cl_args.machine_name is None:
            # go with 16G of memory by default
            slurm_args["mem"] = "16G"
        else:
            # if a specific machine is requested, allocate percent gpu's requested of resources
            total_ram = MACHINE_INFO[cl_args.machine_name]["mem"]
            machine_gpu_count = MACHINE_INFO[cl_args.machine_name]["gpu_count"]
            percent_of_gpus_requested = float(cl_args.gpu_count)/float(machine_gpu_count)
            slurm_args["mem"] = str(int(math.floor(percent_of_gpus_requested * total_ram))) + "G"
    else:
        slurm_args["mem"] = cl_args.memory
    # set job name
    slurm_args["job-name"] = cl_args.job_name
    # set output file name
    if cl_args.output is None:
        slurm_args["output"] = cl_args.job_name + ".out"
    else:
        slurm_args["output"] = cl_args.output
    # set time length for job
    slurm_args["time"] = cl_args.time
    # set output files to append so you can add logging info
    slurm_args["open-mode"] = 'append'
    # return the slurm args
    return slurm_args


# function for creating the body of the sbatch script
def create_sbatch_script_body(cl_args):
    # set up commands to run
    job_commands = ["'"+command_to_run+"'" for command_to_run in cl_args.commands]
    launch_command = "run_as_child_processes "+" ".join(job_commands)
    # create sbatch script body
    sbatch_script_body = (
        "\n# activate your desired anaconda environment\n"
        "source activate %s\n"
        "\n"
        "# cd to working directory\n"
        "cd %s\n"
        "\n"
        "# launch commands\n"
        "srun --unbuffered %s"
    )
    sbatch_script_body = sbatch_script_body % (cl_args.anaconda_environment,
                                               cl_args.working_directory,
                                               launch_command)
    return str(sbatch_script_body)


# function that creates entire sbatch script
def create_sbatch_script(cl_args):
    sbatch_options = [(k, "#SBATCH --%s=%s" % (k, v)) for k, v in map_nlprun_args_to_slurm_args(cl_args).items()]
    sbatch_options.sort()
    sbatch_options = [v for (k, v) in sbatch_options]
    sbatch_body = create_sbatch_script_body(cl_args)
    sbatch_script = "#!/bin/bash\n\n" + "\n".join(sbatch_options) + "\n" + sbatch_body
    return sbatch_script


# function that creates srun command
def create_srun_command(cl_args, interactive=False):
    srun_options = ["--%s=%s" % (k, v) for k, v in map_nlprun_args_to_slurm_args(cl_args).items()]
    # filter out output
    srun_options = [srop for srop in srun_options if srop[:8] != "--output"]
    srun_options.sort()
    if interactive:
        srun_options.append('--export=ANACONDA_ENV=%s,ALL' % cl_args.anaconda_environment)
        srun_options.append("--pty %s" % cl_args.shell)
    srun_command = "srun "+(" ".join(srun_options))
    return srun_command


# create a random job id
def create_random_job_id():
    # handle Python 2 vs. Python 3
    if sys.version_info[0] < 3:
        return subprocess.check_output("whoami")[:-1] + "-job-" + str(random.randint(0, 5000000))
    else:
        return str(subprocess.check_output("whoami")[:-1], encoding="utf8") + "-job-" + str(random.randint(0, 5000000))


if __name__ == '__main__':
    # parse arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--anaconda-environment', help='anaconda environment to start up | default: py-3.6.8',
                        default='py-3.6.8')
    parser.add_argument('-c', '--cpu-count', help='number of cpus to claim | default: None', default=None)
    parser.add_argument('-d', '--gpu-type', choices=['k40', 'titanx', 'titanxp', 'titanv'],
                        help='type of gpu: titanv, titanxp, titanx, or k40 | default: None', default=None)
    parser.add_argument('-g', '--gpu-count', help='number of gpus to claim | default: 1', default='1')
    parser.add_argument('-m', '--machine-name', choices=ALL_MACHINE_NAMES,
                        help='name of machine to use | default: None', default=None)
    parser.add_argument('-n', '--job-name', help='name of job | default: username-job-random_id',
                        default=create_random_job_id())
    parser.add_argument('-o', '--output', help='path to write output of slurm job | default: None', default=None)
    parser.add_argument('-p', '--priority', choices=['urgent', 'high', 'standard', 'low'],
                        help='priority of job: urgent, high or low | default: low', default='standard')
    parser.add_argument('-q', '--queue', choices=['jag', 'john'],
                        help='which machine type to use: jag or john | default: jag', default=None)
    parser.add_argument('-r', '--memory', help='amount of memory to request | default: 8G', default=None)
    parser.add_argument('-t', '--time',
                        help='max job run time ; specify minutes or days-hours | default: 10-0 (10 days, 0 hours)',
                        default='10-0')
    parser.add_argument('-w', '--working-directory', help='working directory | default: current directory', default='.')
    parser.add_argument('-x', '--exclude', help='comma separated list of machines to not request', default=None)
    parser.add_argument('-s', '--shell', default='bash', help='shell to use (only applies to interactive sessions) | default: bash')
    parser.add_argument('-H', '--hold', action='store_true', help='Hold job in queue | default: False')
    parser.add_argument('commands', nargs='*')
    args = parser.parse_args()

    # create sbatch script or start interactive session based on args
    if len(args.commands) > 0 and args.commands[-1] == "test":
        args.commands = args.commands[:-1]
        print("###############################")
        print("test mode")
        print("")
        srun_command_for_job = create_srun_command(args)
        print("starting interactive session with this command:")
        print("")
        print(srun_command_for_job)
        print("")
        sbatch_script_for_job = create_sbatch_script(args)
        print("created following sbatch script: ")
        print("")
        print("###############################")
        print(sbatch_script_for_job)
        print("")
        print("###############################")
    elif args.commands == [] or args.commands == ["interactive"]:
        srun_command_for_job = create_srun_command(args, interactive=True)
        print("starting interactive session with this command:")
        print("")
        print(srun_command_for_job)
        subprocess.call(srun_command_for_job, shell=True)
    else:
        # create sbatch script and submit if not interactive
        sbatch_script_for_job = create_sbatch_script(args)
        sbatch_script_pre_message = "created following sbatch script: \n\n###############################\n\n"
        sbatch_script_post_message = "\n\n###############################\n\nsubmission to slurm complete!\n\n"
        print(sbatch_script_pre_message)
        print(sbatch_script_for_job)
        output_file_path = map_nlprun_args_to_slurm_args(args)['output']
        log_file = open(output_file_path, "w")
        log_file.write('slurm submission log: '+str(datetime.now()) + '\n')
        log_file.write(sbatch_script_pre_message+sbatch_script_for_job+sbatch_script_post_message)
        log_file.close()
        p = subprocess.Popen([SBATCH_COMMAND, "--hold" if args.hold else ""], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        slurm_sub_out, slurm_sub_err = p.communicate(input=str.encode(sbatch_script_for_job))
        slurm_submission_message = \
            '\n###############################\nslurm submission output\n\n' + str(slurm_sub_out, encoding='utf8') + '\n\n' + \
            str(slurm_sub_err, encoding='utf8') + '\n###############################\n\n'
        log_file = open(output_file_path, "a")
        log_file.write(slurm_submission_message)
        log_file.close()
        print(sbatch_script_post_message)
        print(slurm_submission_message)
