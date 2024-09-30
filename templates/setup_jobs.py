import json
import subprocess
import os
import sys

hashes_and_nodes = [("437c5bbe5cef427177a8d4d7dabfbcd0", 1)]#[("156ee3365b98696abc280b34aef5d7ca", 2)]#, ("5fa4716c841d64a69dc9fe7c2f2a0946", 3), ("8074d3273ffaa22501a685b061194716", 3), ("c3568e3dd81be3cfdc615d5dc420eb9d", 2)]
batch_size = 4096
model = "lapnet"

for hash, num_nodes in hashes_and_nodes:
    geom_file = "/leonardo/home/userexternal/garctaed/develop/sparse_wf/data/geometries.json"
    with open(geom_file) as inp:
        geometries_by_hash = json.load(inp)
    geom = geometries_by_hash[hash]

    run_dir = "/leonardo_work/L-AUT_005/garctaed/runs/"+model+"/auto_restart/" + geom["comment"] + "_restart_test"
    if os.path.exists(run_dir):
        print(f"Skipping existing run {run_dir}")
        continue
    os.makedirs(run_dir, exist_ok=False)

    atom = "; ".join([f"{charge} {x} {y} {z}" for charge, (x, y, z) in zip(geom["Z"], geom["R"])])
    atom = "'" + atom + "'"
    with open("config_template.py", "r") as f:
        config_string = f.read()
    config_values = dict(atom_str=atom, num_nodes=num_nodes, batch_size=batch_size, restore_path="'" + "'")
    config_string = eval('f"""' + config_string + '"""', None, config_values)
    with open(run_dir + "/config.py", "w") as f:
        f.write(config_string)

    with open("job_template.sh", "r") as f:
        job_string = f.read()
    job_values = dict(job_name=model+"_" + geom["name"], output_file=run_dir+"/stdout.txt", num_nodes=num_nodes)
    job_string = eval('f"""' + job_string + '"""', None, job_values)
    with open(run_dir + "/job.sh", "w") as f:
        f.write(job_string)

    restart_dir = run_dir + "/restart"
    os.makedirs(restart_dir)
    os.makedirs(restart_dir + "/start_checkpoint")

    with open("config_template.py", "r") as f:
        config_string = f.read()
    config_values = dict(atom_str=atom, num_nodes=num_nodes, batch_size=batch_size, restore_path="'" + "./start_checkpoint" + "'")
    config_string = eval('f"""' + config_string + '"""', None, config_values)
    with open(restart_dir + "/config.py", "w") as f:
        f.write(config_string)

    with open("job_template.sh", "r") as f:
        job_string = f.read()
    job_values = dict(job_name=model+"_" + geom["name"] + "_restart", output_file=restart_dir+"/stdout.txt", num_nodes=num_nodes)
    job_string = eval('f"""' + job_string + '"""', None, job_values)
    with open(restart_dir + "/job.sh", "w") as f:
        f.write(job_string)

    #with open("restart_if_dead_template.py", "r") as f:
    #    string = f.read()
    #values = dict(restart_path="'"+restart_dir+"'", output_file="'"+run_dir+"/stdout.txt"+"'")
    #string = eval('f"""' + string + '"""', None, values)
    #with open(run_dir + "/restart_if_dead.py", "w") as f:
    #    f.write(string)

    current_dir = os.getcwd()
    os.chdir(run_dir)
    subprocess.run(["sbatch", "job.sh"])
    os.chdir(current_dir)
