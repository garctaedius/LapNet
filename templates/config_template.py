from pyscf import gto
from lapnet import base_config

atom = {atom_str}
num_nodes = {num_nodes}
batch_size = {batch_size}
restore_path = {restore_path}

def get_config():
    mol = gto.Mole()
    mol.build(
    atom = atom, basis = 'sto-3g', unit='bohr')

    cfg = base_config.default()
    cfg.system.pyscf_mol = mol

    # Set run directory
    if restore_path:
        cfg.log.restore_path = restore_path
        cfg.log.restore_path_full = restore_path + "/" 
    cfg.log.save_path = "./results"

    # Set Parameters
    cfg.pretrain.iterations = 5000

    cfg.batch_size = batch_size

    cfg.mcmc.steps = 30

    cfg.optim.iterations = 50000
    cfg.optim.forward_laplacian = True

    cfg.network.determinants = 16

    return cfg

config_file = "full_config.txt"
with open(config_file, "w") as f:
    f.write(str(get_config()))
