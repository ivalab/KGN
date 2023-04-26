import os
import yaml

def recursive_key_value_assign(d,ks,v):
    """
    Recursive value assignment to a nested dict

    Arguments:
        d {dict} -- dict
        ks {list} -- list of hierarchical keys
        v {value} -- value to assign
    """
    
    if len(ks) > 1:
        recursive_key_value_assign(d[ks[0]],ks[1:],v)
    elif len(ks) == 1:
        d[ks[0]] = v
 
def load_config(config_file, arg_configs=[], save=False):
    """
    Loads yaml config file and overwrites parameters with function arguments and --arg_config parameters

    Arguments:
        config_file {str} -- The yaml configuration file path

    Keyword Arguments:
        batch_size {int} -- [description] (default: {None})
        max_epoch {int} -- "epochs" (number of scenes) to train (default: {None})
        data_path {str} -- path to scenes with contact grasp data (default: {None})
        arg_configs {list} -- Overwrite config parameters by hierarchical command line arguments (default: {[]})
        save {bool} -- Save overwritten config file (default: {False})

    Returns:
        [dict] -- Config
    """
    assert os.path.exists(config_file)
    with open(config_file,'r') as f:
        global_config = yaml.safe_load(f)
    for conf in arg_configs:
        k_str, v = conf.split(':')
        try:
            v = eval(v)
        except:
            pass
        ks = [int(k) if k.isdigit() else k for k in k_str.split('.')]
        
        recursive_key_value_assign(global_config, ks, v)
    
    if save:
        with open(config_file,'w') as f:
            yaml.dump(global_config, f)

    return global_config


if __name__=="__main__":
    root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    config_file = os.path.join(root_dir, "configs/ps_grasp.yaml")
    load_config(config_file)
