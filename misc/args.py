import argparse

def parse_cfg_args(cfg_loader, add_clean_option = True):
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg_file')
    if add_clean_option:
        parser.add_argument('-c', '--clean', action='store_true')
        parser.add_argument('-C', '--strict-clean', action='store_true')
    args = parser.parse_args()
    options = (cfg_loader(args.cfg_file),)
    if add_clean_option:
        options += (args.clean, args.strict_clean)
    return options
