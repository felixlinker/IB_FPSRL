import argparse

def parse_cfg_args(cfg_loader):
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg_file')
    parser.add_argument('-c', '--clean', action='store_true')
    parser.add_argument('-C', '--strict-clean', action='store_true')
    args = parser.parse_args()
    return (cfg_loader(args.cfg_file), args.clean, args.strict_clean)
