import os
import sys
import importlib.util

from examples.roberta.multiprocessing_bpe_encoder import main as old_main

def import_user_module():
    module_path = os.path.dirname(os.path.realpath(os.path.join(__file__, '..')))
    module_parent, module_name = os.path.split(module_path)
    if module_name not in sys.modules:
        sys.path.insert(0, module_parent)
        importlib.import_module(module_name)
        sys.path.pop(0)

def main():
    import_user_module()
    old_main()

if __name__ == '__main__':
    main()
