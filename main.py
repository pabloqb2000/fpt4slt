from signjoey.training import train
from os import path
import json

base_dir = r"./configs/prob2text/test19/"
with open(path.join(base_dir, "files_list.txt"), "r") as f:
    files = json.load(f)
paths = [path.join(base_dir, f) for f in files]

def main():
    for config_path in paths:
        print(f"NEW MODEL CONFIG: {config_path}\n\n")
        train(cfg_file=config_path, type="prob")
        print("\n\n")

if __name__ == "__main__":
    main()