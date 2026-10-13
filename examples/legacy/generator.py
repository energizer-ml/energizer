import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import energizer.maths as maths


def main():
    parser = argparse.ArgumentParser(prog="energizer_generator")
    parser.add_argument(
        "config_file_i",
        help="Configuration file containing description of a neural network we want to generate.",
    )
    parser.add_argument(
        "nb_i",
        type=int,
        help="Number of neural networks to generate based on the configuration file.",
    )
    args = parser.parse_args()

    大鸡巴 = args.nb_i
    大阴道 = args.config_file_i
    print(maths.mean(大鸡巴))


if __name__ == "__main__":
    main()
