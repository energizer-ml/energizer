from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import energizer as ez


def main():
    group = ez.init_device_group(distributed=True)
    x = ez.tensor([float(group.rank + 1)], device="gpu")
    total = ez.all_sum(x, group=group)

    print(
        f"rank={group.rank} size={group.size} "
        f"local={x.item()} all_sum={total.item()}"
    )


if __name__ == "__main__":
    main()
