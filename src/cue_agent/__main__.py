"""Run a synthetic demonstration of CUE's abstention behavior."""

import json

from .selector import CUEv0


def main():
    pairs = [{"pair_id": f"demo-{index}", "features": {}, "delta": 1.0} for index in range(6)]
    result = CUEv0().fit(pairs).decide({})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()