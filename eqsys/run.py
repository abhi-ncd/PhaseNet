"""Command-line entry point.

    python -m eqsys.run --config configs/ridgecrest.yaml --stage all
    python -m eqsys.run --config configs/ridgecrest.yaml --stage associate locate
    python -m eqsys.run --config configs/ridgecrest.yaml --stage forecast
"""

import argparse
import json
import logging
import time

from .config import load_config, path

STAGES = ["download", "pick", "associate", "locate", "magnitude", "catalog", "forecast"]


def run_stage(name, config):
    if name == "download":
        from . import download as mod
    elif name == "pick":
        from . import pick as mod
    elif name == "associate":
        from . import associate as mod
    elif name == "locate":
        from . import locate as mod
    elif name == "magnitude":
        from . import magnitude as mod
    elif name == "catalog":
        from . import catalog as mod
    elif name == "forecast":
        from .forecast import evaluate as mod
    else:
        raise ValueError(name)
    return mod.run(config)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", nargs="+", default=["all"], choices=STAGES + ["all", "catalog-all"],
                        help="'all' runs every stage; 'catalog-all' runs download..catalog without forecasting")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        level=logging.DEBUG if args.verbose else logging.INFO)
    config = load_config(args.config)
    with open(path(config, "config_resolved.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)

    if "all" in args.stage:
        stages = STAGES
    elif "catalog-all" in args.stage:
        stages = STAGES[:-1]
    else:
        stages = [s for s in STAGES if s in args.stage]
    for name in stages:
        t0 = time.time()
        logging.info(f"===== stage {name} =====")
        run_stage(name, config)
        logging.info(f"===== stage {name} done in {time.time() - t0:.0f} s =====")


if __name__ == "__main__":
    main()
