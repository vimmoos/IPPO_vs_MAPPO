from dww.agents.run_exp import run_exp
from dww.agents.mappo.utils import make_mappo_policy, make_mappo_manager
import wandb
from .var import VarRef
from functools import partial

import argparse


def main(args):
    exp_confs = VarRef(args.exp_id).resolve()
    wandb_logger = not args.no_wlogger

    for conf in exp_confs:
        for x in range(args.repetitions):
            try:
                if args.algorithm == "mappo":
                    conf._conf.log_group = conf._conf.log_group.replace("ppo", "mappo")
                    conf._conf.log_tags = [
                        tag.replace("ppo", "mappo") for tag in conf._conf.log_tags
                    ]
                    manager_fn = partial(make_mappo_manager, conf)
                    run_exp(
                        conf,
                        logger=wandb_logger,
                        policy_make_fn=make_mappo_policy,
                        manager_make_fn=manager_fn,
                    )
                else:
                    run_exp(conf, logger=wandb_logger)
            except Exception as e:
                print("ERROR ABORTING RUN, GOING TO NEXT ONE:")
                print(e)
            if wandb_logger:
                wandb.finish()
            conf.seed += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run experiments specified by a module:var configuration."
    )
    parser.add_argument(
        "exp_id",
        help="Module:var specifier for the experiment (e.g., 'dww.agents.experiments:conf_test')"
        "The available options are: dww.agents.experiments:conf_test,dww.agents.experiments:confs"
        "dww.agents.experiments:confs2,dww.agents.experiments:conf_final",
    )
    parser.add_argument(
        "--algorithm",
        help="The algorithm to be used during the experiments",
        choices=["ippo", "mappo"],
    )
    parser.add_argument(
        "--no-wlogger",
        action="store_true",
        help="Disable logging (default: logging enabled)",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
        help="Number of times to repeat each experiment (default: 5)",
    )

    args = parser.parse_args()
    main(args)
