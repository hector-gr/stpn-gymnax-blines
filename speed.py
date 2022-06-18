import jax
import numpy as np
from utils.models import get_model_ready
from utils.helpers import load_config
from utils.benchmark import (
    speed_numpy_random,
    speed_numpy_network,
    speed_gymnax_random,
    speed_gymnax_network,
)


def speed(
    env_name: str,
    num_env_steps: int,
    num_envs: int,
    num_runs: int,
    use_network: bool,
    use_np_env: bool,
):
    # Get the policy and parameters (if not random)
    configs = load_config(f"agents/{env_name}/es.yaml")
    rng = jax.random.PRNGKey(0)

    if use_network:
        model, params = get_model_ready(rng, configs.train_config)

    run_times = []
    for run_id in range(num_runs):
        rng, rng_run = jax.random.split(rng)
        np.random.seed(run_id)
        # Use numpy env + random policy
        if not use_network and use_np_env:
            r_time = speed_numpy_random(env_name, num_env_steps, num_envs)

        # Use numpy env + MLP policy
        elif use_network and use_np_env:
            r_time = speed_numpy_network(
                env_name, num_env_steps, num_envs, rng_run, model, params
            )

        # Use gymnax env + random policy
        elif not use_network and not use_np_env:
            r_time = speed_gymnax_random(
                env_name, num_env_steps, num_envs, rng_run
            )

        # Use gymnax env + MLP policy
        elif use_network and not use_np_env:
            r_time = speed_gymnax_network(
                env_name, num_env_steps, num_envs, rng_run, model, params
            )

        # Store the computed run time
        print(f"Run {run_id + 1} - Env: {env_name} - Done after {r_time}")
        run_times.append(r_time)
    print(run_times)
    return np.mean(run_times), np.std(run_times)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-env",
        "--env_name",
        type=str,
        default="CartPole-v1",
        help="Environment to estimate step speed for.",
    )
    parser.add_argument(
        "-steps",
        "--num_env_steps",
        type=int,
        default=1_000_000,
        help="Environment steps.",
    )
    parser.add_argument(
        "-runs",
        "--num_runs",
        type=int,
        default=10,
        help="Number of random seeds for estimation.",
    )
    parser.add_argument(
        "-n_envs",
        "--num_envs",
        type=int,
        default=1,
        help="Number of parallel workers/envs.",
    )
    parser.add_argument(
        "-network",
        "--use_network",
        action="store_true",
        default=False,
        help="MLP/Random rollout.",
    )
    parser.add_argument(
        "-np",
        "--use_np_env",
        action="store_true",
        default=False,
        help="Numpy rollout.",
    )

    args, _ = parser.parse_known_args()
    mean_rt, std_rt = speed(
        args.env_name,
        args.num_env_steps,
        args.num_envs,
        args.num_runs,
        args.use_network,
        args.use_np_env,
    )
