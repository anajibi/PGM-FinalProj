import json
from argparse import ArgumentParser
from pathlib import Path
from train import main



def perform_experiment(num_variables, num_edges, data_size, seed, num_iterations):
    print(
        f"Performing experiment with {num_variables} variables and {num_edges} edges, data size: {data_size}, seed: {seed}")
    args = {
        '--num_iterations': num_iterations,
        '--prefill': num_iterations / 100,
        '--output_folder': Path('/kaggle/working/output') / f'{num_variables}_{num_edges}_{data_size}_{seed}',
        '--seed': seed,
        'erdos_renyi_lingauss': True,
        '--num_variables': num_variables,
        '--num_edges': num_edges,
        '--num_samples': data_size,

    }

    args = parse_args(args)

    main(args)


def parse_args(args_dict=None):
    """Parse arguments for DAG-GFlowNet for Structure Learning.

    Parameters
    ----------
    args_dict : dict, optional
        Dictionary of arguments. If None, parses command-line arguments.

    Returns
    -------
    args : Namespace
        Parsed arguments.
    """
    parser = ArgumentParser(description='DAG-GFlowNet for Structure Learning.')

    # Environment
    environment = parser.add_argument_group('Environment')
    environment.add_argument('--num_envs', type=int, default=8,
                             help='Number of parallel environments (default: %(default)s)')
    environment.add_argument('--scorer_kwargs', type=json.loads, default='{}',
                             help='Arguments of the scorer.')
    environment.add_argument('--prior', type=str, default='uniform',
                             choices=['uniform', 'erdos_renyi', 'edge', 'fair'],
                             help='Prior over graphs (default: %(default)s)')
    environment.add_argument('--prior_kwargs', type=json.loads, default='{}',
                             help='Arguments of the prior over graphs.')

    # Optimization
    optimization = parser.add_argument_group('Optimization')
    optimization.add_argument('--lr', type=float, default=1e-5,
                              help='Learning rate (default: %(default)s)')
    optimization.add_argument('--delta', type=float, default=1.0,
                              help='Value of delta for Huber loss (default: %(default)s)')
    optimization.add_argument('--batch_size', type=int, default=32,
                              help='Batch size (default: %(default)s)')
    optimization.add_argument('--num_iterations', type=int, default=100_000,
                              help='Number of iterations (default: %(default)s)')

    # Replay buffer
    replay = parser.add_argument_group('Replay Buffer')
    replay.add_argument('--replay_capacity', type=int, default=100_000,
                        help='Capacity of the replay buffer (default: %(default)s)')
    replay.add_argument('--prefill', type=int, default=1000,
                        help='Number of iterations with a random policy to prefill '
                             'the replay buffer (default: %(default)s)')

    # Exploration
    exploration = parser.add_argument_group('Exploration')
    exploration.add_argument('--min_exploration', type=float, default=0.1,
                             help='Minimum value of epsilon-exploration (default: %(default)s)')
    exploration.add_argument('--update_epsilon_every', type=int, default=10,
                             help='Frequency of update for epsilon (default: %(default)s)')

    # Miscellaneous
    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--num_samples_posterior', type=int, default=1000,
                      help='Number of samples for the posterior estimate (default: %(default)s)')
    misc.add_argument('--update_target_every', type=int, default=1000,
                      help='Frequency of update for the target network (default: %(default)s)')
    misc.add_argument('--seed', type=int, default=0,
                      help='Random seed (default: %(default)s)')
    misc.add_argument('--num_workers', type=int, default=4,
                      help='Number of workers (default: %(default)s)')
    misc.add_argument('--mp_context', type=str, default='spawn',
                      help='Multiprocessing context (default: %(default)s)')
    misc.add_argument('--output_folder', type=Path, default='output',
                      help='Output folder (default: %(default)s)')

    subparsers = parser.add_subparsers(help='Type of graph', dest='graph')

    # Erdos-Renyi Linear-Gaussian graphs
    er_lingauss = subparsers.add_parser('erdos_renyi_lingauss')
    er_lingauss.add_argument('--num_variables', type=int, required=True,
                             help='Number of variables')
    er_lingauss.add_argument('--num_edges', type=int, required=True,
                             help='Average number of edges')
    er_lingauss.add_argument('--num_samples', type=int, required=True,
                             help='Number of samples')

    # Flow cytometry data (Sachs) with observational data
    sachs_continuous = subparsers.add_parser('sachs_continuous')

    # Flow cytometry data (Sachs) with interventional data
    sachs_intervention = subparsers.add_parser('sachs_interventional')

    if args_dict is not None:
        # Convert args_dict to a list of arguments
        args_list = []
        for key, value in args_dict.items():
            if isinstance(value, bool):
                args_list.append(f"{key}")
            else:
                args_list.append(f"{key}")
                args_list.append(str(value))
        return parser.parse_args(args_list)
    else:
        return parser.parse_args()


if __name__ == '__main__':
    for num_variables, config in configs.items():
        if isinstance(config['num_edges'], int):
            for data_size in config['data_size']:
                perform_experiment(num_variables, config['num_edges'], data_size)
        else:
            for index, num_edges in enumerate(config['num_edges']):
                for data_size in config['data_size'][index]:
                    perform_experiment(num_variables, num_edges, data_size)
