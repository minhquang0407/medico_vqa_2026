import argparse
import subprocess
import os


report = '''\n⚠️⚠️⚠️\n
Try installing latest version of the library by running the following command:
    pip install -u medvqa
If it still cannot solve the problem, don't hesitate to add an issue at https://github.com/SushantGautam/MedVQA/issues with the log above! We will try to solve the problem ASAP. Can also interact with us on Discord: https://discord.gg/22V9huwc3R.\n
⚠️⚠️⚠️'''


def validate(args, unk_args, submit=False, challenge_evaluate=False, full_evaluate=False):
    # Dynamically find the base directory of the MedVQA library
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Check if competition directory exists
    competition_dir = os.path.join(base_dir, 'competitions', args.competition)
    if not os.path.isdir(competition_dir):
        raise FileNotFoundError(
            f"Competition '{args.competition}' does not exist! Need to update library?"+report)
    # Check if task file exists
    task_file = os.path.join(competition_dir, f'task_{args.task}.py')
    if not os.path.isfile(task_file):
        raise FileNotFoundError(
            f"Task '{args.task}' does not exist! Need to update library?"+report)
    if submit:
        subprocess.run(['python', task_file] + unk_args,
                       env={**os.environ, "_MEDVQA_SUBMIT_FLAG_": "TRUE"})
    elif full_evaluate:
        subprocess.run(['python', task_file] + unk_args,
                       env={**os.environ, "_MEDVQA_FULL_EVALUATE_FLAG_": "TRUE"})
    elif challenge_evaluate:
        subprocess.run(['python', task_file] + unk_args,
                       env={**os.environ, "_MEDVQA_CHALLENGE_EVALUATE_FLAG_": "TRUE"})
    else:
        subprocess.run(
            ['python', task_file] + unk_args)


def main():
    parser = argparse.ArgumentParser(description='MedVQA CLI')
    subparsers = parser.add_subparsers(
        dest='command', required=True, help="Either 'validate', 'validate_and_submit', 'full_evaluate, or 'challenge_evaluate'")

    for cmd in ['validate', 'validate_and_submit', 'challenge_evaluate', 'full_evaluate']:
        subparser = subparsers.add_parser(cmd)
        subparser.add_argument(
            '--competition', required=True, help='Name of the competition (e.g., gi-2025)')
        subparser.add_argument('--task', required=True,
                               help='Task number (1 or 2)')

    args, _unk_args = parser.parse_known_args()
    if args.command == 'validate':
        validate(args, _unk_args)
    elif args.command == 'validate_and_submit':
        validate(args, _unk_args, submit=True)
    elif args.command == 'challenge_evaluate':
        validate(args, _unk_args, challenge_evaluate=True)
    elif args.command == 'full_evaluate':
        validate(args, _unk_args, full_evaluate=True)


if __name__ == "__main__":
    main()
