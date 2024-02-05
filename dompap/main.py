import sys

help_message = """      dompap - Simulations of point-like particles in any dimension with any pair potential

Usage: dompap [options]

Options:
  -h, --help            show this help message and exit
  -t, --test            run a test simulation

Repository: <https://github.com/urpedersen/dompap>
"""


def main(argv):
    verbose = True

    # Print error if unknown options are given
    known_options = ['-h', '--help', '-t', '--test']
    for arg in argv[1:]:
        if arg not in known_options:
            print(f'Unknown option: {arg}')
            print(f'Try:\ndompap --help')
            sys.exit()

    # Print help message
    if len(argv) == 1 or '-h' in argv or '--help' in argv:
        print(help_message)
        sys.exit()

    # Run test simulation
    if '-t' in argv or '--test' in argv:
        from dompap.tools import run_test_simulation
        run_test_simulation(verbose=verbose)
        sys.exit()


if __name__ == '__main__':
    main(sys.argv)
