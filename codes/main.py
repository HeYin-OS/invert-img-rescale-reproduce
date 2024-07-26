import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str)
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()


if __name__ == '__main__':
    main()
