"""Make offer decisions based on customer behavior."""

import sys
import os
import datetime
import argparse
import logging
import json

def get_args():
    """Build arg parser and get command line arguments

    :return: parsed args namespace
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--arg-name",   default="default_value", help="Default help")

    args = parser.parse_args()

    return args


def main(args):
    """Run the world simulator.

    :param args: command line arguments from argparse
    :return: exit status
    """

    return 0


if __name__ == '__main__':
    logging.basicConfig(filename='world.log',
                        filemode='w',
                        format='%(asctime)s %(filename)s:%(funcName)s:%(levelname)s:%(message)s',
                        level=logging.INFO)
    logging.info('started')

    try:
        args = get_args()
    except:
        logging.error('failed to parse command line arguments')
        logging.error('failure')
        logging.info('finished')
        raise

    try:
        main(args)
        logging.info('success')
        logging.info('finished')
    except:
        logging.error('failure')
        logging.info('finished')
        raise

