#!/usr/bin/env python3

"""
Basic example which iterates through the tasks specified and evaluates the given model
on them.

For more documentation, see parlai.scripts.eval_model.
"""
from parlai.scripts.eval_model import setup_args, eval_model


if __name__ == '__main__':
    parser = setup_args()
    opt = parser.parse_args(print_args=False)
    eval_model(opt, print_parser=parser)
