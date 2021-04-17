import unittest
import argparse

from test.test_reset import TestReset

if __name__ == "__main__":
    """
        __
      <(o )__
       (._> /

        DUCK
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-v", "--verbosity", default=0, \
        help="Verbosity, options are 0 (quiet, default), 1 (timid), and 2 (noisy)")

    args = parser.parse_args()

    unittest.main(verbosity=args.verbosity)
