
import unittest
import argparse

class TestSomething(unittest.TestCase):

    def test_something(self):

        print("This is where you test something")

        self.assertTrue(True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-v", "--verbosity", default=0, \
        help="Verbosity, options are 0 (quiet, default), 1 (timid), and 2 (noisy)")

    args = parser.parse_args()

    unittest.main(verbosity=args.verbosity)
