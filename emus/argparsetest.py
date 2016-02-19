import argparse
parser = argparse.ArgumentParser()
parser.add_argument("x", type=float, help="the base")
parser.add_argument("y", type=float, help="the exponent")
parser.add_argument("-v", "--verbosity", action="count", default=0)
args = parser.parse_args()
answer = args.x**args.y
if args.verbosity >= 2:
    print "Running '{}'".format(__file__)
if args.verbosity >= 1:
    print "{}^{} ==".format(args.x, args.y),
print answer
