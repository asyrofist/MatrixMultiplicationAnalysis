import cProfile
import divide_conquer
import ijk
import strassen
import sys
cProfile.run('{}.main(n={})'.format(sys.argv[1],sys.argv[2]))