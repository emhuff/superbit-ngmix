import os,sys
import importlib.util
# Get the location of the main superbit package.
dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0,dir)
import superbit.medsmaker
#spec = importlib.util.spec_from_file_location("superbit", dir)
#superbit = importlib.util.module_from_spec(spec)
#spec.loader.exec_module(foo)
