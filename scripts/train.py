import cv2
import sys
from oarsigrading.kvs import GlobalKVS
from oarsigrading.training import session


cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

DEBUG = sys.gettrace() is not None

if __name__ == "__main__":
    kvs = GlobalKVS()
    session.init_session()
    session.init_metadata()
    writers = session.init_folds()
    session.init_data_processing()