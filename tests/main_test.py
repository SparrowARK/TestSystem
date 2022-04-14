import os
import sys

sys.path.append(os.getcwd())
from recognition.utils import create_dataset


def test_create_dataset():
    output = create_dataset("shekhar", mode="test")
    assert output == True


def test_predict():
    # load bunch of images : dummy for test
    # get faces from the images
    # align the faces
    # add the list of faces in list and send to the predict function
    # define model svc
    assert True == True
