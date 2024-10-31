import torch.utils.data as data
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
from os import listdir
from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile
from PIL import Image

def isImageFile(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def loadImg(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y


class DatasetFromFolder(data.Dataset):
    def __init__(self, imageDir, inTransform=None, tgtTransform=None):
        super(DatasetFromFolder, self).__init__()
        self.imageFilenames = [join(imageDir, x) for x in listdir(imageDir) if isImageFile(x)]

        self.inputTransform = inTransform
        self.targetTransform = tgtTransform

    def __getitem__(self, index):
        inImg = loadImg(self.imageFilenames[index])
        target = inImg.copy()
        if self.inputTransform:
            inImg = self.inputTransform(inImg)
        if self.targetTransform:
            target = self.targetTransform(target)

        return inImg, target

    def __len__(self):
        return len(self.imageFilenames)


def downloadBSD300(dest="dataset"):
    outputImageDir = join(dest, "BSDS300/images")

    if not exists(outputImageDir):
        makedirs(dest)
        url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
        print("downloading url ", url)

        data = urllib.request.urlopen(url)

        filePath = join(dest, basename(url))
        with open(filePath, 'wb') as f:
            f.write(data.read())

        print("Extracting data")
        with tarfile.open(filePath) as tar:
            for item in tar:
                tar.extract(item, dest)

        remove(filePath)

    return outputImageDir


def calculateValidCropSize(cropSize, upscaleFactor):
    return cropSize - (cropSize % upscaleFactor)


def inputTransform(cropSize, upscaleFactor):
    return Compose([
        CenterCrop(cropSize),
        Resize(cropSize // upscaleFactor),
        ToTensor(),
    ])


def targetTransform(cropSize):
    return Compose([
        CenterCrop(cropSize),
        ToTensor(),
    ])


def getTrainingSet(upscaleFactor):
    rootDir = downloadBSD300()
    trainDir = join(rootDir, "train")
    cropSize = calculateValidCropSize(256, upscaleFactor)

    return DatasetFromFolder(trainDir,
                             inTransform=inputTransform(cropSize, upscaleFactor),
                             tgtTransform=targetTransform(cropSize))


def getTestSet(upscaleFactor):
    rootDir = downloadBSD300()
    testDir = join(rootDir, "test")
    cropSize = calculateValidCropSize(256, upscaleFactor)

    return DatasetFromFolder(testDir,
                             inTransform=inputTransform(cropSize, upscaleFactor),
                             tgtTransform=targetTransform(cropSize))