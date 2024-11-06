import torch
import yaml
from torch import optim, nn
from torch.utils.data import DataLoader

from yaml import load

from data import getTrainingSet, getTestSet

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

stream = open("settings.yaml", 'r')
dictionary = yaml.load(stream, Loader=Loader)


if dictionary['model'] == 'Optimized':
    from model_ench import OptimizedESPCN as espcn
else:
    from model import ESPCN as espcn

def prepareData(upscaleFactor):
    print('===> Loading datasets >===')
    trainSet = getTrainingSet(upscaleFactor)
    testSet = getTestSet(upscaleFactor)
    trainingDataLoader = DataLoader(dataset=trainSet, num_workers=dictionary['threads'], batch_size=dictionary['batch_size'], shuffle=True)
    testingDataLoader = DataLoader(dataset=testSet, num_workers=dictionary['threads'], batch_size=dictionary['test_batch_size'], shuffle=False)
    print('===> Datasets loaded >===')
    return trainingDataLoader, testingDataLoader

# Device configuration
device = torch.device("cuda" if dictionary['cuda'] and torch.cuda.is_available() else "cpu")
torch.manual_seed(dictionary['seed'])
model = espcn().to(device)
g_criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=dictionary['learning_rate'], momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
scaler = torch.amp.GradScaler('cuda', enabled=dictionary['mixed_precision'])
trainLoader, testLoader = prepareData(dictionary['upscale_factor'])
