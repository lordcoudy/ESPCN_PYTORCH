from demo import run
from settings import *
from training import train
from tuning import tune

if __name__ == '__main__':
    if dictionary['mode'] == 'train':
        print("Training mode")
        train()
    elif dictionary['mode'] == 'tune':
        print("Tuning mode")
        tune()
    elif dictionary['mode'] == 'demo':
        print("Demo mode")
        run()
