from demo import run
from settings import instance as settings
from training import train
from tuning import tune

if __name__ == '__main__':
    if settings().mode == 'train':
        while(True):
            if settings().tuning:
                print("Tuning mode")
                tune(settings())
            print("Training mode")
            if (train(settings()) != -2):
                break
    elif settings().mode == 'demo':
        print("Demo mode")
        run(settings())
