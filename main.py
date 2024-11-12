from demo import run
from settings import instance as settings
from training import train
from tuning import tune

if __name__ == '__main__':
    if settings().mode == 'train':
        print("Training mode")
        train(settings())
    elif settings().mode == 'tune':
        print("Tuning mode")
        tune(settings())
    elif settings().mode == 'demo':
        print("Demo mode")
        run(settings())
