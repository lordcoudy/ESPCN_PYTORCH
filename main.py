from demo import run
from settings import instance as settings
from training import train
from tuning import tune
from utils import prune_model

if __name__ == '__main__':
    if settings().mode == 'train':
        if settings().pruning:
            print("Pruning")
            prune_model(model=settings().model, amount=settings().prune_amount)
        if settings().tuning:
            print("Tuning mode")
            tune(settings())
        print("Training mode")
        train(settings())
    elif settings().mode == 'demo':
        print("Demo mode")
        run(settings())
