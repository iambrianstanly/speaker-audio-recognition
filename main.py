import yaml
import argparse
from src.data_loader import get_dataloaders
from src.model import get_model
from src.train import train_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if config["mode"] == "train":

        train_ds, valid_ds = get_dataloaders(config)
        
        model = get_model(config)

        train_model(model, train_ds, valid_ds, config)

        
if __name__ == "__main__":

    main()