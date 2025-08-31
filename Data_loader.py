import logging 
import pandas as pd 
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoader:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.logger = logging.getLogger(__name__)

    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def load_data(self):
        try:
            df = pd.read_csv(self.config['Dataset_path'])
            self.logger.info(f"Successfully loaded data")
            return df
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return None

if __name__ == "__main__":
    data_loader = DataLoader('config.yml')
    df = data_loader.load_data 
    logging.info(f"Successfully loaded data")
