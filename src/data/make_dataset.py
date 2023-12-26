# library importation
import pathlib
import yaml
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# function for data loading
def load_data(data_path):
    # load your dataset from given path
    df = pd.read_csv(data_path)
    return df

# function for data spliting
def split_data(df, test_split, seed):
    # Spliting the dataset into Train Test sets
    train, test = train_test_split(df, test_size = test_split, random_state = seed)
    return train, test

# function for saving data splits
def save_data(train, test, output_path):
    # Save the split datasets to the specified output path
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    train.to_csv(output_path + '/train.csv', index = False)
    test.to_csv(output_path + '/test.csv', index = False)

def main():
    """
    Function to finding current prerequisits
    """
    cur_dir = pathlib.Path(__file__)
    home_dir = cur_dir.parent.parent.parent
    params_file = home_dir.as_posix() + '/params.yaml'
    params = yaml.safe_load(open(params_file))['make_dataset']

    input_file = sys.argv[1]
    data_path = home_dir.as_posix() + input_file
    output_path = home_dir.as_posix() + '/data/processed'

    # calling load_data function
    data = load_data(data_path)
    train_data, test_data = split_data(data, params['test_split'], params['seed'])
    save_data(train_data, test_data, output_path)

if __name__ == '__main__':
    main()