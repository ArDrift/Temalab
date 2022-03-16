import data_loader
import sys


def main():
    data_loader.loadShapeAndSaveDataset(sys.argv[1], sys.argv[2])

if __name__ == "__main__":
    main()