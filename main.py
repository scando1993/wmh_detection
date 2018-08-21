from data_process import main_preprossed
from model_segmentation import main_segmentation
from test_results import main_test


def main():
    main_preprossed()
    main_segmentation()
    main_test()


if "__main__" == __name__:
    main()
