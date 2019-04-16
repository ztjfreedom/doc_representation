import dataset


def main():
    dataset_csv_file = 'Google Job Skills'
    cols = ['Title', 'Responsibilities', 'Category']
    dataset.init_common_dataset(dataset_csv_file, cols, lan_chk=True)
    dataset.pre_process()


if __name__ == '__main__':
    main()
