from data_prepare.load_file import load_dicom_series # TODO： WTF ？

if __name__ == "__main__":
    folder_path = "data/fix_test_data"
    series_data = load_dicom_series(folder_path, convert_type='auto')