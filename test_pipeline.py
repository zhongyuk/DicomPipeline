"""Unit tests for pipeline."""

import pipeline
import parsers
import pandas as pd


def test__extract_contour_id():
    link_fn = 'link.csv'
    batch_size = 8
    train_pipe = pipeline.TrainingPipeline(link_fn, batch_size)

    icontour_filename1 = 'IM-0001-0059-icontour-manual.txt'
    icontour_type = 'i'
    fid1 = train_pipe._extract_contour_id(icontour_filename1, icontour_type)
    assert fid1 == 59

    train_pipe = pipeline.TrainingPipeline(link_fn, batch_size)
    ocontour_filename2 = 'IM-0001-0079-ocontour-manual.txt'
    ocontour_type = 'o'
    fid2 = train_pipe._extract_contour_id(ocontour_filename2, ocontour_type)
    assert fid2 == 79


def test_prepare_training_data():
    link_fn = 'link.csv'
    batch_size = 8
    train_pipe = pipeline.TrainingPipeline(link_fn, batch_size)
    train_pipe.prepare_training_data()

    pid_oid_df = train_pipe._pair_dicom_and_contour()
    cnt = 0
    for _, row in pid_oid_df.iterrows():
        cnt += len(row['file_id'])

    assert len(train_pipe._inputs) == cnt
    assert len(train_pipe._targets) == cnt
    assert train_pipe._inputs.shape == (cnt, 256, 256, 3)
    assert train_pipe._targets.shape == (cnt, 256, 256, 3, 2)


def test_next_batch():
    link_fn = 'link.csv'
    batch_size = 8
    train_pipe = pipeline.TrainingPipeline(link_fn, batch_size)
    train_pipe.prepare_training_data()

    expected_start = 0
    for i in range(15):
        assert expected_start == train_pipe._start
        batch_input, batch_target = train_pipe.next_batch()
        expected_start += batch_size
        if expected_start >= len(train_pipe._inputs):
            expected_start %= len(train_pipe._inputs)


def test__plot_input_output():
    link_fn = 'link.csv'
    link_df = pd.read_csv(link_fn)

    pid = link_df['patient_id'][0]
    oid = link_df['original_id'][0]
    fid = 59

    dicom_parser = parsers.DicomParser(pid, fid)
    dicom_image = dicom_parser.parse()

    icontour_type = 'i'
    icontour_parser = parsers.ContourParser(oid, fid, icontour_type)
    icontour_polygon = icontour_parser.parse()

    imask_parser = parsers.MaskParser(dicom_image, icontour_polygon)
    imask = imask_parser.parse()

    ocontour_type = 'o'
    ocontour_parser = parsers.ContourParser(oid, fid, ocontour_type)
    ocontour_polygon = ocontour_parser.parse()

    omask_parser = parsers.MaskParser(dicom_image, ocontour_polygon)
    omask = omask_parser.parse()

    train_pipe = pipeline.TrainingPipeline(link_fn, 8)
    train_pipe._plot_input_output(dicom_image, imask, omask)


if __name__ == '__main__':
    print('testing TrainingPipeline...')
    test__extract_contour_id()
    test_prepare_training_data()
    test_next_batch()
    test__plot_input_output()
