"""Unit tests for parsers."""

import numpy as np
import parsers
import pandas as pd


def load_pid_oid():
    link_fn = 'link.csv'
    return pd.read_csv(link_fn)


def test__construct_dicom_filename():
    link_df = load_pid_oid()

    pid = link_df['patient_id'][0]
    fid = 59
    expected_fn = 'dicoms/' + pid + '/' + str(fid) + '.dcm'
    dicom_parser = parsers.DicomParser(pid, fid)
    dicom_fn = dicom_parser._construct_filename()
    assert dicom_fn == expected_fn

    fid = -123
    dicom_parser = parsers.DicomParser(pid, fid)
    try:
        dicom_fn = dicom_parser._construct_filename()
    except IOError:
        print('passed detecting non-existing dicom file.')


def test_parse_dicom_file():
    link_df = load_pid_oid()

    pid = link_df['patient_id'][0]
    fid = 59
    dicom_parser = parsers.DicomParser(pid, fid)
    dicom_image = dicom_parser.parse()
    assert dicom_image.shape == (256, 256, 3)

    fid = -30
    dicom_parser = parsers.DicomParser(pid, fid)
    dicom_image = dicom_parser.parse()
    assert dicom_image is None


def test__construct_contour_filename():
    link_df = load_pid_oid()

    oid = link_df['original_id'][0]
    fid = 59
    contour_type = 'i'
    expected_fn = ('contourfiles/' + oid +
                   '/i-contours/IM-0001-0059-icontour-manual.txt')
    contour_parser = parsers.ContourParser(oid, fid, contour_type)
    contour_fn = contour_parser._construct_filename()
    assert contour_fn == expected_fn

    oid = link_df['patient_id'][0]
    contour_parser = parsers.ContourParser(oid, fid, contour_type)
    try:
        contour_fn = contour_parser._construct_filename()
    except IOError:
        print('passed detecting non-existing contour file.')


def test_parse_contour_file():
    link_df = load_pid_oid()

    oid = link_df['original_id'][0]
    fid = 59
    icontour_type = 'i'
    icontour_parser = parsers.ContourParser(oid, fid, icontour_type)
    icontour_polygon = icontour_parser.parse()
    assert isinstance(icontour_polygon, list)

    fid = 59
    ocontour_type = 'o'
    ocontour_parser = parsers.ContourParser(oid, fid, ocontour_type)
    ocontour_polygon = ocontour_parser.parse()
    assert isinstance(ocontour_polygon, list)

    oid = link_df['patient_id'][0]
    contour_parser = parsers.ContourParser(oid, fid, icontour_type)
    contour_polygon = contour_parser.parse()
    assert contour_polygon is None


def test_parse_boolean_mask():
    link_df = load_pid_oid()

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
    assert imask.shape == dicom_image.shape

    ocontour_type = 'o'
    ocontour_parser = parsers.ContourParser(oid, fid, ocontour_type)
    ocontour_polygon = ocontour_parser.parse()

    omask_parser = parsers.MaskParser(dicom_image, ocontour_polygon)
    omask = omask_parser.parse()
    assert omask.shape == dicom_image.shape

    mask_parser = parsers.MaskParser(None, icontour_polygon)
    mask = mask_parser.parse()
    assert mask is None

    mask_parser = parsers.MaskParser(dicom_image, None)
    mask = mask_parser.parse()
    assert mask is None


def test__process_dicom_image():
    link_df = load_pid_oid()

    pid = link_df['patient_id'][0]
    fid = 59
    dicom_parser = parsers.DicomParser(pid, fid)
    fake_dicom_image = np.random.randint(0, 1000, size=[256, 256])
    dicom_rgb = dicom_parser._process_dicom_image(fake_dicom_image)
    assert dicom_rgb.min() >= 0
    assert dicom_rgb.max() <= 255
    assert dicom_rgb.shape == (256, 256, 3)


def test_visualize_dicom_image():
    link_df = load_pid_oid()

    pid = link_df['patient_id'][0]
    fid = 59
    dicom_parser = parsers.DicomParser(pid, fid)
    dicom_parser.visualize()


def test_visualize_boolean_mask():
    link_df = load_pid_oid()

    pid = link_df['patient_id'][0]
    fid = 59
    dicom_parser = parsers.DicomParser(pid, fid)
    dicom_image = dicom_parser.parse()

    oid = link_df['original_id'][0]
    icontour_type = 'i'
    icontour_parser = parsers.ContourParser(oid, fid, icontour_type)
    icontour_polygon = icontour_parser.parse()

    imask_parser = parsers.MaskParser(dicom_image, icontour_polygon)
    imask_parser.visualize()

    ocontour_type = 'o'
    ocontour_parser = parsers.ContourParser(oid, fid, ocontour_type)
    ocontour_polygon = ocontour_parser.parse()

    omask_parser = parsers.MaskParser(dicom_image, ocontour_polygon)
    omask_parser.visualize()


if __name__ == '__main__':
    print('testing DicomParser...')
    test__construct_dicom_filename()
    test_parse_dicom_file()
    test__process_dicom_image()
    test_visualize_dicom_image()

    print('testing ContourParser...')
    test__construct_contour_filename()
    test_parse_contour_file()

    print('testing MaskParser...')
    test_parse_boolean_mask()
    test_visualize_boolean_mask()
