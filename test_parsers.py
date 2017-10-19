"""Unit tests for parsers."""

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
	assert dicom_image.shape == (256, 256)

	fid = -30
	dicom_parser = parsers.DicomParser(pid, fid)
	dicom_image = dicom_parser.parse()
	assert dicom_image is None


def test__construct_contour_filename():
	link_df = load_pid_oid()

	oid = link_df['original_id'][0]
	fid = 59
	contour_type = 'i'
	expected_fn = 'contourfiles/' + oid + '/i-contours/IM-0001-0059-icontour-manual.txt'
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
	contour_type = 'i'
	contour_parser = parsers.ContourParser(oid, fid, contour_type)
	contour_polygon = contour_parser.parse()
	assert isinstance(contour_polygon, list)

	oid = link_df['patient_id'][0]
	contour_parser = parsers.ContourParser(oid, fid, contour_type)
	contour_polygon = contour_parser.parse()
	assert contour_polygon is None


def test_parse_boolean_mask():
	link_df = load_pid_oid()

	pid = link_df['patient_id'][0]
	oid = link_df['original_id'][0]
	fid = 59
	dicom_parser = parsers.DicomParser(pid, fid)
	dicom_image = dicom_parser.parse()

	contour_type = 'i'
	contour_parser = parsers.ContourParser(oid, fid, contour_type)
	contour_polygon = contour_parser.parse()

	mask_parser = parsers.MaskParser(dicom_image, contour_polygon)
	mask = mask_parser.parse()
	assert mask.shape == dicom_image.shape

	mask_parser = parsers.MaskParser(None, contour_polygon)	
	mask = mask_parser.parse()
	assert mask is None

	mask_parser = parsers.MaskParser(dicom_image, None)
	mask = mask_parser.parse()
	assert mask is None


def test__process_dicom_image():
	link_df = load_pid_oid()

	pid = link_df['patient_id'][0]
	oid = link_df['original_id'][0]
	fid = 59
	dicom_parser = parsers.DicomParser(pid, fid)
	dicom_image = dicom_parser.parse()

	contour_type = 'i'
	contour_parser = parsers.ContourParser(oid, fid, contour_type)
	contour_polygon = contour_parser.parse()

	mask_parser = parsers.MaskParser(dicom_image, contour_polygon)
	dicom_image3D = mask_parser._process_dicom_image()
	assert dicom_image3D.shape == (dicom_image.shape[0], dicom_image.shape[1], 3)


if __name__ == '__main__':
	print('testing DicomParser...')
	test__construct_dicom_filename()
	test_parse_dicom_file()

	print('testing ContourParser...')
	test__construct_contour_filename()
	test_parse_contour_file()

	print('testing MaskParser...')
	test__process_dicom_image()
	test_parse_boolean_mask()


