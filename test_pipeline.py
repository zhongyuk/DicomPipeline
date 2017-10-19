"""Unit tests for pipeline."""

import parsers
import pipeline


def test__extract_contour_id():
	link_fn = 'link.csv'
	contour_type = 'i'
	batch_size = 8
	train_pipe = pipeline.TrainingPipeline(link_fn, contour_type, batch_size)

	contour_filename1 = 'IM-0001-0059-icontour-manual.txt'
	fid1 = train_pipe._extract_contour_id(contour_filename1)
	assert fid1 == 59

	contour_type = 'o'
	train_pipe = pipeline.TrainingPipeline(link_fn, contour_type, batch_size)
	contour_filename2 = 'IM-0001-0079-ocontour-manual.txt'
	fid2 = train_pipe._extract_contour_id(contour_filename2)
	assert fid2 == 79


def test_prepare_training_data():
	link_fn = 'link.csv'
	contour_type = 'i'
	batch_size = 8
	train_pipe = pipeline.TrainingPipeline(link_fn, contour_type, batch_size)
	train_pipe.prepare_training_data()

	pid_oid_df = train_pipe._pair_dicom_and_contour()
	cnt = 0
	for _, row in pid_oid_df.iterrows():
		cnt += len(row['file_id'])

	assert len(train_pipe._inputs) == cnt
	assert len(train_pipe._targets) == cnt


def test_next_batch():
	link_fn = 'link.csv'
	contour_type = 'i'
	batch_size = 8
	train_pipe = pipeline.TrainingPipeline(link_fn, contour_type, batch_size)
	train_pipe.prepare_training_data()

	expected_start = 0
	for i in range(15):
		assert expected_start == train_pipe._start
		batch_input, batch_target = train_pipe.next_batch()
		expected_start += batch_size
		if expected_start >= len(train_pipe._inputs):
			expected_start %= len(train_pipe._inputs)


if __name__ == '__main__':
	print('testing TrainingPipeline...')
	test__extract_contour_id()
	test_prepare_training_data()
	test_next_batch()
