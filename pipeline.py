"""Training pipeline.

__author__ = 'zhongyu kuang'
__email__ = 'zhongyu.kuang@gmail.com'
"""

import os
import numpy as np
import parsers
import pandas as pd


class TrainingPipeline(object):
	"""A training pipeline for parsing and batch serving all data (DICOM image, contours, mask)."""

	def __init__(self, link_fn, contour_type, batch_size):
		"""A constructor for initializing a TrainingPipeline instance.

		Params:
			link_fn: str - filename of csv file which stores patient_id - original_id pairs
			contour_type: str - 'o' for `o-contours`, 'i' for `i-contours`
			batch_size: int - batch size for each serving
		Member variables:
			self._inputs: all the available DICOM image pixel data
			self._targets: all the available boolean mask
			self._start: initial starting index for batching the entire dataset
		"""

		assert contour_type in ['o', 'i'], 'unkown contour type: %s' % (contour_type)
		self.link_fn = link_fn
		self.batch_size = batch_size
		self.contour_type = contour_type

		self._inputs = None
		self._targets = None
		self._start = 0

	def _get_all_dicom_ids(self, patient_id):
		"""Fetch all DICOM image file ID in the directory.

		Params:
			patient_id: str - encoded patient ID for constructing DICOM image filename
		Return:
			dicom_ids: list of int - list of DICOM image file ID
		"""

		dicom_path = (parsers.DicomParser.DIR + patient_id +
					  parsers.DicomParser.FOLDER)
		dicom_fns = [fn for fn in os.listdir(dicom_path)
					 if fn.endswith(parsers.DicomParser.FN_POSTFIX)]
		dicom_ids = map(lambda fn: int(fn[: fn.index(parsers.DicomParser.FN_POSTFIX)]), dicom_fns)
		return dicom_ids

	def _extract_contour_id(self, contour_filename):
		"""Extract a single contour file ID from the contour filename.

		Params:
			contour_filename: str - the contour filename. i.e. `IM-0001-0009-icontour-manual.txt`
		Return:
			contour_file_id: int - contour file ID
		"""

		start = contour_filename.index(parsers.ContourParser.FN_PREFIX)
		start += len(parsers.ContourParser.FN_PREFIX)
		if self.contour_type == 'i':
			end = contour_filename.index(parsers.ContourParser.ICONTOUR_FN_POSTFIX)
		else:
			end = contour_filename.index(parsers.ContourParser.OCONTOUR_FN_POSTFIX)
		return int(contour_filename[start + 1 : end])

	def _get_all_contour_ids(self, original_id):
		"""Fetch all contour file ID in the directory.

		Params:
			original_id: str - encoded original ID for constructing i-contour filename
		Return:
			contour_ids: list of int - list of contour file ID
		"""

		if self.contour_type == 'i':
			contour_path = (parsers.ContourParser.DIR + original_id + 
					    	parsers.ContourParser.ICONTOUR_FOLDER)
			contour_fns = [fn for fn in os.listdir(contour_path)
						   if fn.endswith(parsers.ContourParser.ICONTOUR_FN_POSTFIX)]

		else:
			contour_path = (parsers.ContourParser.DIR + original_id + 
					    	parsers.ContourParser.OCONTOUR_FOLDER)
			contour_fns = [fn for fn in os.listdir(contour_path)
						   if fn.endswith(parsers.ContourParser.OCONTOUR_FN_POSTFIX)]

		contour_ids = map(lambda fn: self._extract_contour_id(fn), contour_fns)
		return contour_ids

	def _pair_dicom_and_contour(self):
		"""Innersect/Inner-join (SQL-alike) available dicom file IDs and contour file IDs.

		Return:
			pid_oid: a pd.DataFrame with columns of ['patient_id', 'original_id', 'file_id']
		"""
		pid_oid = pd.read_csv(self.link_fn)
		pid_oid['file_id'] = pid_oid.apply(
			lambda row: list(set(self._get_all_dicom_ids(row['patient_id'])).intersection(
				set(self._get_all_contour_ids(row['original_id'])))), axis=1)
		return pid_oid

	def _parse_input_output(self, pid_oid):
		"""Parse all the paired up DICOM images, contour files, and producing boolean masks.

		Params:
			pid_oid: DataFrame stores `patient_id`, `original_id`, list of `file_id`
		"""

		dicom_input = []
		mask_output = []

		for _, row in pid_oid.iterrows():
			pid = row['patient_id']
			oid = row['original_id']
			fids = row['file_id']

			for fid in fids:
				dicom = parsers.DicomParser(pid, fid).parse()
				contour = parsers.ContourParser(oid, fid, self.contour_type).parse()
				if dicom is not None and contour is not None:
					mask = parsers.MaskParser(dicom, contour).parse()

					dicom_input.append(dicom)
					mask_output.append(mask)

		self._inputs = np.stack(dicom_input, axis=0)
		self._targets = np.stack(mask_output, axis=0)

	def _random_shuffle(self):
		"""Randomly shuffle `self._inputs` and `self._targets`."""

		assert self._inputs is not None, 'inputs have not been parsed yet!'
		assert self._targets is not None, 'targets have not been parsed yet!'
		assert len(self._inputs) == len(self._targets), 'inputs size does not equal to targets size!'

		indices = np.arange(len(self._inputs))
		np.random.shuffle(indices)

		self._inputs = self._inputs[indices]
		self._targets = self._targets[indices]

	def prepare_training_data(self):
		"""Prepare trianing data by parsing inputs and preparing targets."""

		print('Preparing training data...')
		pid_oid = self._pair_dicom_and_contour()
		self._parse_input_output(pid_oid)

	def next_batch(self):
		"""An API for generating a new batch of training input and target output."""

		assert self._inputs is not None, 'inputs have not been parsed yet!'
		assert self._targets is not None, 'targets have not been parsed yet!'

		batch_input, batch_target = None, None

		end = self._start + self.batch_size
		if end < len(self._inputs):
			batch_input = self._inputs[self._start: end, :, :]
			batch_target = self._targets[self._start: end, :, :]
			
		else:
			end = end % len(self._inputs)
			batch_input = np.concatenate([self._inputs[self._start:, :, :],
										 self._inputs[: end, :, :]], axis=0)
			batch_target = np.concatenate([self._targets[self._start:, :, :],
										  self._targets[: end, :, :]], axis=0)

			print('Finished iterating one epoch, reshuffling...')
			self._random_shuffle()

		self._start = end
		return batch_input, batch_target