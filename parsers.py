"""A class for parsing boolean mask.

__author__ = 'zhongyu kuang'
__email__ = 'zhongyu.kuang@gmail.com'
"""

import abc
import os

import dicom
from dicom.errors import InvalidDicomError

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


class Parser(object):

	@abc.abstractmethod
	def parse(self):
		raise NotImplementedError('parse method is not implemented!')


class DicomParser(Parser):
	"""DICOM Image Parser."""

	DIR = 'dicoms/'
	FOLDER = '/'
	FN_POSTFIX = '.dcm'

	def __init__(self, patient_id, dicom_id):
		"""A constructor for initializing a DicomParser instance.

		Params:
			patient_id: str - encoded patient ID for constructing DICOM image filename
			dicom_id: int - the specified IDCOM image ID
		"""
		self.patient_id = patient_id
		self.dicom_id = dicom_id

	def _construct_filename(self):
		"""Construct DICOM image filename.

		Return:
			dicom_fn: str - the specified IDCOM image filename
			raise IOError if file does not exist
		"""

		dicom_fn = (DicomParser.DIR + self.patient_id + DicomParser.FOLDER+
					str(self.dicom_id) + DicomParser.FN_POSTFIX)

		if os.path.isfile(dicom_fn):
			return dicom_fn
		else:
			raise IOError('dicom file does not exist: %s' % (dicom_fn))

	def parse(self):
		"""Parse the given DICOM filename

		Return:
			dcm_image: ndarray - DICOM image data
			Note - it outputs `None` if it runs into the InvalidDicomError.
		"""
		try:
			dicom_fn = self._construct_filename()
		except IOError:
			return None

		try:
			dcm = dicom.read_file(dicom_fn)
			dcm_image = dcm.pixel_array

			try:
				intercept = dcm.RescaleIntercept
			except AttributeError:
				intercept = 0.0

			try:
				slope = dcm.RescaleSlope
			except AttributeError:
				slope = 0.0

			if intercept != 0.0 and slope != 0.0:
				dcm_image = dcm_image*slope + intercept

			return dcm_image

		except InvalidDicomError:
			return None


class ContourParser(Parser):
	"""Contour File Parser."""

	DIR = 'contourfiles/'
	FN_PREFIX = 'IM-0001-'
	ICONTOUR_FOLDER = '/i-contours/'
	OCONTOUR_FOLDER = '/o-contours/'
	ICONTOUR_FN_POSTFIX = '-icontour-manual.txt'
	OCONTOUR_FN_POSTFIX = '-ocontour-manual.txt'

	def __init__(self, original_id, contour_id, contour_type):
		"""A constructor for initializing a ContourParser instance.

		Params:
			original_id: str - encoded original ID for constructing i-contour filename
			contour_type: str - 'o' for `o-contours`, 'i' for `i-contours`
			contour_id: int - specify contour file ID
		"""
		assert contour_type in ['o', 'i'], 'unknown contour type: %s' % (contour_type)
		assert contour_id < 10 ** 5, 'contour ID is too large: %d' % (contour_id)
		self.original_id = original_id
		self.contour_type = contour_type
		self.contour_id = contour_id

	def _construct_filename(self):
		"""Construct i-contour filename.

		Return:
			contour_fn: str - the specified i-contour filename
			raise IOError if file does not exist
		"""

		import math

		num_digits = int(math.log10(self.contour_id)) + 1
		contour_id_str = '0' * (4 - num_digits) + str(self.contour_id)

		if self.contour_type == 'i':
			contour_fn = (ContourParser.DIR + self.original_id + ContourParser.ICONTOUR_FOLDER +
						  ContourParser.FN_PREFIX + contour_id_str +
						  ContourParser.ICONTOUR_FN_POSTFIX)
		elif self.contour_type == 'o':
			contour_fn = (ContourParser.DIR + self.original_id + ContourParser.OCONTOUR_FOLDER +
						  ContourParser.FN_PREFIX + contour_id_str +
						  ContourParser.OCONTOUR_FN_POSTFIX)

		if os.path.isfile(contour_fn):
			return contour_fn
		else:
			raise IOError('contour file does not exist: %s' % (contour_fn))

	def parse(self):
		"""Parse the given contour filename.

		Return:
			coords_lst: a list of tuples - holding x, y coordinates of the contour
		"""

		try:
			contour_fn = self._construct_filename()
		except IOError:
			return None

		coords_lst = []

		with open(contour_fn, 'r') as infile:
			for line in infile:
				coords = line.strip().split()

				x_coord = float(coords[0])
				y_coord = float(coords[1])
				coords_lst.append((x_coord, y_coord))

		return coords_lst


class MaskParser(Parser):
	"""A pipeline for creating boolean mask from DICOM images and contour files."""

	def __init__(self, dicom_image, contour_polygon):
		"""A constructor for initializing a MaskParser instance.

		Params:
			dicom_image: ndarray - dicom image pixel data
			contour_polygon: list of tuples - contour polygon coordinates
		"""

		self.dicom_image = dicom_image
		self.contour_polygon = contour_polygon

	def _convert_polyon_to_mask(self, polygon, width, height):
		"""Convert polygon to mask.

		http://stackoverflow.com/a/3732128/1410871

		Params:
			polygon: list of tuples - x, y coords [(x1, y1), (x2, y2), ...] in units of pixels
			width: int - scalar image width
			height: int - scalar image height

		Return:
			mask: ndarray - a boolean mask in shape of (height, width)
		"""

		img = Image.new(mode='L', size=(width, height), color=0)
		ImageDraw.Draw(img).polygon(xy=polygon, outline=0, fill=1)
		mask = np.array(img).astype(bool)
		return mask

	def parse(self, visualize=False):
		"""Create boolean mask based on DICOM image and i-contour polygon.

		Conventionally, dicom_id and icontour_id should be the same value.
		If dicom_id != icontour_id, a warning is raised to warn user.
		However, the program will continue execution.

		Params:
			visualize: boolean - if True, invoke visualization of DICOM image and mask

		Return:
			boolean_mask: ndarray - the parsed boolean mask
			Note - it outputs `None` if any of the following is true:
			1) the specified DICOM image does not exist;
			2) the specified i-contour file does not exist;
			3) run into InvalidDicomError while parsing DICOM image.
		"""

		if self.dicom_image is None or self.contour_polygon is None: return None

		img_width, img_height = self.dicom_image.shape

		boolean_mask = self._convert_polyon_to_mask(self.contour_polygon, img_width, img_height)

		if visualize:
			self._visualize_dicom_with_mask(boolean_mask)

		return boolean_mask

	def _visualize_dicom_with_mask(self, boolean_mask):
		"""Display DICOM image and DICOM image with mask side by side."""

		dicom_image_rgb = self._process_dicom_image()
		dicom_with_mask = self._combine_dicom_with_mask(dicom_image_rgb, boolean_mask)

		plt.subplot(1, 2, 1)
		plt.imshow(dicom_image_rgb)
		plt.subplot(1, 2, 2)
		plt.imshow(dicom_with_mask)
		plt.show()

	def _combine_dicom_with_mask(self, dicom_image_rgb, boolean_mask, rgb=[147, 112, 219]):
		"""Combine DICOM image with boolean mask.

		Params:
			dicom_image_rgb: ndarray - 3D numpy ndarray of DICOM image in RGB format
			boolean_mask: ndarray - 2D numpy ndarray of boolean
			rgb: list of int - color mask RGB value

		Return:
			dicom_with_mask: ndarray - 3D numpy ndarray of DICOM image with boolean mask
		"""

		mask_3D = np.repeat(boolean_mask[:, :, None], 3, 2)
		r = rgb[0] * np.ones_like(boolean_mask)
		g = rgb[1] * np.ones_like(boolean_mask)
		b = rgb[2] * np.ones_like(boolean_mask)
		color_mask = np.stack([r, g, b], axis=2).astype('uint8')
		dicom_with_mask = dicom_image_rgb + (-1) * dicom_image_rgb * mask_3D + mask_3D * color_mask
		return dicom_with_mask.astype('uint8')

	def _process_dicom_image(self):
		"""Process DICOM image into RGB format.

		Return:
			dicom_image_3D: ndarray - 3D numpy ndarray of DICOM image in RGB format
		"""

		dicom_image_normalized = 255 * self.dicom_image / float(self.dicom_image.max())
		dicom_image_3D = np.repeat(dicom_image_normalized[:, :, None], 3, 2)
		return dicom_image_3D.astype('uint8')

	def visualize(self):
		"""API for user to visualize DICOM image and boolean mask.

		Conventionally, dicom_id and icontour_id should be the same value.
		If dicom_id != icontour_id, a warning is raised to warn user.
		However, the program will continue execution.

		Return:
			None: the program will plot an image.
			Note - no image will be plotted if any of the following is true:
			1) the specified DICOM image does not exist;
			2) the specified i-contour file does not exist;
			3) run into InvalidDicomError while parsing DICOM image.
		"""
		if self.dicom_image is not None and self.contour_polygon is not None:
			img_width, img_height = self.dicom_image.shape
			boolean_mask = self._convert_polyon_to_mask(self.contour_polygon,
														img_width, img_height)
			self._visualize_dicom_with_mask(boolean_mask)