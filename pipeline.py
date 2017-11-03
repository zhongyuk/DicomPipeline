"""Training pipeline.

__author__ = 'zhongyu kuang'
__email__ = 'zhongyu.kuang@gmail.com'
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import parsers
import pandas as pd


class TrainingPipeline(object):
    """A pipeline for preparing training data.

    A training pipeline for parsing and batch serving training data:
        - parsing DICOM image, contours, mask
        - batch serving input-output pairs of training data
        - random reshuffle all training data per epoch
    """

    def __init__(self, link_fn, batch_size):
        """A constructor for initializing a TrainingPipeline instance.

        Params:
            link_fn: str - filename of csv file which stores
                     patient_id-original_id pairs
            batch_size: int - batch size for each serving
        Member variables:
            self._contour_type: str - 'o' for `o-contour`, 'i' for `i-contour`
            self._inputs: all the available DICOM image pixel data
            self._targets: all the available boolean mask
            self._start: initial starting index for batching the entire dataset
        """

        self.link_fn = link_fn
        self.batch_size = batch_size

        self._contour_types = ['i', 'o']
        self._inputs = None
        self._targets = None
        self._start = 0

    def _get_all_dicom_ids(self, patient_id):
        """Fetch all DICOM image file ID in the directory.

        Params:
            patient_id: str - encoded patient ID for constructing
                        DICOM image filename
        Return:
            dicom_ids: list of int - list of DICOM image file ID
        """

        dicom_path = (parsers.DicomParser.DIR + patient_id +
                      parsers.DicomParser.FOLDER)
        dicom_fns = [fn for fn in os.listdir(dicom_path)
                     if fn.endswith(parsers.DicomParser.FN_POSTFIX)]
        dicom_ids = map(lambda fn: int(
            fn[: fn.index(parsers.DicomParser.FN_POSTFIX)]), dicom_fns)
        return dicom_ids

    def _extract_contour_id(self, contour_filename, contour_type):
        """Extract a single contour file ID from the contour filename.

        Params:
            contour_filename: str - the contour filename.
                              i.e. `IM-0001-0009-icontour-manual.txt`
            contour_type: str - 'i' or 'o' indicating contour types
        Return:
            contour_file_id: int - contour file ID
        """

        start = contour_filename.index(parsers.ContourParser.FN_PREFIX)
        start += len(parsers.ContourParser.FN_PREFIX)
        if contour_type == 'i':
            end = contour_filename.index(
                parsers.ContourParser.ICONTOUR_FN_POSTFIX)
        if contour_type == 'o':
            end = contour_filename.index(
                parsers.ContourParser.OCONTOUR_FN_POSTFIX)
        return int(contour_filename[start + 1: end])

    def _get_all_contour_ids(self, original_id):
        """Fetch all contour file ID in the directory.

        Params:
            original_id: str - encoded original ID for constructing
                         i-contour filename
        Return:
            contour_ids: list of int - list of contour file ID
        """

        for contour_type in self._contour_types:
            if contour_type == 'i':
                icontour_path = (parsers.ContourParser.DIR + original_id +
                                 parsers.ContourParser.ICONTOUR_FOLDER)
                icontour_fns = [fn for fn in os.listdir(icontour_path)
                                if fn.endswith(
                                parsers.ContourParser.ICONTOUR_FN_POSTFIX)]
                icontour_ids = map(lambda fn: self._extract_contour_id(
                                   fn, contour_type), icontour_fns)

            if contour_type == 'o':
                ocontour_path = (parsers.ContourParser.DIR + original_id +
                                 parsers.ContourParser.OCONTOUR_FOLDER)
                ocontour_fns = [fn for fn in os.listdir(ocontour_path)
                                if fn.endswith(
                               parsers.ContourParser.OCONTOUR_FN_POSTFIX)]
                ocontour_ids = map(lambda fn: self._extract_contour_id(
                                   fn, contour_type), ocontour_fns)

        return list(set(icontour_ids).intersection(set(ocontour_ids)))

    def _pair_dicom_and_contour(self):
        """Innersect/Inner-join (SQL-alike) available dicom IDs and contour IDs.

        Return:
            pid_oid: a pd.DataFrame with columns of
                    ['patient_id', 'original_id', 'file_id']
        """
        pid_oid = pd.read_csv(self.link_fn)
        pid_oid['file_id'] = pid_oid.apply(
            lambda row: list(set(self._get_all_dicom_ids(
                row['patient_id'])).intersection(set(self._get_all_contour_ids(
                    row['original_id'])))), axis=1)
        return pid_oid

    def _parse_input_output(self, pid_oid, visualize):
        """Parse all the paired up DICOM, contour, and producing bool masks.

        Params:
            pid_oid: DataFrame stores: `patient_id`, `original_id`,
                     list of `file_id`
            visualize: boolean - if True, invoke data visualization
        """

        dicom_input = []
        mask_output = []

        for _, row in pid_oid.iterrows():
            pid = row['patient_id']
            oid = row['original_id']
            fids = row['file_id']

            for fid in fids:
                dicom = parsers.DicomParser(pid, fid).parse()
                masks = []

                for contour_type in self._contour_types:
                    contour = parsers.ContourParser(
                        oid, fid, contour_type).parse()
                    if dicom is not None and contour is not None:
                        mask = parsers.MaskParser(
                            dicom, contour).parse()
                        masks.append(mask)

                if len(masks) == 2:
                    dicom_input.append(dicom)
                    mask_output.append(np.stack(masks, -1))

                if visualize and dicom is not None and len(masks) == 2:
                    self._plot_input_output(dicom, masks[0], masks[1])

        self._inputs = np.stack(dicom_input, axis=0)
        self._targets = np.stack(mask_output, axis=0)

    def _plot_input_output(self, dicom, imask, omask):
        """Plot DICOM image, i-contour mask, o-contour mask side by side."""

        dicom_imsk = parsers.combine_dicom_with_mask(dicom, imask)
        dicom_omsk = parsers.combine_dicom_with_mask(dicom, omask)

        plt.subplot(1, 3, 1)
        plt.imshow(dicom)
        plt.title('Raw IDCOM Image')
        plt.subplot(1, 3, 2)
        plt.imshow(dicom_imsk)
        plt.title('I-Contour')
        plt.subplot(1, 3, 3)
        plt.imshow(dicom_omsk)
        plt.title('O-Contour')
        plt.show()

    def _random_shuffle(self):
        """Randomly shuffle `self._inputs` and `self._targets`."""

        assert self._inputs is not None, 'inputs have not been parsed yet!'
        assert self._targets is not None, 'targets have not been parsed yet!'
        assert len(self._inputs) == len(self._targets), \
            'inputs size does not equal to targets size!'

        indices = np.arange(len(self._inputs))
        np.random.shuffle(indices)

        self._inputs = self._inputs[indices]
        self._targets = self._targets[indices]

    def prepare_training_data(self, visualize=False):
        """Prepare trianing data by parsing inputs and preparing targets.

        Params:
            visualize: boolean - if True, invoke data visualization
                       while parsing data
        """

        print('Preparing training data...')
        pid_oid = self._pair_dicom_and_contour()
        self._parse_input_output(pid_oid, visualize)

    def next_batch(self):
        """An API for generating a new batch of input and output."""

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
