## DICOM Image Parsing and Training Pipeline

### Programming Language
- Python 2.7

### Library Dependencies
- numpy r1.13.0
- pandas r0.20.2
- dicom r0.9.9
- matplotlib
- PIL

### Parsers (file: `parsers.py`, testing file: `test_parsers.py`)
- DicomParser: a class for parsing DICOM image
- ContourParser: a class for parsing contour files and rendering a contour polygon
- MaskParser: a class for producing a boolean mask given DICOM image and contour polygon

### Training Pipeline (file: `pipeline.py`, testing file: `test_pipeline.py`)
- TrainingPipeline: a class for pairing DICOM images and contour files, producing boolean masks, parsing all data, and batch serving `(input, target)` pairs

### Demo (file `demo.ipynb`)
- A demoing of the main APIs of `parsers.py` and `pipeline.py` is illustrated in the `demo.ipynb` notebook. In addition to the demonstration of the major APIs, a few discussions on how the pieces are developed, how the correctness of the program is verified, and how the `parser` and `pipeline` can be further improved in the future are also included.
