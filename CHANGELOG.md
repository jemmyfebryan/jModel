# Changelog

All notable changes to this project will be documented in this file.

## v0.1.0 - 2024-03-24

### Added
- Added compatibility for **SimpleRNN** layers with hyperparameters: **units**, **activation**, **use_bias**, **return_sequences**
- Added non-weighted layers: InputLayer, Reshape, Flatten, Dropout

### Changed
- Fixed Dense layer not working well with batch data
- Fixed Flatten layer is flattening all dimension including batch data
- Fixed Reshape layer is reshaping all dimension including batch data

## v0.0.2 - 2024-03-24

### Added
- Added compatibility for **LSTM** layers with hyperparameters: **units**, **activation**, **use_bias**, **return_sequences**
- Added Softmax Activation Function

### Changed
- Changed the layer hyperparameters sequence for tf2jModel to same as TensorFlow

## v0.0.1 - 2024-03-24

### Added

- jModel released