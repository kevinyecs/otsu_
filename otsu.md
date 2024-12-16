# Otsu Thresholding Implementation Documentation

## Overview
This MATLAB implementation provides both single-level and multi-level Otsu thresholding for image segmentation. The algorithm automatically determines optimal thresholds by maximizing the between-class variance of the pixel intensity levels.

## Main Functions

### complete_otsu()
The main entry point that demonstrates the usage of the Otsu thresholding implementation.

#### Operations:
1. Loads a test image ('Cat03.jpg')
2. Performs both single and 4-level thresholding
3. Displays results in a 2x2 figure layout
4. Prints threshold values

#### Usage:
```matlab
complete_otsu()
```

### otsu(img, N)
The core function that handles both single and multi-level thresholding.

#### Parameters:
- `img`: Input image (RGB or grayscale)
- `N`: Number of desired thresholds (default: 1)

#### Returns:
- `thresholds`: Array of computed threshold values (normalized [0,1])
- `quantized`: Quantized image based on the computed thresholds

#### Usage:
```matlab
[thresholds, quantized] = otsu(image, 4);  % For 4-level thresholding
[threshold, binary] = otsu(image);         % For single-level thresholding
```

## Supporting Functions

### singleOtsu(P)
Implements the classical Otsu method for single threshold computation.

#### Parameters:
- `P`: Normalized histogram (probability distribution) of the image

#### Returns:
- `threshold`: Single optimal threshold value (normalized [0,1])

### multiOtsu(P, N)
Extends Otsu's method to multiple thresholds.

#### Parameters:
- `P`: Normalized histogram of the image
- `N`: Number of desired thresholds

#### Returns:
- `thresholds`: Array of N optimal threshold values (normalized [0,1])

### calculateVariance(P, thresholds)
Computes the between-class variance for multi-level thresholding.

#### Parameters:
- `P`: Normalized histogram
- `thresholds`: Current threshold values being evaluated

#### Returns:
- `variance`: Computed between-class variance

### quantizeImage(img, thresholds)
Creates the output image based on computed thresholds.

#### Parameters:
- `img`: Input image
- `thresholds`: Array of threshold values

#### Returns:
- `quantized`: Quantized image with intensity levels based on thresholds

## Implementation Details

### Preprocessing
1. Converts RGB images to grayscale if necessary
2. Normalizes pixel values to [0,1] range
3. Computes image histogram and probability distribution

### Algorithm Steps
1. **Single-level Thresholding:**
   - Computes cumulative sums and means
   - Calculates between-class variance for each possible threshold
   - Selects threshold that maximizes variance

2. **Multi-level Thresholding:**
   - Iteratively finds optimal thresholds
   - For each new threshold:
     - Evaluates all possible threshold positions
     - Selects position that maximizes total variance
   - Sorts final thresholds in ascending order

3. **Quantization:**
   - Divides image into regions based on thresholds
   - Assigns intensity levels as midpoints between thresholds

## Error Handling
- Validates input parameters
- Handles RGB to grayscale conversion
- Normalizes input data if necessary
- Prevents division by zero in variance calculations

## Example Output
```matlab
Single threshold: 0.500
Multi thresholds: 0.200, 0.400, 0.600, 0.800
```

## Visualization
The implementation includes automatic visualization of:
- Original image
- Grayscale conversion (if applicable)
- Single-threshold result
- Multi-threshold result

## Performance Considerations
- Histogram computation: O(n) where n is number of pixels
- Single threshold search: O(L) where L is number of intensity levels
- Multi-threshold search: O(L * N) where N is number of thresholds

## Requirements
- MATLAB with Image Processing Toolbox
- Input image file ('Cat03.jpg' for default test case)

## Limitations
- Multi-level thresholding computational complexity increases with N
- Assumes unimodal or multimodal histogram distribution
- Memory usage scales with image size and number of thresholds