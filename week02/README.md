# Week 02 — SVD Image Compression

This project investigates image compression using truncated Singular Value Decomposition (SVD). It compares how well images with different levels of visual complexity can be represented using a reduced number of singular components.

The project measures both reconstruction quality and the actual number of values that must be stored in the truncated SVD representation.

## Project Structure

```text
week02/
├── project2_svd.ipynb
├── images/
│   ├── smooth.jpg
│   ├── detailed.jpg
│   └── mixed.jpg
├── REPORT.md
└── README.md
```

## Images Used

Three images with different structural characteristics were selected:

* `smooth.jpg` — sky image representing a smooth, low-detail image
* `detailed.jpg` — foliage image representing a highly detailed and textured image
* `mixed.jpg` — building image containing a mixture of smooth regions and sharp edges

### Image Sources and Licences

**Smooth – Sky**

Source: **[ADD ORIGINAL IMAGE PAGE URL]**
Licence: **[ADD LICENCE / USAGE PERMISSION]**

**Detailed – Foliage**

Source: **[ADD ORIGINAL IMAGE PAGE URL]**
Licence: **[ADD LICENCE / USAGE PERMISSION]**

**Mixed – Buildings**

Source: **[ADD ORIGINAL IMAGE PAGE URL]**
Licence: **[ADD LICENCE / USAGE PERMISSION]**

> Before submission, the original webpage and licence for each image should be provided. Google Images itself is a search engine and should not be listed as the image source.

## Requirements

The project was developed in Python and requires:

```text
numpy
pandas
matplotlib
pillow
scikit-image
```

The packages can be installed with:

```bash
pip install numpy pandas matplotlib pillow scikit-image
```

## How to Run

1. Clone or download the repository.
2. Open the `week02` folder in VS Code.
3. Open `project2_svd.ipynb`.
4. Select a Python environment containing the required packages.
5. Restart the notebook kernel.
6. Run all cells from top to bottom.

The notebook uses relative file paths, so no changes to paths should be necessary as long as the images remain inside the `images/` folder.

## Reproducibility

A fixed NumPy random seed is used for the Gaussian-noise experiment:

```python
seed = 42
rng = np.random.default_rng(seed)
```

This ensures that the same noise is generated whenever the notebook is run again.

Images larger than 750 pixels are resized while preserving their aspect ratio to keep the SVD computations manageable and make the comparison between images more consistent.

## Greyscale Conversion

Images are converted from RGB to greyscale using Rec. 709 luminance weights:

```text
Y = 0.2126R + 0.7152G + 0.0722B
```

Pixel intensities are then normalised to floating-point values between 0 and 1.

## Analysis Included

The notebook includes:

* reduced SVD using `full_matrices=False`
* full-rank reconstruction verification
* rank-\(k\) image reconstruction
* correct SVD storage calculation: `k × (m + n + 1)`
* compression ratio calculation
* relative Frobenius reconstruction error
* retained singular-value energy
* break-even rank calculation
* 90%, 99% and 99.9% energy thresholds
* singular-value spectrum and elbow analysis
* selection of `k` using a 5% relative-error target
* comparison between smooth, detailed and mixed images
* mean-centring experiment
* Gaussian-noise experiment at two noise levels
* SVD denoising analysis
* SSIM as an additional perceptual-quality measure

## Main Finding

The effectiveness of truncated SVD depends strongly on the structure of the image.

The smooth sky image reached relative error below 5% at only `k = 6`, corresponding to approximately **49.27× compression** under the scalar-storage definition used in the project.

The building image required `k = 138`, giving approximately **2.48× compression**, while the highly textured foliage image required `k = 206` and achieved only approximately **1.31× compression**.

This demonstrates that smooth, highly correlated images are much more suitable for low-rank compression than images containing large amounts of texture and fine detail.