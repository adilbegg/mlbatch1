# SVD Image Compression Analysis

## 1. Introduction and SVD Interpretation

The aim of this project was to investigate how Singular Value Decomposition (SVD) can be used to compress images and to measure the trade-off between storage and reconstruction quality. I used three greyscale images with different characteristics: a smooth sky image, a highly detailed foliage image, and a building image containing a mixture of smooth regions and sharp edges.

For an image represented by a matrix \(A\), SVD decomposes the matrix as

$$
A = U\Sigma V^T
$$

where \(U\) and \(V^T\) contain spatial patterns and \(\Sigma\) contains the singular values. The singular values are arranged from largest to smallest. A large singular value means that its corresponding component makes a large contribution to the image, while smaller singular values represent progressively less important components.

A truncated SVD keeps only the first \(k\) components. This gives a rank-\(k\) approximation of the original image. If the important structure of an image can be represented by only a few components, the image can be compressed effectively. Images with complicated textures and many sharp local changes usually require more components.

The three images were converted to greyscale using Rec. 709 luminance weights:

$$
Y = 0.2126R + 0.7152G + 0.0722B
$$

and pixel values were converted to floating-point values between 0 and 1. The final image dimensions were 493 × 740 for the sky, 450 × 675 for the foliage, and 750 × 631 for the building image.

## 2. Measurement Methodology

An important part of this project was measuring compression honestly. The original \(m \times n\) greyscale image contains \(mn\) numbers. A rank-\(k\) SVD does not store only \(k\) numbers. It must store the first \(k\) columns of \(U\), the first \(k\) singular values, and the first \(k\) rows of \(V^T\).

The number of stored values is therefore

$$
mk + k + kn = k(m+n+1).
$$

The compression ratio used in this project was consequently

$$
\text{Compression Ratio} =
\frac{mn}{k(m+n+1)}.
$$

A ratio greater than 1 means fewer numbers are stored than in the original image. A ratio below 1 means that the SVD representation actually requires more values than the original.

The break-even ranks were 295.64 for the sky, 269.76 for the foliage, and 342.44 for the building image. Therefore, integer ranks of 296, 270, and 343 respectively are the first values at which the SVD representation stores more numbers than the original image.

Reconstruction quality was measured using relative Frobenius error,

$$
\frac{\|A-A_k\|_F}{\|A\|_F},
$$

and energy retained,

$$
\frac{\sum_{i=1}^{k}\sigma_i^2}
{\sum_i \sigma_i^2}.
$$

Full-rank reconstruction was also checked using `np.allclose`. All three images returned `True`, with maximum absolute differences on the order of \(10^{-13}\), confirming that the SVD decomposition and reconstruction were working correctly. Interestingly, full-rank SVD had compression ratios of only about 0.60 for the sky and foliage and 0.54 for the building image. This shows that exact reconstruction does not imply useful compression.

## 3. Choosing the Rank k

I compared three methods for selecting \(k\): an energy threshold, an elbow in the singular-value spectrum, and a fixed reconstruction-quality target.

The results were:

| Image              | 90% energy | 99% energy | 99.9% energy | Elbow k | k for error < 5% | Compression at 5% error |
| ------------------ | ---------: | ---------: | -----------: | ------: | ---------------: | ----------------------: |
| Smooth – Sky       |          1 |          2 |           16 |      50 |                6 |                  49.27× |
| Detailed – Foliage |          6 |        112 |          260 |     391 |              206 |                   1.31× |
| Mixed – Buildings  |          2 |         51 |          205 |      19 |              138 |                   2.48× |

The energy criterion behaved particularly strangely for the sky image. At only \(k=1\), the reconstruction already retained 98.86% of the mathematical energy, yet its relative error was still about 10.67%. This happens because a smooth image has a large overall brightness component. A nearly constant brightness level can itself be represented as a rank-1 matrix, making the first singular value extremely large even though fine visual information has not yet been reconstructed.

The mean-centring experiment helps explain this behaviour. Subtracting the mean brightness removes the global brightness offset, so the remaining singular values represent spatial variation more directly.

The elbow method also produced quite different results between images. The estimated elbows were 50 for the sky, 391 for foliage, and 19 for buildings. In the foliage image, the elbow was even above the break-even rank of 269.76. This shows that an elbow is useful for understanding the spectrum but is not necessarily a good compression rule on its own.

For practical selection of \(k\), I would use a predefined quality target such as relative error below 5% and then check whether the resulting rank remains below break-even. This method provides an explicit and reproducible quality requirement rather than choosing \(k\) visually.

## 4. Comparison Across Image Types

The smooth sky image compressed by far the best. It required only \(k=6\) to reach relative error below 5%, producing a compression ratio of 49.27×. This result is consistent with the structure of the image. Large neighbouring regions have very similar intensities, creating strong redundancy that can be represented with relatively few SVD components.

The building image required \(k=138\) for the same error target and achieved 2.48× compression. It contains smooth areas but also many boundaries, windows, straight lines and other geometric features. These edges require additional components compared with the sky.

The foliage image was the hardest to compress. It required \(k=206\) to achieve relative error below 5%, giving only 1.31× compression. Leaves, branches, shadows and fine texture create rapid local changes across the image. As a result, information is spread across a much larger number of singular values.

This difference was also visible in the singular-value spectra. The sky spectrum decayed rapidly, showing that a small number of components dominate. The foliage spectrum had a much longer tail, indicating that many components remained important. The building image was between these two cases.

## 5. Noise and SVD as a Denoising Method

Gaussian noise was added to the smooth sky image at standard deviations of 0.02 and 0.08. Adding noise raised the smaller singular values and made the spectrum flatter. This occurs because random noise is not strongly low-rank and contributes variation across many different directions.

This behaviour also suggests a use for truncated SVD as a denoising method. The relative error between the high-noise image and the original clean image was 0.0996. After reconstructing the noisy image at different ranks, the errors relative to the clean image were 0.0536 at \(k=5\), 0.0445 at \(k=10\), 0.0397 at \(k=20\), 0.0481 at \(k=40\), and 0.0632 at \(k=80\).

Among the tested ranks, \(k=20\) performed best. It reduced the error relative to the clean image by about 60% compared with the noisy observation. Very small \(k\) values remove noise but also remove genuine detail, while larger \(k\) values begin to reconstruct the noise again. This demonstrates that truncated SVD can perform denoising, but the rank must be selected carefully.

## 6. Limitations and Conclusion

Relative Frobenius error is useful because it is objective and reproducible, but it is not a perceptual metric. Two images with similar numerical errors may look different to a human observer. As an additional check, I calculated SSIM at the 5% error ranks. The SSIM values were 0.813 for the sky, 0.956 for foliage and 0.912 for buildings, demonstrating that numerical error and perceived structural similarity do not always behave identically.

SVD compression is also a poor choice when an image contains a large amount of texture or when the required \(k\) approaches the break-even rank. Unlike JPEG, SVD must store an image-specific \(U\) and \(V^T\) basis. JPEG uses a fixed transform, block processing, quantisation and entropy coding, making it much more practical for normal image storage.

Overall, the experiment shows that SVD is most effective when the data contains strong low-rank structure. Its main value here is not that it outperforms modern image codecs, but that it demonstrates how a dataset can be represented using a limited number of ranked components. This is the same basic idea that later appears in PCA.