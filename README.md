# Mean-Shift-Segmentation
Segmentation of image regions using Mean Shift algorithm

Parametric segmentation techniques such as Optimal Thresholding are faced with mainly two problems - (1)
assumption that gray levels follow Gaussian distribution and (2) the need to compute a number of parameters
of a pre-determined probability density function. 

Mean Shift Segmentation is an elegant technique which over-
comes these problems by avoiding estimation of probability density function altogether. It is a non-parametric
technique which can analyze complex multi-modal feature space and works by iterative shifts of each data point
towards it's local mean.

Please refer the [report.pdf](https://github.com/adrsh18/Mean-Shift-Segmentation/blob/master/report.pdf) for further details.

To see a demo of this implementation, openup [mean_shift_test.ipynb](https://github.com/adrsh18/Mean-Shift-Segmentation/blob/master/mean_shift_test.ipynb)

To see how it was implemented, openup [mean_shift.ipynb](https://github.com/adrsh18/Mean-Shift-Segmentation/blob/master/mean_shift.ipynb)
