# COVID-19-Prediction
This is the code for the A machine learning based forecast model for the COVID-19 pandemic and investigation of the impact of government intervention on COVID-19 transmission in China paper submission.

The data/extract repository stores all the data we used. The data were extracted from the Baidu Index. The data covers from 2019.01.01 to 2019.03.31. The cities we analyzed are the those whose the number confirmed cases is the top-101, including the Wuhan.

The code repository stores all the codes we used. compute_risk.m computes the internal risk and the external risks for time series analysis. train.m trains a Backward Propagation Neural Network (BPNN) to estimate the increasing cases trend. infer.m applied the model obtained in train.m to estimate the confirmed cases trend if all the indices double.

The data visualization code is not uploaded yet because of the confidential issue. What's more, the cities are also masked.
