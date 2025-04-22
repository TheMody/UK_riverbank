# UK_riverbank

What could solve Outlier detection here the most sensibly?

TabPFNv2 fails completely, time series from single measuring places do not provide enough data and the detected outliers are completely arbitrary. Furthermore, the outliers method seems highly unstable and often just throws an error without a good reason.

Rolling Z Score abs(x‑μ₅₀)/σ_IQR > 3 should be interesting

Maybe try adding more examples to TabPFN at the moment context is a bit unreasonably small.

A solution could be to use some kind of tokenization for the input data, this would also handle missing data (TabPFN just set missing data to 0) a very common occurrence in the dataset.

The tokenization could use one-hot to tokens for all categorical features, but i am not sure what to use for regression? The timestamp could be used as either another feature, or more appropriately, it could be used to generate the positional encoding. I am not sure what makes more sense.

Then a model could be trained using autoregressive generation or MLM. autoregressive i think makes more sense due to time series nature, no forward information should be known.

With the trained model we would automatically get uncertainty estimates for all generated tokens. High uncertainty could be marked.

The problem with this solution is maybe that the amount of data we have is too small, for a complex model, but i am not sure.
