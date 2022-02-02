# nbNMF
nbNMF is an easy-to-use Python library for robust dimensionality reduction with count data.

## Background
*Non-negative matrix factorization (NMF)* is a dimensionality reduction method where we find two non-negative, low-dimensional matrices `W` and `H`  such that their product `WH` approximates a given data matrix `X`. For this method to work, we need a way to measure the distance (a *loss function*, in the machine learning lingo) between the input matrix `X` and the approximation `WH`.

The default choice here is simple quadratic distance. Statistically, this corresponds to maximum likelihood estimation under a normal distribution. For count data, a discrete distribution such as Poisson makes more sense, and that leads to a modified KL-divergence.

The issue with the Poisson distribution, however, is that it implies that conditional variances equal conditional means, which can be a unreasonable assumption  in practice. **nbNMF offers a simple alternative to Scikit-Learn's NMF, accounting for overdispersion using a negative binomial distribution**.

This is similar to what [Gouvert, Oberlin and Févotte](https://arxiv.org/abs/1801.01708) proposed (in particular, the multiplication updates are the same). An important difference is that this implementation estimates the overdispersion parameter from the data itself, instead of requiring a hyperparameter.

## Installation
Available soon!

## Usage
This package was designed to be as similar to [scikit-learn's original NMF implementation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html) as possible:
```python
from nbNMF import nbNMF

# the input X is a 2D Numpy array or Pandas dataframe with count (integer) data
model = nbNMF(n_components=10)
X_transformed = model.fit_transform(X)
```

## Roadmap
- [x] Basic NMF with negative-binomial divergence
- [x] L1/L2 regularization
- [x] Unit tests
- [ ] Packaging
- [ ] Examples
- [ ] Documentation

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)