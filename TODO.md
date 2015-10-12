PCA calculates the full eigendecomposition even if we need only a few of the components. Fix this. Implement a PCA algorithm that will run on large matrices.
MultiFileArray indexing improvements. Right now we can only index with int or slices. 
    - Indexing with lists of indices, or numpy arrays
    - Boolean indexing???
Every algorithm should have the same interface: dataset (or input matrices), params. Make params a dictionary.
	- Do I want to force each algorithm to accept DataSet instances???

Algorithms to implement
- From Bishop's book
	- Least Squares (+Regularized) for Regression and Classification
	- Iterated Weighted Least Squares	
	- Linear Discriminant Analysis	
	- ML Gaussian for classification
	- Logistic Regression (implemented in neural_networks)
	- Neural networks backprop (implemented)
		- generalize to different types of hidden and output units?
	- Message Passing
	- Gaussian Processes
	- Nonparametric Bayesian models: Chinese Restaurant, Indian Buffet, Dirichlet Process
