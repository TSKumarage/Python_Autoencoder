hyper_parameters = {'ntrees':[10,50], 'max_depth':[20,10]}
grid_search = H2OGridSearch(H2ORandomForestEstimator, hyper_params=hyper_parameters)
grid_search.train(x=["x1", "x2"], y="y", training_frame=train)
grid_search.show()