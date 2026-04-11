## TODO

### Data normalization

1. For each of the 4 datasets, create an Enriched_DF_PL version of the dataset, which includes all the typehints

- for each dataset class, include the load_from_parquet() method

2. Implement a load_all method in CDNNBAMemoPL that calls each of the individual load_from_parquet() methods and stores the results in the inputs dict. This will allow us to load all datasets at once when we create an instance of CDNNBAMemoPL.

3. create a memo_series for each dataset that maps the dataset to the core cdnnba dataset. Essentially the finished result of the merge process, but as a memo_series that can be used in the rest of the codebase. This will allow us to easily access the merged dataset in our analysis

4. Ensure that this works correctly, write a notebook in Notebooks/ that demonstrates loading the data and aligning it to the cdnnba spine. This will also ensure that these operations are actually legal
