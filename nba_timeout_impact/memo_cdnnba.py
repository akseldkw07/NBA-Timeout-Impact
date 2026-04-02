"""
Ok lets go with cdnnba data

I want you to fill in CDNNBAPipelieHelper with two main functions:
1) load_and_stack_all: loads all seasons of cdnnba data, both regular and playoff, and stacks them into a single dataframe. It should also add a column indicating whether the row is from regular season or playoffs. save this to parquet
2) clean_stacked: loads the stacked dataframe from parquet and performs a cleaning pipeline on it. This should include the relevant parts of clean_pipeline_nbastatsv3 (but don't blindly copy paste, figure out what's different about this dataset). lets go from pandas to parquet for the processing to speed this up

Run the pipeline yourself and make sure it works, then:

1) add the relevant imports to nb_imports.py
2) implement a minimal memo_cdnnba.py that has the same structure as memo_nbastatsv3.py, but only implement the loader and enriched dataset typehints
3) create a new notebook cdn-nba-pipeline.ipynb where you provide cells that run the pipeline. model it after explore_cdnnba.ipynb but instead of exploring, just run the pipeline and show the head of the cleaned dataframe at the end
4) use the built in functions in kretsinger/ wherever possible
"""
