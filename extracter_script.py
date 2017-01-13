from helpers.extracter import Spider
import sys

# download the data from https://snap.stanford.edu/data/web-FineFoods.html and save as raw_data/food_raw.txt.
# The provided food_raw.txt is a placeholder. Also make sure that extracted_data directory exists.
# python extracter_script.py raw_data/finefoods.txt extracted_data/review_summary.csv
args = sys.argv
inputfile = args[1]
outputfile = args[2]

num_reviews = 200000
spider = Spider(num_reviews)
spider.crawl_for_reviews_and_summary(inputfile)
spider.save_review_summary_frame(outputfile)
