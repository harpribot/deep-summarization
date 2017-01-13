from helpers.extracter import Spider

inputfile = 'raw_data/foods.txt'
outputfile = 'extracted_data/review_summary.csv'

num_reviews = 100000
spider = Spider(num_reviews)
spider.crawl_for_reviews_and_summary(inputfile)
spider.save_review_summary_frame(outputfile)
