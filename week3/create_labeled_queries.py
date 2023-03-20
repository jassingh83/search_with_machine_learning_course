import os
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import csv
import re

# Useful if you want to perform stemming.
import nltk

def normalize_query(query):
    str_alpha_numeric = re.sub('[^0-9a-zA-Z]+', ' ', query.lower())
    stemmed_tokens = [stemmer.stem(token) for token in str_alpha_numeric.split()]
    return ' '.join(stemmed_tokens)

def get_min_key(category_query_count_dict, root_category_id):
    min_val = min(category_query_count_dict.values())
    min_keys = [category for category, query_count in category_query_count_dict.items() if (query_count == min_val) and (category != root_category_id)]
    return min_keys

stemmer = nltk.stem.PorterStemmer()

categories_file_name = r'/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'

queries_file_name = r'/workspace/datasets/train.csv'
output_file_name = r'/workspace/datasets/fasttext/labeled_queries_10000.txt'

parser = argparse.ArgumentParser(description='Process arguments.')
general = parser.add_argument_group("general")
general.add_argument("--min_queries", default=1,  help="The minimum number of queries per category label (default is 1)")
general.add_argument("--output", default=output_file_name, help="the file to output to")

args = parser.parse_args()
output_file_name = args.output

if args.min_queries:
    min_queries = int(args.min_queries)

# The root category, named Best Buy with id cat00000, doesn't have a parent.
root_category_id = 'cat00000'

tree = ET.parse(categories_file_name)
root = tree.getroot()

# Parse the category XML file to map each category id to its parent category id in a dataframe.
categories = []
parents = []
for child in root:
    id = child.find('id').text
    cat_path = child.find('path')
    cat_path_ids = [cat.find('id').text for cat in cat_path]
    leaf_id = cat_path_ids[-1]
    if leaf_id != root_category_id:
        categories.append(leaf_id)
        parents.append(cat_path_ids[-2])
parents_df = pd.DataFrame(list(zip(categories, parents)), columns =['category', 'parent'])

print(normalize_query("Beats By Dr. Dre- Monster Pro Over-the-Ear Headphones -"))

# Read the training data into pandas, only keeping queries with non-root categories in our category tree.
queries_df = pd.read_csv(queries_file_name)[['category', 'query']]
queries_df = queries_df[queries_df['category'].isin(categories)]

# IMPLEMENT ME: Convert queries to lowercase, and optionally implement other normalization, like stemming.
queries_df['query'] = queries_df['query'].apply(normalize_query)

# IMPLEMENT ME: Roll up categories to ancestors to satisfy the minimum number of queries per category.
# group by categories.
# get the count by categories.
# turn the df to a dictionary.
# find the minimum value in the dictionary and check to see if it is less than threshold.
# get the category for the minimum from the map key.
# get the parent category.
# add the parent category to the dictionary if it does not exist.
# add the number of elements in the category to the parent category.
# get all the queries for the category and create a new tuples with category as parent category and value as the query.
# remove the category from the group_by dictionary.
# remove all the queries with a given category from the categories df.

queries_group_by_df = queries_df.groupby('category')
queries_cat_by_count = queries_group_by_df.size().reset_index(name='count')
category_num_queries_dict = queries_cat_by_count.set_index('category')['count'].to_dict()

print(f"Count for abcat0701001: {category_num_queries_dict['abcat0701001']}")
min_count_categories = get_min_key(category_num_queries_dict, root_category_id)

iteration = 1
earlier_queries_df_len = len(queries_df)
print(f"total queries: {len(queries_df)}")
while (len(min_count_categories) > 0 and category_num_queries_dict[min_count_categories[0]] < min_queries):
    min_count_category = min_count_categories[0]
    print(f"Iteration: {iteration}, Min Category Count: {category_num_queries_dict[min_count_category]}")
    parent_result = parents_df[parents_df['category'] == min_count_category]
    if not parent_result.empty and len(parent_result['parent'].values) > 0:
        parent_category = parent_result['parent'].values[0]
        if parent_category not in category_num_queries_dict:
            category_num_queries_dict[parent_category] = 0
        category_num_queries_dict[parent_category] += category_num_queries_dict[min_count_category]
        queries_min_category = queries_df[queries_df['category'] == min_count_category]['query'].values
        queries_parent_category_tuples = [(parent_category, query) for query in queries_min_category]
        add_on_df = pd.DataFrame(queries_parent_category_tuples, columns=['category', 'query'])
        queries_df = pd.concat([queries_df, add_on_df])

        del category_num_queries_dict[min_count_category]
        queries_df = queries_df[queries_df['category'] != min_count_category]
        min_count_categories = get_min_key(category_num_queries_dict, root_category_id)
        iteration += 1

print(f"earlier total queries: {earlier_queries_df_len}")
print(f"total queries: {len(queries_df)}")
print(category_num_queries_dict)
print(sum(category_num_queries_dict.values()))
# print(queries_df.head())
print(f"unique values in map: {len(category_num_queries_dict)}")
print(f"nunique: {queries_df['category'].nunique()}")

# Create labels in fastText format.
queries_df['label'] = '__label__' + queries_df['category']

# Output labeled query data as a space-separated file, making sure that every category is in the taxonomy.
queries_df = queries_df[queries_df['category'].isin(categories)]
queries_df['output'] = queries_df['label'] + ' ' + queries_df['query']
queries_df[['output']].to_csv(output_file_name, header=False, sep='|', escapechar='\\', quoting=csv.QUOTE_NONE, index=False)
