import csv
import json
import sys
import os
import pandas as pd
import numpy as np


def json2csv(json_path, csv_path):
    # 打开business.json文件,取出第一行列名
    with open(json_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            line_contents = json.loads(line)
            headers = line_contents.keys()
            break

    with open(csv_path, 'w', newline='', encoding='utf-8') as fout:
        writer = csv.DictWriter(fout, headers)
        writer.writeheader()
        with open(json_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                line_contents = json.loads(line)
                # if 'Phoenix' in line_contents.values():
                writer.writerow(line_contents)

    df_bus = pd.read_csv(csv_path)
    if csv_path == 'Yelp/business.csv':
        df_reduced = df_bus.drop(['state', 'postal_code', 'is_open', 'attributes', 'hours'], axis=1)
    elif csv_path == 'Yelp/user.csv':
        df_reduced = df_bus.drop(['useful', 'funny', 'cool', 'elite', 'average_stars', 'compliment_hot',
                                  'compliment_more', 'compliment_profile', 'compliment_cute', 'compliment_list',
                                  'compliment_note', 'compliment_plain', 'compliment_cool', 'compliment_funny',
                                  'compliment_writer', 'compliment_photos'], axis=1)
    else:
        df_reduced = df_bus.drop(['useful', 'funny', 'cool', 'text'], axis=1)

    df_cleaned = df_reduced.dropna()
    df_cleaned.to_csv(csv_path, index=False)


if __name__ == '__main__':
    json_business_path = 'Yelp/yelp_academic_dataset_business.json'
    json_user_path = 'Yelp/yelp_academic_dataset_user.json'
    json_review_path = 'Yelp/yelp_academic_dataset_review.json'

    csv_business_path = 'Yelp/business.csv'
    csv_user_path = 'Yelp/user.csv'
    csv_review_path = 'Yelp/review.csv'

    json2csv(json_business_path, csv_business_path)
    json2csv(json_user_path, csv_user_path)
    json2csv(json_review_path, csv_review_path)
