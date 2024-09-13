# RecTour 2024 Challenge
In this challenge, we propose a sentence-transformer based feature extraction method for review ranking. We show that instead using traditional "helpfullness vote", integrating linguistic features enhance the performance of the review ranking. We evaluate the performance using MRR@10 and Precision@10.

# Implemetation Details
0. pip install requirements.txt
1. This verson of the scritp uses only helpfulness votes and to run this uncomment line 53
-run: python rectour_v0.py
2. This verson of the scritp uses only review score and to run this uncomment line 53
-run: python rectour_v0.py 
3. This verson of the scritp uses the following features for user and accommodation profile
--accommodation profile: 'review_positive, 'review_negative','review_score','review_helpful_votes'
--user profile: 'guest_type','guest_country', 'accommodation_type','accommodation_country','location_is_ski','location_is_beach','location_is_city_center'
-run: python rectour_v1.py