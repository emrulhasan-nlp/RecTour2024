import pandas as pd
import time
from collections import defaultdict
import torch
from sentence_transformers import SentenceTransformer, util

#########Similarity score calculator #####################
def Simscore_calculator(user_emb, rev_emb):
    """Calculate similarity score between user profile and review"""
    cosine_scores = util.pytorch_cos_sim(user_emb, rev_emb)
    return cosine_scores.item()

############sorting the review based on Similarity ##################
def SortedReview(data):
    """Sort reviews based on similarity scores"""
    review_dict = defaultdict(float)
    for d in data:
        for key, value in d.items():
            review_dict[key] += value
    
    sorted_items = sorted(review_dict.items(), key=lambda x: x[1], reverse=True)
    return [k for k, _ in sorted_items[:10]]

##########Review ranker #########
def review_rank(model,user_data_path,review_data_path, save_path):
    """The method accepts csv file containing user_id, accommodation_id, review_id and features. The goal is the rank the reviews for each user and accommodation.

    Args:
        data (csv): csv file containing user_id, accommodation_id, review_id and features
        save_path (csv): file will contain user_id, accommodation_id, and review_1...review_10 (ranked).
    """
    # Preprocess accommodation data
    rev_df = pd.read_csv(review_data_path)
    user_df = pd.read_csv(user_data_path)
    acco_id = defaultdict(list)
    for (_, row1), (_, row2) in zip(rev_df.iterrows(), user_df.iterrows()):
        acco_id[row1['accommodation_id']].append((row1['review_id'], f"{row1['review_title']}-{row1['review_positive']}-{row1['review_negative']}-{row2['accommodation_type']}-{row2['accommodation_country']}-{row2['location_is_ski']}-{row2['location_is_beach']}-{row2['location_is_city_center']}"))

    # Pre-encode all review titles
    all_reviews = [rev for reviews in acco_id.values() for _, rev in reviews]
    all_review_embeddings = model.encode(all_reviews, convert_to_tensor=True, device=device)

    review_embeddings = defaultdict(dict)
    idx = 0
    for acco, reviews in acco_id.items():
        for rev_id, _ in reviews:
            review_embeddings[acco][rev_id] = all_review_embeddings[idx]
            idx += 1

    result_dict = defaultdict(list)
    for _, row in user_df.iterrows():
        result_dict['accommodation_id'].append(row['accommodation_id'])
        result_dict['user_id'].append(row['user_id'])
        
        user_prof = f"{row['guest_type']}-{row['guest_country']}"
        user_emb = model.encode(user_prof, convert_to_tensor=True, device=device)
        
        acco = row['accommodation_id']
        rev_score = [{rev_id: Simscore_calculator(user_emb, rev_emb)} 
                    for rev_id, rev_emb in review_embeddings[acco].items()]

        rank_rev = SortedReview(rev_score)
        for i in range(10):
            result_dict[f'review_{i+1}'].append(rank_rev[i] if i < len(rank_rev) else None)

    df = pd.DataFrame(result_dict)
    df.to_csv(save_path, index=False)
    print(df.info())

if __name__ == '__main__':
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
    start_time = time.time()
    
    #Model name
    model_name='run5' ##### In this case, we combined the features from both user and review data for review profile, and user profile is 'guest_type' and 'guest_country'
    user_data_path="./datasets/test_users.csv"
    review_data_path="./datasets/test_reviews.csv"
    save_path=f"./results/{model_name}.csv"
    review_rank(model,user_data_path,review_data_path, save_path)
    end_time = time.time()

    hours = (end_time - start_time) // 3600
    minutes = (end_time - start_time) % 3600 // 60

    print(f"Execution time: {hours} hours, {minutes} minutes")