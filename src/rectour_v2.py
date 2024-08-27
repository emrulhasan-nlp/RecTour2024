"""This ranks the reviews based on user preference for accommodation"""
import pandas as pd
import json
import time
from collections import defaultdict
import torch
from sentence_transformers import SentenceTransformer, util

#########Similarity score calculator #####################
def Simscore_calculator(model, user_prof, rev_prof):
    """Calculate similarity score between user profile and review"""
    user_emb=model.encode(user_prof, convert_to_tensor=True, device=device)
    rev_emb=model.encode(rev_prof, convert_to_tensor=True, device=device)
    
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
def review_rank(model,data_path, save_path):
    """The method accepts csv file containing user_id, accommodation_id, review_id and features. The goal is the rank the reviews for each user and accommodation.

    Args:
        data (csv): csv file containing user_id, accommodation_id, review_id and features
        save_path (csv): file will contain user_id, accommodation_id, and review_1...review_10 (ranked).
    """
    # Preprocess accommodation data
    df = pd.read_csv(data_path)
    acco_id = defaultdict(list)
    for _, row in df.iterrows():
        acco_id[row['accommodation_id']].append((row['review_id'], f"{row['review_title']} {row['review_positive']} {row['review_negative']}"))

    result_dict = defaultdict(list)
    for _, row in df.iterrows():
        result_dict['accommodation_id'].append(row['accommodation_id'])
        result_dict['user_id'].append(row['user_id'])
        
        user_prof = f"{row['guest_type']} {row['guest_country']} {row['room_nights']} {row['accommodation_type']}"
        
        rev_score=[{rev_id:Simscore_calculator(model, user_prof, rev_prof)} for rev_id, rev_prof in acco_id[row['accommodation_id']]]
        rank_rev=SortedReview(rev_score)

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
    
    model_name='sbert_mini_v2'
    data_path="./datasets/cleaned_valdata.csv"
    save_path=f"./datasets/{model_name}.csv"
    review_rank(model,data_path, save_path)
    end_time = time.time()

    hours = (end_time - start_time) // 3600
    minutes = (end_time - start_time) % 3600 // 60

    print(f"Execution time: {hours} hours, {minutes} minutes")