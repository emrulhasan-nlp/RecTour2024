"""This ranks the reviews based on user preference for accommodation"""
import pandas as pd
from tqdm import tqdm
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

def review_rank(model, data_path, save_path, batch_size):
    """
    Ranks reviews for each user and accommodation using batch processing.

    Args:
        model: The model used for similarity calculation
        data_path (str): Path to the CSV file containing user_id, accommodation_id, review_id, and features
        save_path (str): Path to save the output CSV file with ranked reviews
        batch_size (int): Number of rows to process in each batch
    """
    # Read the CSV file in chunks
    chunks = pd.read_csv(data_path, chunksize=batch_size)
    
    all_results = []
    acco_id = defaultdict(list)

    for chunk in tqdm(chunks, desc="Processing batches"):
        # Update acco_id dictionary
        for _, row in chunk.iterrows():
            acco_id[row['accommodation_id']].append((row['review_id'], f"{row['review_title']} {row['review_positive']} {row['review_negative']}"))

        result_dict = defaultdict(list)
        for _, row in chunk.iterrows():
            result_dict['accommodation_id'].append(row['accommodation_id'])
            result_dict['user_id'].append(row['user_id'])
            
            user_prof = f"{row['guest_type']} {row['guest_country']} {row['room_nights']} {row['accommodation_type']}"
            
            rev_score = [{rev_id: Simscore_calculator(model, user_prof, rev_prof)} for rev_id, rev_prof in acco_id[row['accommodation_id']]]
            rank_rev = SortedReview(rev_score)

            for i in range(10):
                result_dict[f'review_{i+1}'].append(rank_rev[i] if i < len(rank_rev) else None)

        all_results.append(pd.DataFrame(result_dict))

    # Combine all batch results
    final_df = pd.concat(all_results, ignore_index=True)
    
    # Save the final result
    final_df.to_csv(save_path, index=False)
    print(final_df.info())

if __name__ == '__main__':
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
    start_time = time.time()
    
    batch_size=30000
    model_name='sbert_mini_v3'
    data_path="./datasets/cleaned_valdata.csv"
    save_path=f"./datasets/{model_name}.csv"
    review_rank(model,data_path, save_path,batch_size)
    end_time = time.time()

    hours = (end_time - start_time) // 3600
    minutes = (end_time - start_time) % 3600 // 60

    print(f"Execution time: {hours} hours, {minutes} minutes")