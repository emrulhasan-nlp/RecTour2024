import pandas as pd
import time
from collections import defaultdict

def rank_reviews_by_score(review_tuples, top_n=10):
    # Convert the second entry in each tuple to a float for proper numerical comparison
    review_tuples = [(review_id, float(score)) for review_id, score in review_tuples]
    
    # Sort the list of tuples by score in descending order
    ranked_reviews = sorted(review_tuples, key=lambda x: x[1], reverse=True)
    
    # Extract only the top `n` review IDs in ranked order
    top_review_ids = [review_id for review_id, _ in ranked_reviews[:top_n]]
    
    return top_review_ids


##########Review ranker #########
def review_rank(user_data_path,review_data_path, save_path):
    # Preprocess accommodation data
    rev_df = pd.read_csv(review_data_path)
    user_df = pd.read_csv(user_data_path)
    
    # create a dictionary with accommodation id and review profile
    acco_id = defaultdict(list)
    for _, row in rev_df.iterrows():
        
        acco_id[row['accommodation_id']].append((row['review_id'], f"{row['review_helpful_votes']}"))

    # Final dictionary with all the required fields.
    result_dict = defaultdict(list)
    for _, row in user_df.iterrows():
        result_dict['accommodation_id'].append(row['accommodation_id'])
        result_dict['user_id'].append(row['user_id'])
        
        votes=acco_id[row['accommodation_id']]
        
        rank_rev=rank_reviews_by_score(votes, top_n=10) #######Ranked reviews
        
        #####create columns with each review id ##########    
        for i in range(10):
            result_dict[f'review_{i+1}'].append(rank_rev[i] if i < len(rank_rev) else None)

    df = pd.DataFrame(result_dict)
    df.to_csv(save_path, index=False)
    print(df.info())


if __name__ == '__main__':
    #copmute the time
    start_time = time.time()
    
    #run_version='run1' # In this case, we use only helpfulness vote
    run_version='run2' # In this case, we use only review score
    user_data_path="./datasets/test_users.csv"
    review_data_path="./datasets/test_reviews.csv"
    save_path=f"./results/{run_version}.csv" # save path 
    review_rank(user_data_path,review_data_path, save_path)
    end_time = time.time()

    hours = (end_time - start_time) // 3600
    minutes = (end_time - start_time) % 3600 // 60

    print(f"Execution time: {hours} hours, {minutes} minutes")