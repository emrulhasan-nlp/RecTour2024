import pandas as pd

def data_prep(match_file, user_file, review_file, save_file):
    """_summary_

    Args:
        match_file (csv): This file contains all the matched user, accommodation and review
        user_file (csv): detail about users
        review_file (csv): detail about reviews
        save_file (csv): _where to save the cleaned csv file
    """

    match_df=pd.read_csv(match_file)
    user_df=pd.read_csv(user_file)
    review_df=pd.read_csv(review_file)
    
    #merge match and user file
    match_user=pd.merge(match_df,user_df, on='user_id',how='inner')
    
    ## merge all three
    user_review_df=pd.merge(match_user,review_df, on='review_id',how='inner')

    user_review_df.drop(columns=['accommodation_id_x', 'accommodation_id_y'],inplace=True)

    user_review_df.to_csv(save_file, index=False)
    print(user_review_df.info())

if __name__ == '__main__':
    # This code will only run if the script is executed directly
    data_type='train' 
    #data_type='val'
    
    if data_type:
        match_file = './datasets/train_matches.csv'
        user_file = './datasets/train_users.csv'
        review_file = './datasets/train_reviews.csv'
        save_file = './datasets/cleaned_traindata.csv'
    
    if data_type:
        match_file = './datasets/val_matches.csv'
        user_file = './datasets/val_users.csv'
        review_file = './datasets/val_reviews.csv'
        save_file = './datasets/cleaned_valdata.csv'
    
    data_prep(match_file, user_file, review_file, save_file)
    print("Data preparation completed and the cleaned data is saved successfully!")
    