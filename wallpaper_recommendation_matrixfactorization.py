
# coding: utf-8

# In[1]:


#import libraries
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds


# In[2]:


# Reading useractivity cleaned data
con_dat = pd.read_csv('processeddata_neural.csv',nrows = 1500000, usecols = ['UsrAgentTokenAnum','RssAnum','RssTagAnum'])
con_dat['viewed'] = 1

# Reading wallpaper data
wallp_df = pd.read_csv('Raw\content-wallpaper.csv', 
                       usecols = ['rssanum','UniqueID','Tag','Viewed']).rename(columns = {'rssanum' : 'RssAnum'
                                                                                          ,'Viewed' : 'totalView'})
# Param setting for recommendation function
wallpaper_df = wallp_df[['RssAnum','UniqueID','Tag']]


# In[3]:


# Parameter settings
# value of k in matrix decomposition ( 20 ~ 100)
top_n_matrix = 100

# Number of recommendations
recommendations_num = 2

# Recommendation of which user ?
userID = 3012

# Predection value cut-off
pred_cutoff = 0


# In[4]:


wallpaper_df.head()


# In[5]:


# column UserId for useractivity dataframe
f1 = con_dat['UsrAgentTokenAnum'].unique()
f1 = pd.DataFrame({'UsrAgentTokenAnum':f1})
f1['userId'] = np.arange(1,(len(f1)+1))

m1 = pd.merge(con_dat,f1,on='UsrAgentTokenAnum', how='left')


# In[6]:


#Merging userId column in user dataframe
merged_data = pd.merge(m1,wallp_df, on='RssAnum',how='left')
merged_data = merged_data[['UsrAgentTokenAnum','userId','RssAnum','totalView','RssTagAnum','Tag']]


# In[7]:


merged_data.head(2)


# In[8]:


# User per wallpaper Hit count
t_df = merged_data.groupby(['userId','RssAnum']).size().reset_index(name = 'useridWallpHit')


# In[9]:


t_df.head(2)


# In[10]:


t_df.shape


# In[11]:


#Pivot of user data for totalView value
v_df = t_df.pivot_table(index = 'userId', columns = 'RssAnum', values = 'useridWallpHit').fillna(0)


# In[12]:


v_df.head(2)


# In[13]:


#Releasing Memory
del(con_dat)
del(f1)
del(m1)
del(t_df)


# In[14]:


# Matrix of pivot values
V = v_df.values

# Mean of user viewed
user_viewed_mean = np.mean(V, axis=1)

# totalView Deviation from Mean and keep shape same ( normalize by each user mean )
V_demeaned = V - user_viewed_mean.reshape(-1,1)


# In[15]:


##### Singular Value Decomposition #####
# U is the user “features” matrix
# Sigma is the diagonal matrix of singular values (essentially weights)
# Vt is the wallper “features” matrix
# k = top k matrix for lower rank approximation

U, sigma, Vt = svds(V_demeaned, k = top_n_matrix )


# In[16]:


#Releasing Memory
del(V)
del(V_demeaned)


# In[17]:


#since U, Vt are diagonal matrix so converting Sigma also to diagonal matrix before multiplication
sigma = np.diag(sigma)


# In[18]:


# Making prediction from the decomposed matrices
all_user_predicted_wallp = np.dot(np.dot(U, sigma), Vt) #+ user_viewed_mean.reshape(-1, 1)
preds_df = pd.DataFrame(all_user_predicted_wallp, columns = v_df.columns)


# In[19]:


#Releasing Memory
del(user_viewed_mean)
del(U)
del(sigma)
del(Vt)
del(all_user_predicted_wallp)
del(v_df)


# In[20]:


preds_df.head(2)


# In[21]:


def sorted_user_prediction(predictions_df, uid):
    
    # Get and sort the user's predictions
    user_row_number = uid - 1 # UserID starts at 1, not 0
    sup = (predictions_df.iloc[user_row_number])
    sup = pd.DataFrame(sup).rename(
        columns = {user_row_number: 'Predictions'}).sort_values(by ='Predictions', ascending=False)

    return sup.reset_index()

s_u_p = sorted_user_prediction(preds_df, userID)


# In[22]:


s_u_p.head(2)


# In[23]:


def user_viewed_wallp(uid, wallpap_df, original_user_df):
    
    # Get the user's data and merge in the wallpaper information.
    user_data = original_user_df[original_user_df.userId == (uid)]
    user_full = (user_data.merge(wallpap_df, how = 'left', left_on = 'RssAnum', right_on = 'RssAnum'))
                     #.sort_values(['totalView'], ascending=False))
    
    user_full.drop('Tag_x', axis = 1, inplace = True)
    user_full = user_full.rename(columns = {'Tag_y':'Tag'})

    return user_full

user_already_viewed = user_viewed_wallp(userID, wallpaper_df, merged_data)


# In[24]:


user_already_viewed


# In[25]:


def wallp_recommendation(wallp_raw, user_full, sorted_user_predictions):

    #print ('User {0} has already viewed {1} wallpapers.'.format(userID, user_full.shape[0]))
    #print ('Recommending the highest {0} predicted view wallpapers which is not already viewed.'.format(num_recommendations))
    
    # Recommend the highest predicted wallpapers that the user hasn't seen yet.
    sorted_user_predictions = sorted_user_predictions[(sorted_user_predictions[['Predictions']] >= pred_cutoff).all(1)]
    wallp_not_seen = wallp_raw[~wallp_raw['RssAnum'].isin(user_full['RssAnum'])]
    #merged_notSeen_userPre = (wallp_not_seen.merge(sorted_user_predictions, how = 'left',left_on = 'RssAnum', 
    #                                              right_on = 'RssAnum').sort_values('Predictions', ascending = False))
    pred_wallpRaw = sorted_user_predictions.merge(wallp_raw, how = 'left',left_on = 'RssAnum',right_on = 'RssAnum')
    pred_wallpRaw = pred_wallpRaw[['Tag', 'Predictions']]
    return pred_wallpRaw, wallp_not_seen

predicted_tag, not_seen = wallp_recommendation(wallp_df, user_already_viewed, s_u_p)


# In[26]:


not_seen.head(5)


# In[27]:


#Function to list top 'recommedation_count' wallpapers from predicted Genre

def list_top_wallpFromEachGenre(recom_tag, recom_wallp_list, recom_count=5):
    pred_Tag = recom_tag.Tag.unique()
    pred_wallp = pd.DataFrame()
    tag_count = len(pred_Tag)
    
    if tag_count < 3 :
        recom_count = 5
    elif tag_count < 5 :
        recom_count = 3
    elif tag_count < 7 :
        recom_count = 3
    else:
        recom_count = 2 
    
    for index, tag in enumerate(pred_Tag):
        pred_wallp1 = recom_wallp_list[(not_seen['Tag'] == pred_Tag[index])].sort_values('totalView',
                                                                                        ascending=False).head(recom_count)
    #pred_wallp1 = not_seen[(not_seen['Tag'] == pred_Tag[index])].sort_values('totalView', ascending=False).head(recom_count)
        pred_wallp = pd.concat([pred_wallp,pred_wallp1])
        
    return pred_wallp.reset_index(drop = True)

top_wallpFromEachGenre = list_top_wallpFromEachGenre(predicted_tag, not_seen, recommendations_num)    


# In[28]:


top_wallpFromEachGenre


# In[31]:


top_wallpFromEachGenre

