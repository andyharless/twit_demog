# Utilities for Tweet Demographics Project



# From https://gist.github.com/timothyrenner/dd487b9fd8081530509c

#Gets the text, sans links, hashtags, mentions, media, and symbols.
def get_text_cleaned(tweet):
    text = tweet['text']
    
    slices = []
    #Strip out the urls.
    if 'urls' in tweet['entities']:
        for url in tweet['entities']['urls']:
            slices += [{'start': url['indices'][0], 'stop': url['indices'][1]}]
    
    #Strip out the hashtags.
    if 'hashtags' in tweet['entities']:
        for tag in tweet['entities']['hashtags']:
            slices += [{'start': tag['indices'][0], 'stop': tag['indices'][1]}]
    
    #Strip out the user mentions.
    if 'user_mentions' in tweet['entities']:
        for men in tweet['entities']['user_mentions']:
            slices += [{'start': men['indices'][0], 'stop': men['indices'][1]}]
    
    #Strip out the media.
    if 'media' in tweet['entities']:
        for med in tweet['entities']['media']:
            slices += [{'start': med['indices'][0], 'stop': med['indices'][1]}]
    
    #Strip out the symbols.
    if 'symbols' in tweet['entities']:
        for sym in tweet['entities']['symbols']:
            slices += [{'start': sym['indices'][0], 'stop': sym['indices'][1]}]
    
    # Sort the slices from highest start to lowest.
    slices = sorted(slices, key=lambda x: -x['start'])
    
    #No offsets, since we're sorted from highest to lowest.
    for s in slices:
        text = text[:s['start']] + text[s['stop']:]
        
    return text



def is_male_g(gd,name):
    if len(name)>0:
        return(gd.get_gender(name.split()[0])=='male')
    else:
        return(False)
        

def is_female_g(gd,name):
    if len(name)>0:
        return(gd.get_gender(name.split()[0])=='female')
    else:
        return(False)


def get_rows(df, ids, level=0):  # Select rows that correspond to list of ids
    select = [v in ids for v in df.index.get_level_values(level).values.tolist()]
    return(df[select])    


def balance(df, seed=0):
	import pandas as pd
	n_males = df.male.sum()
	n_females = df.shape[0] - n_males
	n_each = min(n_males, n_females)
	df_m = df[df.male].sample(n_each, random_state=seed)
	df_f = df[~df.male].sample(n_each, random_state=seed+1)
	xy = pd.concat([df_m, df_f]).sample(frac=1, random_state=seed+2)
	return(xy)

                
def balanced_split_by_time_and_id(df, train_frac=.6, val_frac=.2, seed=0):

	df = df.sort_index(level='time')

	n = df.shape[0]
	n_train = int(train_frac*n)
	n_val = int(val_frac*n)
	n_test = n - n_val - n_train

	train_ids = set(df.index.get_level_values(0)[:n_train])
	val_ids = set(df.index.get_level_values(0)[n_train:n_train+n_val]) - train_ids
	test_ids = set(df.index.get_level_values(0)[n_train+n_val:]) - train_ids - val_ids
	
	df_train = balance(get_rows(df.iloc[:n_train,:], train_ids), seed)
	df_valid = balance(get_rows(df.iloc[n_train:n_train+n_val,:], val_ids), seed+3)
	df_test = balance(get_rows(df.iloc[n_train+n_val:,:], test_ids), seed+6)
	
	return(df_train, df_valid, df_test)

	
