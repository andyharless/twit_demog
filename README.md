# Learning Demographics from Text: Twitter Gender

Broadly, the intent of this project is to infer, from written text, information about the demographic profile of the writer.  The project as it stands seeks specifically to infer _gender_ from Twitter statuses (_tweets_).  

For purposes of model training and evaluation, I've defined gender operationally as the gender typically associated with the first word of someone's display name (as ascertained by the [gender-guesser](https://pypi.org/project/gender-guesser/) Python package).  In a sense, what I'm calling "gender" is really a projection of "name" onto a binary variable.  It will not correspond in all cases to either biological sex or personal gender identification, but nonetheless it seems like a meaningful and useful projection.  (For example, it seems more meaningful and useful than projecting "name" onto "first letter.")

The basic approach is to take principal components of sentence-level embeddings and fit these to a quadratic-logistic model to predict gender.  The embeddings (or, more generally, activations, if some don't meet the strict definition of embeddings) come from three separate models: 

1. A fine-tuned version Google's Universal Sentence Encoder (Large), which uses a transformer-based approach to embed sentences.

2. An LSTM network built on word-level embeddings initialized with Glove vectors pre-trained on Twitter.

3. A simple max-pooling network built on similarly initialized word-level embeddings. (Fine tuning of word-level embeddings in the max-pooling model is meant to capture word-level differences in male-female usage, which may be present even when the presumably more substantive differences captured by the other models are not).

## Files

### Code

#### Intitial data processing
- [get_tweets.ipynb](https://nbviewer.jupyter.org/github/andyharless/twit_demog/blob/master/code/get_tweets.ipynb): extract a set of relevant tweets from Twitter API stream
- [process_tweets.ipynb](https://nbviewer.jupyter.org/github/andyharless/twit_demog/blob/master/code/process_tweets.ipynb): examine one set of tweets
- [aggregate_tweets.ipynb](https://nbviewer.jupyter.org/github/andyharless/twit_demog/blob/master/code/aggregate_tweets.ipynb): extract relevant fields from tweet sets and aggregate into one file
- [utils.py](https://github.com/andyharless/twit_demog/blob/master/code/utils.py): functions used to process tweets
- [split_initial_tweet_corpus.ipynb](https://nbviewer.jupyter.org/github/andyharless/twit_demog/blob/master/code/split_initial_tweet_corpus.ipynb): split aggregated tweet data into train/valid/test sets

#### Baseline model using tuned USE-Large sentence embeddings
- [twitgen_use_large_best.ipynb](https://nbviewer.jupyter.org/github/andyharless/twit_demog/blob/master/code/twitgen_use_large_best.ipynb): fit tuned USE-Large model and save embeddings
- [lr_poly_corpus_tweets.ipynb](https://nbviewer.jupyter.org/github/andyharless/twit_demog/blob/master/code/lr_poly_corpus_tweets.ipynb): fit PCA-quadratic-logistic model on USE-L embedding features
- [analyze_lr_poly_results.ipynb](https://nbviewer.jupyter.org/github/andyharless/twit_demog/blob/master/code/analyze_lr_poly_results.ipynb): some rough model interpretation from looking at scored tweets

#### Second model adding activations from LSTM network with tuned Glove embeddings
- [save_glove_embeddings.ipynb](https://nbviewer.jupyter.org/github/andyharless/twit_demog/blob/master/code/save_glove_embeddings.ipynb): save Glove embeddings for training data words (for future tuning)
- [twitgen_glovinit_best_dl_model.ipynb](https://nbviewer.jupyter.org/github/andyharless/twit_demog/blob/master/code/twitgen_glovinit_best_dl_model.ipynb): fit tuned LSTM network and save final-hidden-layer activations
- [twitgen_glovinit_lstm_save_model.ipynb](https://nbviewer.jupyter.org/github/andyharless/twit_demog/blob/master/code/twitgen_glovinit_lstm_save_model.ipynb): fit tuned LSTM network and save weights
- [twitgen_glovinit_lstm_get_activations.ipynb](https://nbviewer.jupyter.org/github/andyharless/twit_demog/blob/master/code/twitgen_glovinit_lstm_get_activations.ipynb): get final-hidden-layer activations from LSTM model using saved weights
- [lr_poly_corpus_tweets_try_add_lstm.ipynb](https://nbviewer.jupyter.org/github/andyharless/twit_demog/blob/master/code/lr_poly_corpus_tweets_try_add_lstm.ipynb): fit PCA-quadratic-logistic with USE-L embeddings and LSTM activations

#### Complete model adding activations from max-pooling network
- [twitgen_glovinit_best_pooling.ipynb](https://nbviewer.jupyter.org/github/andyharless/twit_demog/blob/master/code/twitgen_glovinit_best_pooling.ipynb)
- [twitgen_explore_words.ipynb](https://nbviewer.jupyter.org/github/andyharless/twit_demog/blob/master/code/twitgen_explore_words.ipynb)
- [lr_poly_with_lstm_and_pooled.ipynb](https://nbviewer.jupyter.org/github/andyharless/twit_demog/blob/master/code/lr_poly_with_lstm_and_pooled.ipynb)

#### Processing data for online learning evaluation
- [aggreagate_new_tweets.ipynb](https://nbviewer.jupyter.org/github/andyharless/twit_demog/blob/master/code/aggreagate_new_tweets.ipynb)
- [big_corpus.ipynb](https://nbviewer.jupyter.org/github/andyharless/twit_demog/blob/master/code/big_corpus.ipynb)

#### Full model with online learning
- [twitgen_use_large_make_embed.ipynb](https://nbviewer.jupyter.org/github/andyharless/twit_demog/blob/master/code/twitgen_use_large_make_embed.ipynb)
- [twitgen_lstm_full_corpus_activations.ipynb](https://nbviewer.jupyter.org/github/andyharless/twit_demog/blob/master/code/twitgen_lstm_full_corpus_activations.ipynb)
- [twitgen_pooling_full_corpus_activations.ipynb](https://nbviewer.jupyter.org/github/andyharless/twit_demog/blob/master/code/twitgen_pooling_full_corpus_activations.ipynb)
- [twitgen_online_learning.ipynb](https://nbviewer.jupyter.org/github/andyharless/twit_demog/blob/master/code/twitgen_online_learning.ipynb)
