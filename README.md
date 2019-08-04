# Learning Demographics from Text: Twitter Gender

Broadly, the intent of this project is to infer, from written text, information about the demographic profile of the writer.  The project as it stands seeks specifically to infer _gender_ from Twitter statuses (_tweets_).  

For purposes of model training and evaluation, I've defined gender operationally as the gender typically associated with the first word of someone's display name (as ascertained by the [gender-guesser](https://pypi.org/project/gender-guesser/) Python package).  In a sense, what I'm calling "gender" is really a projection of "name" onto a binary variable.  It will not correspond in all cases to either biological sex or personal gender identification, but nonetheless it seems like a meaningful and useful projection.  (For example, it seems more meaningful and useful than projecting "name" onto "first letter.")

...to be continued...

## Files

### Code

#### Intitial data processing
- get_tweets.ipynb
- process_tweets.ipynb
- aggregate_tweets.ipynb
- utils.py
- split_initial_tweet_corpus.ipynb

#### Baseline model using tuned USE-Large sentence embeddings
- twitgen_use_large_best.ipynb
- lr_poly_corpus_tweets.ipynb
- analyze_lr_poly_results.ipynb

#### Second model adding activations from LSTM network with tuned Glove embeddings
- save_glove_embeddings.ipynb
- twitgen_glovinit_best_dl_model.ipynb
- twitgen_glovinit_lstm_get_activations.ipynb
- twitgen_glovinit_lstm_save_model.ipynb
- lr_poly_corpus_tweets_try_add_lstm.ipynb

#### Complete model adding activations from max-pooling network
- twitgen_glovinit_best_pooling.ipynb
- twitgen_explore_words.ipynb
- lr_poly_with_lstm_and_pooled.ipynb

#### Processing data for online learning evaluation
- aggreagate_new_tweets.ipynb
- big_corpus.ipynb

#### Full model with online learning
- twitgen_use_large_make_embed.ipynb
- twitgen_lstm_full_corpus_activations.ipynb
- twitgen_pooling_full_corpus_activations.ipynb
- twitgen_online_learning.ipynb
