# ift6266h17
Deep Learning Course Repository: Image In-Painting using DCGAN and lemmatized captions  
[Course](http://ift6266h17.wordpress.com) Instructor: [Prof. Aaron Courville](https://mila.umontreal.ca/en/person/aaron-courville/) @ [MILA](https://mila.umontreal.ca/en/), Winter 2017 Session

Please refer to my [blog](https://vderm.wordpress.com) for more information.

# Description
There are two files in this repo. I'm not a software developer, so I didn't neatly split the code into different folders. I rather have everything in one file to ease troubleshooting. When the time comes and I need to build more neural nets, I'll split things up.

The model file =dcgan_doc2vec_embed.py= is the approach I used for this course. The paths at the very top of the file need to be changed to reflect the directories on your machine. The rest of the settings can be called via the command line and/or through a batch file. An example is given, see =batch.sh=.

The script will display training images at a given interval and print out the training error per epoch. When the training is complete, the models can be saved and a validation image will be displayed. The user can optionally not quit the script and remain in an interactive mode to plot more examples or continue training. The training error information is saved to a CSV file in the script directory. Images can optionally be saved in the script directory.

The model uses the [doc2vec](http://radimrehurek.com/gensim/models/word2vec.html) encoding using the gensim package for the captions (doc2vec is an augmented word2vec model that embeds whole sentences or documents). Since the caption dataset is relatively small, I'm applying [lemmatization](http://textminingonline.com/dive-into-nltk-part-iv-stemming-and-lemmatization) using the NLTK package to keep only the root of the word and thus augment the data set.

I'm also giving a shot at trying the [Wasserstein-GAN](https://arxiv.org/abs/1701.07875). I might be able to have the code ready, but might not have results by the project deadline. I'll post it here nevertheless.


Vasken Dermardiros
