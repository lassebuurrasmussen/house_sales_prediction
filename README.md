Welcome! In [this notebook](main.ipynb) I look at a data set with houses and a range of features.
I will do some exploratory data analysis to learn a bit about the trends in the data.
Then I will try to make a regressor to predict the prices of the houses based on the features.

The point of [this notebook](main.ipynb) is to illustrate my capabilities (and hopefully not lack thereof).
For that reason I will go very through the process very pedagogically.
So apologies to those who'd like for this to be more fast-paced :-)

So for those of you who'd like to follow along, go ahead and install Anaconda if you don't have it already.
You can download it from https://www.anaconda.com/distribution/.
Now go ahead and make a fresh conda environment:

`conda create -yn house_prices`

And let's install some of the packages that we will need for sure.

`conda install -yn house_prices pandas numpy seaborn scikit-learn`

And finaly we can go ahead and activate the environment.

`conda activate house_prices`

Now. Let's get started. Open [main.ipynb](main.ipynb) and let's predict some house prices!