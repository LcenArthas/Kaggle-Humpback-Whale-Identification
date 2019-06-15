# Kaggle-Humpback-Whale-Identification
This is my first kaggle competition,and the final score is 120/2131, Top 6%.

The baseline is the MGN, and you can find it in this https://arxiv.org/abs/1804.01438

And I also ueed the whale bounding boxes by the Kaggle kernal: https://www.kaggle.com/suicaokhoailang/generating-whale-bounding-boxes

First you can used the file pre_crop to make the new-tsetset and the new-trainset, also include the queryset.

Than train this model by main.py

After trained, you may test your modle by output.py. Be careful you have to do the following:

* set the modle_load way.

* set the new_whale thread.

Then the result will out!

P.S.
I also write the vote.py to sensemble this result.And the compute_new.py to comput the ratio of new whales.
