# Beginner-s_Hypo_2y-
My Attempt at solving the Beginner's Hypo for 2y 2024

The first few days of the challenge were spent in understanding the challenge, how the code works and what the individual components do, and also in learning about the CVAE model working.

Variational Auto-Encoders work through probabilistic encoding, to embody the input as a probability distribution, usually a Gaussian Distribution. Conditional VAEs do the same, but also take input labels as well and thus allow us some control on the output generated.

The loss function used is the Kullback-Leibler Divergence Loss for evaluating the latent space, and MSE loss for measuring reconstruction.

Then I used the cvae_example file to get some results working with tier_1 dataset to gain some insights.

Setting KLD loss weight to zero : Everything seems to converge for X, but not very fast. KLD loss is very high . Even when trained once, and then setting Beta to zero, it starts exploding. KLD Loss ensures that a good latent space is established. Setting it to zero does not make sense.

Latent_dims : This dictates how well the hidden features in the data are captured by the latent space, however, more latent_dims does not directly correlate to better performance. On tier_1 data, a model with 50 latent_dims performed better than one with 100 latent dims.

Then I started experimenting with the number of layers in the model, and added some linear layers to both encoder and decoder, and added a convolutional layer and conv_transpose layer to the Decoder, this result was cvae_1.
Best Result of CVAE 1 : 
Recon 1

After this, I decided to do away with the convolutional layers in the encoder and decided to stick to only Linear layers in the encoder. The same could not be done with the decoder as decoder does upsampling, and removing the conv_transpose layers is counter-intuitive to that idea. This file is Linear_1.

Beyond_Linear uses the same encoder from Linear_1, but has more convolutional layers in the decoder. This model gave the highest scores after training for only a 100 epochs, but the performance dropped after more training.

Skip is a model that just implemented Skip connections to check how well that worked.

Putting together these models and their results, it seems that cvae_1’s encoder works best, and Beyond_Linear’s decoder. This is visible due to the fact that the linear encoder and models that used it had a poorer performance than cvae_1, except for Beyond_Linear. Combining these and adding some dropouts for regularization, the final model Best_model has been made.

Now to cover the coverage score as well, I added a combination of reciprocal of Euclidean distance similarity as closeness penalty, and an even distribution penalty to the loss and trained Best_model on this loss function. I have also tried cosine similarity, but it did not work as well.

All models have better scores mostly at 100 epochs of training, which is suspected to be overfitting. Only the Best model that has regularization has not overfit and has better performance with more training. 

All the .pt files have been trained for the Tier-2 task.

I also tried to make and optimize a CGAN, but did not know how to write an optimal loss function for it.

## To run the notebook 
Import notebook to kaggle, and import these 3 datasets, which I have made for my ease

https://www.kaggle.com/datasets/vshashankbharadwaj/hypo-stuff

https://www.kaggle.com/datasets/vshashankbharadwaj/hypo-stuff-tier-1

https://www.kaggle.com/datasets/vshashankbharadwaj/hypo-stuff-tier-2

These 3 datasets contain all the files required to run the notebook
