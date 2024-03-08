# Beginner-s_Hypo_2y-
My Attempt at solving the Beginner's Hypo for 2y 2024

The first few days of the challenge were spent in understanding the challenge, how the code works and what the individual components do, and also in learning about the CVAE model working.

Variational Auto-Encoders work through probabilistic encoding, to embody the input as a probability distribution, usually a Gaussian Distribution. Conditional VAEs do the same, but also take input labels as well and thus allow us some control on the output generated.

The loss function used is the Kullback-Leibler Divergence Loss for evaluating the latent space, and MSE loss for measuring reconstruction.

Then I used the cvae_example file to get some results working with tier_1 dataset to gain some insights.

Setting KLD loss weight to zero : Everything seems to converge for X, but not very fast. KLD loss is very high . Even when trained once, and then setting Beta to zero, it starts exploding. KLD Loss ensures that a good latent space is established. Setting it to zero does not make sense.

Latent_dims : This dictates how well the hidden features in the data are captured by the latent space, however, more latent_dims does not directly correlate to better performance. On tier_1 data, a model with 50 latent_dims performed better than one with 100 latent dims.

Then I started experimenting with the number of layers in the model, and added some linear layers to both encoder and decoder, and added a convolutional layer and conv_transpose layer to the Decoder, this result was V2-cvae_1.
Best Result of V2 : (100 Epochs)
----- | 100 Epochs |  
----- | ----- |  
Recon 1 Err | 1.0614171028137207  
Recon 2 Err | 0.6028445363044739   
Lin1 Score  | 0.8542283187068859  
Lin2 Score  | 0.782013635020141  

After this, I decided to do away with the convolutional layers in the encoder and decided to stick to only Linear layers in the encoder. The same could not be done with the decoder as decoder does upsampling, and removing the conv_transpose layers is counter-intuitive to that idea. This file is V3 Linear_1.
Result of V3 : (In Ascending Order of Epochs)  
----- | 100 Epochs | 200 Epochs | 300 Epochs | 400 Epochs | 500 Epochs | 600 Epochs |  
------ | ------ | ------ | ------ | ------ | ------ | ------ |  
Recon 1 Err | 1.0721356868743896 | 1.0819694995880127 | 1.013979434967041 | 0.9897443652153015 | 1.0338083505630493 | 1.0455443859100342 |  
Recon 2 Err | 0.6740265488624573 | 0.6764230132102966 | 0.737133800983429 | 0.7457465529441833 | 0.7296338081359863 | 1.027361512184143  |  
Lin1 Score  | 0.9257214689849004 | 0.8895569490175673 | 0.858702329288658 | 0.8489043480160983 | 0.8376032983277392 | 0.8103843703751765 |  
Lin2 Score  | 0.8418052521232774 | 0.8559079916990848 | 0.853090243119385 | 0.7793806294125724 | 0.7823109394032953 | 0.7516764684647277 |  

V4 Beyond_Linear uses the same encoder from Linear_1, but has more convolutional layers in the decoder. This model gave the highest scores after training for only a 100 epochs, but the performance dropped after more training.
Best Result of V4 : (In Ascending Order of Epochs)  
------ | 100 Epochs | 200 Epochs | 300 Epochs |  
------ | ------ | ------ | ------ |  
Recon 1 Err | 0.9940018653869629 | 1.0420992374420166 | 1.0037682056427002 |
Recon 2 Err | 0.6178206205368042 | 0.6497029066085815 | 0.1.009466528892517 |
Lin1 Score  | 0.8572659590057902 | 0.8281206818710691 | 0.8188335282488488 |
Lin2 Score  | 0.7789309525434085 | 0.8098275403682743 | 0.7757711640056959 |

Skip is a model that just implemented Skip connections to check how well that worked.
Best Result of Skip :  (In Ascending Order of Epochs)  
---- | 100 Epochs | 200 Epochs |
------ | ------ | ------ |  
Recon 1 Err | 1.0030987262725830 | 1.0190328359603882 
Recon 2 Err | 0.6736232042312622 | 0.7300440073013306 
Lin1 Score  | 0.8202953261645757 | 0.7651719968069316 
Lin2 Score  | 0.7887654951677125 | 0.7419151046068092 

Putting together these models and their results, it seems that cvae_1’s encoder works best, and Beyond_Linear’s decoder. This is visible due to the fact that the linear encoder and models that used it had a poorer performance than cvae_1, except for Beyond_Linear. Combining these and adding some dropouts for regularization, the final model Best_model has been made.
Best Result of V5 :  (In Ascending Order of Epochs)  
---- | 100 Epochs | 200 Epochs | 300 Epochs | 400 Epochs |  
------ | ------ | ------ | ------ | ----- |  
Recon 1 Err | 1.0489706993103027 | 1.0676474571228027 | 1.0555456876754760 | 0.9890109300613403 |  
Recon 2 Err | 0.6596177220344543 | 0.6179133057594299 | 0.5726384520530701 | 0.6016169786453247 |  
Lin1 Score  | 0.9027096487851273 | 0.9031888720024863 | 0.8918424494237588 | 0.8593957102081423 |  
Lin2 Score  | 0.8711553901410075 | 0.8870686267839023 | 0.8729499002831363 | 0.8702589040594421 |  

Now to cover the coverage score as well, I added a combination of reciprocal of Euclidean distance similarity as closeness penalty, and an even distribution penalty to the loss and trained Best_model on this loss function. 
Final Result of V5 :  (Tier 2 then Tier 1, Using the new Loss Function)  
----- | Tier-1 | Tier-2 |
------ | ------ | ------ |  
Recon 1 Err | 0.9764147400856018 | 0.6431618928909302 |  
Recon 2 Err | 0.5468675494194031 | 0.7007080316543579 |  
Lin1 Score  | 0.8394270455688785 | 0.7561864140551667 |  
Lin2 Score  | 0.8465821156579041 | 0.6104851900220721 |  

I have also tried cosine similarity, but it did not work as well.

All models have better scores mostly at 100 epochs of training, which is suspected to be overfitting. Only the Best model that has regularization has not overfit and has better performance with more training. 

All the .pt files in the respective folders have been trained for the Tier-2 task.

I also tried to make and optimize a CGAN, but did not know how to write an optimal loss function for it.

## To run the notebook 
Import notebook to kaggle, and import these 3 datasets, which I have made for my ease  
Or you could just load the notebook from here.  

https://www.kaggle.com/vshashankbharadwaj/beginnner-s-hypo-cvae  

https://www.kaggle.com/datasets/vshashankbharadwaj/hypo-stuff  

https://www.kaggle.com/datasets/vshashankbharadwaj/hypo-stuff-tier-1  

https://www.kaggle.com/datasets/vshashankbharadwaj/hypo-stuff-tier-2  

The latter 3 are datasets containing all the files required to run the notebook
