# HandwrittenNumbers
This is a Generative Adversarial Network (GAN) trained on the MNIST dataset. 
This python script can generate images that look like handwritten digits from 0 to 9.
The current training depth is set to very low, to be able to test run the GAN without too much computing power.
For actual implementation we up the training depth to a HIGH_VALUE such as 10000 and onwards for the GAN within
## Train the GAN
for epoch in range(HIGH_VALUE):
The higher the GAN training depth, the higher the generated outcome quality due to the higher training cycle depth reducing noise by the addition of fake image to avoid in the training data.
