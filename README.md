# feature-extraction-using-DL


V0_01.pt is the saved model for the LSTM_autoencoder_V_0.py file. It takes as input normalized data (MinMax Scaled)

I extracted a single simulation, and converted the shape from `1,6144,19` to `1,96,64,19`. Now, I can treat this as an image dataset for 19 channels and plot images for each channel (1-19). The visualize_predictions.ipynb contains the Heatmap visualizations of the inputs and the autoencoder outputs
