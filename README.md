## Predicting Stock Market Movements Using LSTM
We analyzed the closing price returns to forecast future prices. Our model predicts the total return for the next ten days.
We have chosen this [**LSTM**](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)  it solves some common issues found in basic recurrent neural networks, like the vanishing gradients problem. 

## Dataset:
We've chosen three types of assets:
- Large caps
- Indices
- Comodities & Currencies

We've used data from 15 stock indices sourced from Yahoo! Finance and Onvista.
- Data Cleaning:
The z-score approach was used to eliminate outliers. Values with z-scores greater than 3 were excluded as outliers.
- Data Preparation:
To normalise our data between 0 and 1, we used the MinMax Scaler.\n
Split the data: 80% for training and 20% for testing.
- Features:
Our model uses the log returns of the closing prices.

## Model Details:
- Input: Uses the share prices from the previous 90 days as an input to forecast the 91st day.
- Layers: 2 layers. There are 70 neurons in the first layer and one neuron in the output layer.
- Activation Functions: Uses 'tanh' and 'sigmoid' internally.  For continuous data, the output layer uses 'linear' activation.
- Dropout: Set to 0.1 to prevent overfitting.
- Training: The model trains in 300 epochs with a batch size of 150.
- Optimization: Uses Mean Squared Error (MSE) as the loss function and the Adam optimizer.
- Evaluation:
Using data from both training and testing sets, we calculate the Root Mean Squared Error (RMSE) to assess the model's correctness.

## Installation

Open the command prompt, navigate to your specific folder and clone the repository inside a folder on your local repository. 

```bash
git clone https://github.com/KunduKamal/Deep-Neural-Network-to-predict-stock-market-up-and-down-movements.git
```
```bash
cd .\Deep-Neural-Network-to-predict-stock-market-up-and-down-movements\
```
Before proceeding, ensure you have Python installed. You can now set up a virtual environment.

```bash
python -m venv env
```
Activate the virtual environment:
```bash
env\\Scripts\\activate.bat
```
Install the requirements:
```bash
pip install -r requirements.tx
```
## Recommandation
To efficiently run this model, we suggest using Google Colab, as it offers GPU support.This will simplify the procedure and save time.

## References: 
- Yahoo! Finance - https://finance.yahoo.com/
- Onvista - https://www.onvista.de/
- Degiro - https://www.degiro.de/?_ga=2.53740846.768932194.1612563755-1041573335.1612445051
- “Stock Trading with Neural Networks” by ERIK IHRÉN, SEAN WENSTRÖM
- Christopher Olah - https://colah.github.io/posts/2015-08-Understanding-LSTMs/
- “Predicting the Trend of Stock Market Index Using the Hybrid Neural Network Based on Multiple Time Scale Feature Learning” by Yaping Hao and Qiang Gao

## Please note: 
Do not use this for actual trading. We are not responsible for any financial losses. The LSTM model was only used in this experiment to predict prices.