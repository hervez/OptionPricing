# OptionPricing

This project aim to compare different pricing methodologies to estimate the price of vanilla options. 
It takes as input the name of a stock as well as an evaluation date and an expiration date for the options. 
It then does the following: 

1. Download the data relative to all vanilla option of the specified underlying stock expiring at the expiration date from Yahoo Finance and cach them on the system. 
2. Compute the prices with different pricing methodologies. 
3. Create a pdf file containing the figures of the computed prices with the different methodologies and the actual options prices.

## Requirements 

Important: pdflatex is necessary to get a the pdf output of the program! 
For the libraries, see the requirements.txt file. 

## Features 

- Black and Scholes vanilla option pricing 
- Cox-Rox-Rubinstein vanilla option pricing 
- Fourier transformation option pricing 
- Fast Fourier transformation option pricing 
- Merton jump option pricing 
- Heston option pricing 
- Latex output 

## Example 

To run the program, you can simply set the appropriate symbol as well as the current date with the expiration date of interest in the method: 
OptionAnalysis(underlying='AAPL', expiration_date='2023-05-26', evaluation_date='2023-05-22').complete_analysis

## Configuration 

OptionPricing allow you to change the underlying stock as well as the expiration date (as long as it is available). 


