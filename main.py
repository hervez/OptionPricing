import datetime

from Analysis import OptionAnalysis
from DataGathering import OptionDataGathering

def command_interface():

    print("######################################################################################################")
    gatherer = OptionDataGathering(verbose=False)

    symbol = input("Enter the symbol of the underlying asset (default AAPL):") or "AAPL"
    evaluation_date = datetime.datetime.today().strftime("%Y-%m-%d")
    available_dates = gatherer.get_dates_available_option(symbol)[1:]
    for i, date in enumerate(available_dates, start =1):
        print(f"{i}. {date}")
    date_index = (int(input("Enter the numer of the date of interest (default 1:") or 1)) - 1
    expiration_date = available_dates[date_index]

    analyser = OptionAnalysis(underlying=symbol, expiration_date=expiration_date)
    analyser.complete_analysis()

if __name__ == "__main__":

    command_interface()


    # analyser = OptionAnalysis(underlying='AAPL', expiration_date='2023-06-02')
    # analyser.complete_analysis()
    # analyser = OptionAnalysis(underlying='AAPL', expiration_date='2023-05-26', evaluation_date='2023-05-25')
    # analyser.complete_analysis()
