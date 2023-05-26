from Analysis import OptionAnalysis

if __name__ == "__main__":
    analyser = OptionAnalysis(underlying='AAPL', expiration_date='2023-05-26', evaluation_date='2023-05-25')
    analyser.complete_analysis()