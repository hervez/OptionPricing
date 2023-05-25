from Analysis import OptionAnalysis

if __name__ == "__main__":
    analyser = OptionAnalysis(underlying='AAPL', expiration_date='2023-05-26', evaluation_date='2023-05-22')
    analyser.complete_analysis()