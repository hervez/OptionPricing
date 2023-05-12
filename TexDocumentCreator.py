import subprocess
import pandas as pd
import os

class TexDocument():

    def __init__(self, underlying: str, evaluation_date: str, option_df: pd.DataFrame):

        self.underlying = underlying
        self.evaluation_date = evaluation_date
        self.option_df = option_df
        self.directory = f"./results/{self.underlying}/analysis"

    def generate_latex_code(self):
        title = f"Option pricing for {self.underlying}"
        author = "Option Pricing Project"
        option_simplified = self.option_df[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'impliedVolatility']]
        option_data_summary = option_simplified.describe()
        option_data_table = option_data_summary.style.to_latex()

        graph = "figure_price_strike.pgf"

        latex_code = f"""
        \\documentclass{{article}}
        \\usepackage{{pgf}}
        
        \\setlength{{\\leftmargin}}{{0cm}}
        
        \\title{{{title}}}
        \\author{{{author}}}
        
        \\begin{{document}}
        \\maketitle
        
        {option_data_table}
        
        \\begin{{figure}}
            \\centering
            \\input{{{self.directory}/figures/figure_price_strike.pgf}}
            \\caption{{Caption of the figure.}}
            \\label{{fig:label_of_figure}}
        \\end{{figure}}
        
        \\end{{document}}
        """

        return latex_code
    def generate_document(self):
        # Change to the correct directory to generate the Latex file

        latex_file = "analysis.tex"
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        # Write the Latex files
        latex_code = self.generate_latex_code()
        file_path = os.path.join(self.directory, latex_file)
        with open(file_path, "w") as f:
            f.write(latex_code)

        # compile latex file
        os.system(f"pdflatex -output-directory={self.directory} {file_path}")


