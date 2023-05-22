import subprocess
import sys
import os


class TexDocument:

    def __init__(self, underlying: str, evaluation_date: str, expiration_date: str, underlying_price: float):

        self.underlying = underlying
        self.evaluation_date = evaluation_date
        self.expiration_date = expiration_date
        self.underlying_price = underlying_price
        self.directory = f"./results/{self.underlying}/analysis"

    def generate_latex_code(self):
        """ Create string containing the latex code"""
        title = f"Option pricing for {self.underlying}"
        author = "Option Pricing Project"

        latex_code = f"""
\\documentclass{{article}}
\\usepackage{{pgf}}
\\usepackage{{graphicx}}

\\setlength{{\\leftmargin}}{{0cm}}

\\title{{{title}}}
\\author{{{author}}}

\\begin{{document}}
\\maketitle

\\begin{{enumerate}}
\\item Evaluation date: {self.evaluation_date}
\\item Redemption date: {self.expiration_date}
\\item Underlying price: {self.underlying_price:.2f}
\\end{{enumerate}}
        
\\begin{{figure}}
    \\centering
    \\begin{{minipage}}[b]{{\\textwidth}}
    \\includegraphics[width=\\textwidth]{{./figures/figure_price_strike_call.png}}
    \\end{{minipage}}
    \\begin{{minipage}}[b]{{\\textwidth}}
    \\includegraphics[width=\\textwidth]{{./figures/figure_price_strike_put.png}}
    \\end{{minipage}}
    \\caption{{Estimated and true option prices.}}
    \\label{{fig:call_put}}
\\end{{figure}}

\\end{{document}}"""

        return latex_code

    def generate_document(self):
        """ Generate a latex document with the figures containing the true option price and the estimated
        option price. """

        # Change to the correct directory to generate the Latex file
        latex_file = "analysis.tex"
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        # Check if the figures were correctly generated
        if not os.path.exists(f"{self.directory}/figures/figure_price_strike_put.pgf") and not os.path.exists(
                f"{self.directory}/figures/figure_price_strike_put.pgf"):
            print("Error: the figures were not generated for the Latex file.")
            sys.exit(1)

        # Write the Latex files
        latex_code = self.generate_latex_code()
        file_path = os.path.join(self.directory, latex_file)
        with open(file_path, "w") as f:
            f.write(latex_code)

        # compile latex file
        # To create a valid latex code, compilable from any other IDE, the directory must be changed so that the figure
        # path is valid.
        os.chdir(self.directory)
        subprocess.run(['pdflatex', latex_file])
        os.chdir("../../..")
