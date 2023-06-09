import subprocess
import sys
import os


class TexDocument:
    """
    Class to generate the LaTEx document containing the results of the option pricing.
    WARNING: pdflatex is necessary for the execution of this class.
    """
    def __init__(self, underlying: str, evaluation_date: str, expiration_date: str, underlying_price: float):
        """
        Args:
            underlying: underlying of the option priced
            evaluation_date: date of the pricing
            expiration_date: date at which the options expire
            underlying_price: price of the underlying at the evaluation date
        """

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
\\usepackage{{subcaption}}
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

\\begin{{figure}}
    \\centering
    \\begin{{subfigure}}{{0.45\\textwidth}}
    \\includegraphics[width=\\textwidth]{{./figures/figure_price_strike_call_BS.png}}
    \\end{{subfigure}}
    \\hfill
    \\begin{{subfigure}}{{0.45\\textwidth}}
    \\includegraphics[width=\\textwidth]{{./figures/figure_price_strike_call_CRR.png}}
    \\end{{subfigure}}
    
    \\begin{{subfigure}}{{0.45\\textwidth}}
    \\includegraphics[width=\\textwidth]{{./figures/figure_price_strike_call_Fourier.png}}
    \\end{{subfigure}}
    \\hfill
    \\begin{{subfigure}}{{0.45\\textwidth}}
    \\includegraphics[width=\\textwidth]{{./figures/figure_price_strike_call_FFT.png}}
    \\end{{subfigure}}
    
    \\begin{{subfigure}}{{0.45\\textwidth}}
    \\includegraphics[width=\\textwidth]{{./figures/figure_price_strike_call_Merton.png}}
    \\end{{subfigure}}
    \\hfill
    \\begin{{subfigure}}{{0.45\\textwidth}}
    \\includegraphics[width=\\textwidth]{{./figures/figure_price_strike_call_Heston.png}}
    \\end{{subfigure}}
    
    \\caption{{Estimated and true call prices with different pricers.}}
    \\label{{fig:call}}
\\end{{figure}}

\\begin{{figure}}
    \\centering
    
    \\begin{{subfigure}}{{0.45\\textwidth}}
    \\includegraphics[width=\\textwidth]{{./figures/figure_price_strike_put_BS.png}}
    \\end{{subfigure}}
    \\hfill
    \\begin{{subfigure}}{{0.45\\textwidth}}
    \\includegraphics[width=\\textwidth]{{./figures/figure_price_strike_put_CRR.png}}
    \\end{{subfigure}}
    
    \\begin{{subfigure}}{{0.45\\textwidth}}
    \\includegraphics[width=\\textwidth]{{./figures/figure_price_strike_put_Fourier.png}}
    \\end{{subfigure}}
    \\hfill
    \\begin{{subfigure}}{{0.45\\textwidth}}
    \\includegraphics[width=\\textwidth]{{./figures/figure_price_strike_put_FFT.png}}
    \\end{{subfigure}}
    
    \\begin{{subfigure}}{{0.45\\textwidth}}
    \\includegraphics[width=\\textwidth]{{./figures/figure_price_strike_put_Merton.png}}
    \\end{{subfigure}}
    \\hfill
    \\begin{{subfigure}}{{0.45\\textwidth}}
    \\includegraphics[width=\\textwidth]{{./figures/figure_price_strike_put_Heston.png}}
    \\end{{subfigure}}
    
    \\caption{{Estimated and true put prices with different pricers.}}
    \\label{{fig:put}}
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
        if not os.path.exists(f"{self.directory}/figures/figure_price_strike_put.png") and not os.path.exists(
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
        # An output_file is created to print the pdf latex log
        output_file = 'output.log'
        os.chdir(self.directory)
        with open(output_file, 'w') as log_file:
            subprocess.run(['pdflatex', latex_file], stdout=log_file, stderr=subprocess.STDOUT)
        #subprocess.run(['pdflatex', latex_file])
        os.chdir("../../..")
