from flask import Flask, render_template, request, send_from_directory
from density_calc import efullz, nbfullz
from thermo_calc import quant_full_soln, boltz_full_soln
from utilities import CalculationContext
from piecewise import PiecewiseSolution
from differential_density import EnergyDensity, NetBaryonDensity

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', display_results = 0)

@app.route('/results', methods = ['POST'])
def results():
    cc = CalculationContext(int(request.form['atomicNum']),
                            int(request.form['massNum']),
                            float(request.form['colEn']),
                            float(request.form['formTime']),
                            request.form['whichStats'],
                            int(request.form['nTimes']))
    
    # ed = EnergyDensity(cc)
    # nd = NetBaryonDensity(cc)

    # ps = PiecewiseSolution(cc, ed)
    # ps.calculate(cc)

    # ps = PiecewiseSolution(cc, nd)
    # ps.calculate(cc)

    efullz(*cc.get_data_for_density_calc())

    nbfullz(*cc.get_data_for_density_calc())

    if (cc.equationOfState == "quantum"):
        quant_full_soln(*cc.get_data_for_thermo_calc())

    if (cc.equationOfState == "boltzmann"):
        boltz_full_soln(*cc.get_data_for_thermo_calc())

    return render_template('index.html', display_results = 1)

@app.route('/results/view')
def view():
    return send_from_directory('results', 'results.pdf')

@app.route('/results/view2')
def view2():
    return send_from_directory('results', 'e-vs-t.pdf')

@app.route('/results/download')
def download():
    return send_from_directory('results', 'T-muB-muS-muQ-vs-t.dat', as_attachment = True)
