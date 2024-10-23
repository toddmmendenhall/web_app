from flask import Flask, render_template, request, send_from_directory
from utilities import CalculationContext
from piecewise import PiecewiseSolution
from differential_density import EnergyDensity, NetBaryonDensity
from thermo_calc import NonInteractingMasslessQuantumEOSFullSolution, NonInteractingMasslessBoltzmannEOSFullSolution
from input_output import IO


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
    
    ed = EnergyDensity(cc)
    nd = NetBaryonDensity(cc)

    pse = PiecewiseSolution(cc, ed)
    pse.calculate(cc)
    psn = PiecewiseSolution(cc, nd)
    psn.calculate(cc)

    if (cc.equationOfState == "quantum"):
        eos = NonInteractingMasslessQuantumEOSFullSolution(cc, pse, psn)

    if (cc.equationOfState == "boltzmann"):
        eos = NonInteractingMasslessBoltzmannEOSFullSolution(cc, pse, psn)

    eos.calculate()

    io = IO(eos)
    io.write_output()
    io.make_plots(cc)

    return render_template('index.html', display_results = 1)


@app.route('/results/view')
def view():
    return send_from_directory('results', 'phase_diagram_trajectory.pdf')


@app.route('/results/view2')
def view2():
    return send_from_directory('results', 'e-vs-t.pdf')


@app.route('/results/download')
def download():
    return send_from_directory('results', 'time_evolution.csv', as_attachment = True)
