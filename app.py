from calc_profile import CalcProfile
from densities.differential_density import EnergyDensity, NetBaryonDensity
from eos.equation_of_state import EquationOfState
from input_output import IO
from flask import Flask, render_template, request, send_from_directory


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html', display_results = 0)


@app.route('/results', methods = ['POST'])
def results():
    cp = CalcProfile(int(request.form['atomicNum']),
                     int(request.form['massNum']),
                     float(request.form['colEn']),
                     float(request.form['formTime']),
                     request.form['whichStats'],
                     int(request.form['nTimes']))
    
    e = EnergyDensity(cp)
    nB = NetBaryonDensity(cp)
    eos = EquationOfState(cp, e, nB)
    io = IO(cp, e, nB, eos)

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
