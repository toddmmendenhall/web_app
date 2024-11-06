from calc_profile import CalcProfile
from densities.differential_density import EnergyDensity, NetBaryonDensity
from eos.equation_of_state import EquationOfState
from input_output import IO
from flask import Flask, render_template, request, send_from_directory
from socket import gethostname


class App(Flask):
    def __init__(self):
        super().__init__(__name__)

        # Find out if running online on pythonanywhere.com
        self.isOffline = True if 'liveconsole' not in gethostname() else False

app = App()

@app.route('/')
def home():
    return render_template('home.html')

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
    io = IO(cp, e, nB, eos, app.isOffline)
    return render_template('results.html')

@app.route('/results/view_energy_density')
def view_energy_density():
    return send_from_directory('results', 'e_vs_t.png')


@app.route('/results/view_trajectory')
def view_trajectory():
    return send_from_directory('results', 'phase_diagram_trajectory.png')

@app.route('/results/download')
def download():
    return send_from_directory('results', 'time_evolution.csv', as_attachment = True)


if __name__ == '__main__':
    if 'liveconsole' not in gethostname():
        app.run()