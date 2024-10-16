from flask import Flask, render_template, request, send_from_directory
from get_results2 import efullz, nbfullz, quant_full_soln, boltz_full_soln

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', display_results = 0)

@app.route('/results', methods = ['POST'])
def results():
    z = float(request.form['atomicNum'])
    a = float(request.form['massNum'])
    sqrtsnn = float(request.form['colEn'])
    tauf = float(request.form['formTime'])
    stats = float(request.form['whichStats'])
    ntimes = int(request.form['nTimes'])

    if (ntimes > 100):
        ntimes = 100

    efullz(a, sqrtsnn, tauf, ntimes)

    nbfullz(a, sqrtsnn, tauf, ntimes)

    if (stats == 0):
        quant_full_soln(z, a, sqrtsnn, tauf, ntimes)
    elif (stats == 1):
        boltz_full_soln(z, a, sqrtsnn, tauf, ntimes)

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
