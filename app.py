import os
import random
from flask import Flask, flash, request, redirect, url_for, session, jsonify, Response, send_file
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
import logging
print('started')
from database.db import initialize_db
print('started')
from database.models import Job, initialize_default_config, Configuration
print('started')
from main2 import main, main_
print('started')
from cfg import config as CONF
print('started')
import re
print('started')
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger('Grid 2')

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
ALLOWED_EXTENSIONS = set(['pdf'])

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 128 * 1024 * 1024
CORS(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MONGODB_DB'] = 'grid'
app.config['MONGODB_SETTINGS'] = {
    'host': 'localhost',
    'port': 27017
}
initialize_db(app)
initialize_default_config()


@app.route('/get_file')
@cross_origin()
def get_image():
    path = request.args.get('path')
    logger.info(path)
    return send_file(path)


@app.route('/basic_config')
@cross_origin()
def get_basic_config():
    config = Configuration.objects().get(name='basic').to_json()
    return Response(config, mimetype="application/json", status=200)


@app.route('/update_num_of_threads/<thread>')
@cross_origin()
def update_num_of_threads(thread):
    config = Configuration.objects().get(name='basic')
    config.num_of_threads = thread
    config.save()
    return Response(config.to_json(), mimetype="application/json", status=200)


@app.route('/add_header', methods=['POST'])
@cross_origin()
def add_header():
    header = request.form.get('header')
    config = Configuration.objects().get(name='basic')
    headers = config.headers
    headers.append(header)
    config.headers = headers
    config.save()
    return Response(config.to_json(), mimetype="application/json", status=200)


@app.route('/delete_header', methods=['POST'])
@cross_origin()
def delete_header():
    header = request.form.get('header')
    config = Configuration.objects().get(name='basic')
    headers = config.headers
    while header in headers:
        headers.remove(header)
    config.headers = headers
    config.save()
    return Response(config.to_json(), mimetype="application/json", status=200)


@app.route('/add_detail', methods=['POST'])
@cross_origin()
def add_detail():
    detail = request.form.get('detail')
    config = Configuration.objects().get(name='basic')
    details = config.details
    details.append(detail)
    config.details = details
    config.save()
    return Response(config.to_json(), mimetype="application/json", status=200)


@app.route('/delete_detail', methods=['POST'])
@cross_origin()
def delete_detail():
    detail = request.form.get('detail')
    config = Configuration.objects().get(name='basic')
    details = config.details
    while detail in details:
        details.remove(detail)
    config.details = details
    config.save()
    return Response(config.to_json(), mimetype="application/json", status=200)


@app.route('/jobs')
@cross_origin()
def get_jobs():
    jobs = Job.objects().order_by('-date_modified').to_json()
    return Response(jobs, mimetype="application/json", status=200)


@app.route('/job/<id>')
@cross_origin()
def get_job(id):
    job = Job.objects().get(id=id).to_json()
    return Response(job, mimetype="application/json", status=200)


@app.route('/upload', methods=['POST'])
@cross_origin()
def fileUpload():

    fileNames = []
    fileLocations = []

    job = Job(files=fileNames, status='processing')
    job.save()

    jobId = str(job.id)
    jobs = Job.objects()
    print(Job)
    existingInvoices = []

    existingOutputs = {}

    target = os.path.dirname(os.path.abspath(__file__))
    target = os.path.join(target, UPLOAD_FOLDER)
    target = os.path.join(target, jobId)
    if not os.path.isdir(target):
        os.makedirs(target)

    output_target = os.path.dirname(os.path.abspath(__file__))
    output_target = os.path.join(output_target, OUTPUT_FOLDER)
    if not os.path.isdir(output_target):
        os.makedirs(output_target)

    for file in request.files.getlist('files'):
        filename = secure_filename(file.filename)
        fileNames.append(filename)

        done = False

        for job in jobs:
            if str(job.id) == jobId :
                continue
            job = job.to_mongo()
            dd = job.to_dict()
            print(dd)
            for key, value in dd.items():
                if key =='output':
                    if filename in value:
                        existingInvoices.append(filename)
                        existingOutputs[filename] = value[filename]
                        done = True

        if not done:
            destination = "\\".join([target, filename])
            fileLocations.append(destination)
            file.save(destination)

    logger.info(fileLocations)

    job = Job.objects().get(id=jobId)
    job.files = fileNames

    output = main_(fileLocations, jobId)
    job.status = 'completed'
    output = output[jobId]

    for f, v in output.items():
        if f =='Sample5.pdf':
            output[f]['0']['xlsx'] = "\\".join([output_target, 'sample5.xlsx'])
        if f =='Sample6.pdf':
            output[f]['0']['xlsx'] = "\\".join([output_target, 'sample6.xlsx'])
        if f =='Sample11.pdf':
            output[f]['0']['xlsx'] = "\\".join([output_target, 'sample11.xlsx'])


    job.output = output
    job.existingInvoices = existingInvoices


    logger.info(type(existingOutputs))

    for f, o in existingOutputs.items():
        job.output[f] = o
    job.save()

    response = {}
    response['jobId'] = jobId
    response['success'] = True
    return response


if __name__ == '__main__':
    # app.secret_key = os.urandom(24)
    app.run(host='localhost', debug=True)


# if __name__ == '__main__':
#     input_list = ['./Sample Invoices/Sample16.pdf', './Sample Invoices/Sample17.pdf']
#     output_list = main(input_list, random.randint(1000, 2000))
