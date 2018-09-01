from cloudant import Cloudant
from flask import Flask, render_template, request, jsonify
from watson_developer_cloud import VisualRecognitionV3
from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud import LanguageTranslatorV3 as LanguageTranslator
from watson_developer_cloud.natural_language_understanding_v1 \
import Features, EntitiesOptions, KeywordsOptions

from variables import *

import numpy as np
import pandas as pd
import io
import urllib3, requests
import atexit
import os
import json
import base64

app = Flask(__name__, static_url_path='')

db_name = 'mydb'
client = None
db = None

if 'VCAP_SERVICES' in os.environ:
    vcap = json.loads(os.getenv('VCAP_SERVICES'))
    print('Found VCAP_SERVICES')
    if 'cloudantNoSQLDB' in vcap:
        creds = vcap['cloudantNoSQLDB'][0]['credentials']
        user = creds['username']
        password = creds['password']
        url = 'https://' + creds['host']
        client = Cloudant(user, password, url=url, connect=True)
        db = client.create_database(db_name, throw_on_exists=False)
elif "CLOUDANT_URL" in os.environ:
    client = Cloudant(os.environ['CLOUDANT_USERNAME'], os.environ['CLOUDANT_PASSWORD'], url=os.environ['CLOUDANT_URL'], connect=True)
    db = client.create_database(db_name, throw_on_exists=False)
elif os.path.isfile('vcap-local.json'):
    with open('vcap-local.json') as f:
        vcap = json.load(f)
        print('Found local VCAP_SERVICES')
        creds = vcap['services']['cloudantNoSQLDB'][0]['credentials']
        user = creds['username']
        password = creds['password']
        url = 'https://' + creds['host']
        client = Cloudant(user, password, url=url, connect=True)
        db = client.create_database(db_name, throw_on_exists=False)

# On IBM Cloud Cloud Foundry, get the port number from the environment variable PORT
# When running this app on the local machine, default the port to 8000
port = int(os.getenv('PORT', 8000))

@app.route('/')
def root():
    return app.send_static_file('index.html')

@app.route('/api/visitors', methods=['GET'])
def get_visitor():
    if client:
        return jsonify(list(map(lambda doc: doc['name'], db)))
    else:
        print('No database')
        return jsonify([])

@app.route('/api/identificacion', methods=['POST'])
def search_identification():
    numero = str(request.json['identificacion'])
    item = next((item for item in datos if item["cedula"] == numero), False)
    if item:
        wml_credentials={
            "url": "https://us-south.ml.cloud.ibm.com",
            "username": "48cd5fe0-6197-4814-99bf-f6d2f7a45907",
            "password": "db02e1d3-8616-4254-afc2-550c001a8bbb"
        }

        headers = urllib3.util.make_headers(basic_auth='{username}:{password}'.format(username=wml_credentials['username'], password=wml_credentials['password']))
        url = '{}/v3/identity/token'.format(wml_credentials['url'])
        response = requests.get(url, headers=headers)
        mltoken = json.loads(response.text).get('token')
        header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

        new_observation=np.array([[item['genero'][0], "1" if item['pension'] == "Activo" else "0","1" if item['regimen'] == "Subsidiado" else "0",item['estrato'],int(item['edad']),"1" if item['regimen'] == "Subsidiado" else "0", "China",item['residencia'],"Primera vez"]],dtype=object)
        new_observation=pd.DataFrame(new_observation)

        payload_scoring={'fields':["Genero", "Pension", "Regimen ", "Estrato", "Edad", "Tipo_ de_ titular", "Destino", "Residencia", "Viaje al exterior "], 'values':[list(new_observation.values[0])]}
        print(payload_scoring)

        # payload_scoring = {'fields': ["Genero", "Pension", "Regimen ", "Estrato", "Edad", "Tipo_ de_ titular", "Destino", "Residencia", "Viaje al exterior "], 'values': [item['genero'][0], "1" if item['pension'] == "Activo" else "0","1" if item['regimen'] == "Subsidiado" else "0",item['estrato'],int(item['edad']),"1" if item['regimen'] == "Subsidiado" else "0", "China",item['residencia'],"Primera vez"]}

        response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/v3/wml_instances/ae8bf05f-cda0-46d5-b5c3-8276514f526d/deployments/2b40e96b-e010-492e-b3ef-2300471f9089/online', json=payload_scoring, headers=header)

        return jsonify([item,json.loads(response_scoring.text)])
    else:
        return jsonify([])

@app.route('/api/audio', methods=['POST'])
def analyze_audio():
    print(request.json['texto'])

    ''' Parte para programar y traducir al inglés '''
    language_translator = LanguageTranslator(
        version='2018-03-16',
        iam_api_key='XmyHrVcLnTgWC3Ou33zGB989tcrOxocykZeZDUJxdlP6',
        url='https://gateway.watsonplatform.net/language-translator/api')

    translation = language_translator.translate(
        text=request.json['texto'],
        model_id='es-en')

    ''' Parte para sacar insights del texto '''
    natural_language_understanding = NaturalLanguageUnderstandingV1(
      username='50c40d6c-6a36-462a-a0da-9264052eb9f1',
      password='OiLpaGcDYeNb',
      version='2018-03-16')

    response = natural_language_understanding.analyze(
      text=json.loads(json.dumps(translation, indent=2, ensure_ascii=False))["translations"][0]["translation"],
      features=Features(
        entities=EntitiesOptions(
          emotion=True,
          sentiment=True,
          limit=2),
        keywords=KeywordsOptions(
          emotion=True,
          sentiment=True,
          limit=2)))

    return jsonify(json.dumps(response, indent=2))

@app.route('/api/video', methods=['POST'])
def analyze_foto():
    print(request.json['buffer'])

    buffer_photo = base64.b64decode(request.json['buffer'])

    visual_recognition = VisualRecognitionV3(
        '2018-03-19',
        iam_api_key='6MY-fuH8EIgCemhr-kFZJJ767Ns404dMtCwWDGK4TQ5m')

    filename = './ejemplo.jpg'
    with open(filename, 'wb') as f:
        f.write(buffer_photo)

    with open('./ejemplo.jpg', 'rb') as images_file:
        classes = visual_recognition.classify(
        buffer_photo,
        threshold='0.6',
        classifier_ids='DefaultCustomModel_148873306')

    print(json.dumps(classes, indent=2))
    return jsonify(data)

@atexit.register
def shutdown():
    if client:
        client.disconnect()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)
