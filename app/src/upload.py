import datetime
import firebase_admin
import json
import logging
import requests

from firebase_admin import credentials
from firebase_admin import firestore
from pathlib import Path


firebase_admin.initialize_app(
    credentials.Certificate('private-key.json')
)

DB = firestore.client()
RDB = 'https://cmput492-cstc-2023-default-rtdb.firebaseio.com'
GPS = 'https://ipinfo.io/loc'
LOCAL = Path('traffic-data.json')
LOCAL_CSV = Path('traffic-data.csv')
TOTAL_LABEL = 'cumulative total'


async def real_time(traffic_datas, class_names, class_filters, save_csv):
    json_data = {
        TOTAL_LABEL: len(traffic_datas)
    }

    for vehicle in traffic_datas.values():
        if vehicle.vehicle_type in json_data:
            json_data[vehicle.vehicle_type] += 1
        else:
            json_data[vehicle.vehicle_type] = 1

    timestamp = datetime.datetime.now().strftime("%y-%m-%d-%H:%M:%S")

    geo_response = requests.get(GPS)
    if geo_response.status_code == 200:
        geo_location = geo_response.text.strip().replace(",", ":").replace(".", ",")
        logging.info(f'GPS Locate: {geo_response.status_code}. Geo Location: {geo_location}')

        response = requests.put(f'{RDB}/{geo_location}/{timestamp}.json', json=json_data)
        
        if response.status_code < 200 or response.status_code > 299:
            logging.error(f'Upload: {response.status_code}. '
                          f'Proceed to write to local storage')
            local(json_data, timestamp)
        logging.info(f'PUT Realtime database: {response.status_code}')
    else:
        logging.error(f'GPS Locate: {geo_response.status_code}. Proceed to write local storage')
        local(json_data, timestamp)


async def static_time(traffic_datas, class_names, class_filters, save_csv):
    json_data = {
        TOTAL_LABEL: len(traffic_datas)
    }
    for vehicle in traffic_datas.values():
        if vehicle.vehicle_type in json_data:
            json_data[vehicle.vehicle_type] += 1
        else:
            json_data[vehicle.vehicle_type] = 1

    timestamp = datetime.datetime.now().strftime("%y-%m-%d-%H:%M:%S")

    geo_response = requests.get(GPS)
    if geo_response.status_code == 200:
        geo_location = geo_response.text.strip()
        logging.info(f'GPS Locate: {geo_response.status_code}. Geo Location: {geo_location}')

        device_ref = DB.collection(u'devices').document(geo_location)
        device_data_ref = device_ref.collection(u'data').document(timestamp)
        device_ref.set({u'blur': False})
        device_data_ref.set(json_data, merge=True)

        if save_csv:
            local_csv(json_data, timestamp, class_names, class_filters)
    else:
        logging.error(f'GPS Locate: {geo_response.status_code}. Proceed to write local storage')
        local(json_data, timestamp)


async def notify_blur():
    geo_response = requests.get(GPS)
    if geo_response.status_code == 200:
        geo_location = geo_response.text.strip()
        response = requests.put(f'{RDB}/device_status.json', json={
            geo_location: {
                'blur': True
            }
        })
        if response.status_code < 200 or response.status_code > 299:
            logging.error(f'Upload: {response.status_code}. '
                          f'Proceed to write to local storage')
    else:
        logging.error(f'GPS Locate: {geo_response.status_code}. Proceed to write local storage')


def local_csv(json_data, timestamp, class_names, class_filters):
    label = ['timestamp', TOTAL_LABEL]
    label += [class_names[i] for i in class_filters]
    data = [timestamp]
    data += [str(0) for i in range(len(label))]
    index_pair = {label: i for i, label in enumerate(label)}
    for key, value in json_data.items():
        data[index_pair[key]] = str(value)
    if not LOCAL_CSV.exists():
        with open('traffic-data.csv', 'w') as f:
            f.write(','.join(label) + '\n')
    with open('traffic-data.csv', 'a') as f:
        f.write(','.join(data) + '\n')


def local(json_data, timestamp):
    if LOCAL.exists():
        with open(LOCAL, 'rb') as f:
            try:
                existence_data = json.load(f)
            except ValueError:
                logging.error('Invalid json data. Proceed to override')
                existence_data = {}
    else:
        existence_data = {}

    with open(LOCAL, 'w') as f:
        existence_data[timestamp] = json_data
        json.dump(existence_data, f, indent=4)
