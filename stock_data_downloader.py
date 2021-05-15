#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals

import os

from data_generator import DataGenerator

from bitstamp import LiveBitstamp, Bitstamp

import argparse


import json
import io

class RawStateDownloader(LiveBitstamp):
    
    def __init__(self, output_dir = "stock_data"):
        self.trade = {}
        file_path_format = output_dir + "/{}.json"
        print(file_path_format)
        self.file_path_format = file_path_format
    
    def get_file_path(self, timestamp):
        return self.file_path_format.format(timestamp)
    
    def save_file(self, filename, data):
        with open(filename, 'w') as json_file:
            json.dump(data, json_file)
            json_file.close()
            
    def get_current_state(self, filename):
        file_path = self.get_file_path(filename)
        #print(file_path)
        if os.path.exists(file_path):
            return self.load_file(filename)
        else:
            return []
            
    def process(self, data):
        #print(data)
        timestamp = data['microtimestamp']
        #current = self.get_current_state(timestamp)
        #current.append(data)
        print(timestamp)
        self.save_file(self.get_file_path(timestamp), data)
        
    def load_file(self, filename):
        file_path = self.get_file_path(filename)
        f = io.open(file_path, mode="r", encoding="utf-8")
        raw = f.read()
        return json.loads(raw)


import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--o', dest="dir", action="store")
parser.add_argument('--c', dest="currency", action="store", default="btcusd")

args = parser.parse_args()

live = RawStateDownloader(args.dir)
bt = Bitstamp(live, currency = args.currency)
bt.connect()




