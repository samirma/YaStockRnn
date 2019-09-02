#!/usr/bin/env python
# coding: utf-8

# In[47]:


from __future__ import absolute_import, division, print_function, unicode_literals

import os

from data_generator import DataGenerator

from bitstamp import LiveBitstamp, Bitstamp


# In[53]:


import json
import io

class RawStateDownloader(LiveBitstamp):
    
    def __init__(self, file_path_format = "stock_data/{}.json"):
        self.trade = {}
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
        timestamp = data['timestamp']
        current = self.get_current_state(timestamp)
        current.append(data)
        self.save_file(self.get_file_path(timestamp), current)
        
    def load_file(self, filename):
        file_path = self.get_file_path(filename)
        f = io.open(file_path, mode="r", encoding="utf-8")
        raw = f.read()
        return json.loads(raw)


# In[ ]:





# In[54]:


live = RawStateDownloader()
bt = Bitstamp(live)
bt.connect()


# In[ ]:




