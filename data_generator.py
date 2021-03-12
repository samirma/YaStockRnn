import os
import json
import io
import random

class DataGenerator:
    def __init__(self, random=False, first_index=0, base_dir = "stock_data/"):
        self.base_dir = base_dir
        self.files = os.listdir(self.base_dir)
        self.files.sort()
        self.steps = len(self.files) - 1000
        self.first_index = first_index
        self.index = self.first_index
        self.is_random = random
        self.index = first_index

    def convert_to_array(self, data):
        if (isinstance(data, dict)):
            return [data]
        else:
            return data
    
    def read_json_from_file(self, file_path):
        f = io.open(file_path, mode="r", encoding="utf-8")
        raw = f.read()
        try:
            return json.loads(raw)
        except Exception as e: 
            print(len(raw))
            print("Unexpected error:", e)
    
    def get_from_index(self, index):
        file_path = self.base_dir + self.files[index]
        return self.read_json_from_file(file_path)
    
    def get_json_from_timestamp(self, timestamp):
        file_path = "{}{}.json".format(self.base_dir, timestamp)
        #print(file_path)
        if os.path.exists(file_path):
            state = self.read_json_from_file(file_path)
            return state
        else:
            print("File not found {}".format(file_path))

    def get_current_state(self):
        return self.get_from_index(self.index)

    def next(self, index = -1):
        #print("next %s %s %s" % (self.index, self.steps, self.has_next()))
        if index == -1:
            index = self.index
        state = self.get_current_state()
        self.index += 1
        return state
    
    def rewind(self):
        if self.is_random:
            self.first_index = int(random.uniform(0, 0.9)*self.steps)
        self.index = self.first_index
        #print("Initial index %s, has_next: %s" % (self.index, self.has_next()))
        
    def has_next(self):
        has_next = ((self.index) < self.steps)
        #print("has_next %s" % has_next)
        return has_next
    
    def max_steps(self):
        return self.steps