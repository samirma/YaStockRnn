

class DataDetail():
    
    def __init__(self,
                windows,
                minutes,
                steps_ahead
                 ):
        self.windows = windows
        self.minutes = minutes
        self.steps_ahead = steps_ahead

    def get_seconds(self):
        return self.minutes * 60

    def __str__(self):
        return f"DataDetail (windows = {self.windows}, minutes = {self.minutes}, steps_ahead = {self.steps_ahead})"

class ModelDetail():
    
    def __init__(self,
                 data_detail: DataDetail,
                 model
                 ):
        self.data_detail = data_detail
        self.model = model

    def __str__(self):
        return f"ModelDetail (model = {self.model}, data_detail = {self.data_detail})"


def print_data_detail(data_detail: DataDetail):
    print(f"windows = {data_detail.windows} minutes = {data_detail.minutes} steps_ahead = {data_detail.steps_ahead}")


def print_model_detail(model_detail: ModelDetail):
    print_data_detail(model_detail.data_detail)
    print(f"{model_detail.model}")
