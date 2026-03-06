import urllib.request
import json
import numpy as np
def main ():
    
    url = "http://10.208.23.248:3000/dataset?student_id=cs5210602&dataset_num=1"
    # url = "http://10.208.23.248:3000/dataset?student_id=cs5210602&dataset_num=2"
    # dataset_num can take values either 1 or 2
    with urllib.request.urlopen(url) as response:
        # Read the response and decode the bytes to a string
        raw_data = response.read().decode('utf-8')
        # Parse the JSON string into a Python object
        data = json.loads(raw_data)

    data = np.array(data["X"])
    print(data)

if __name__ =="__main__":
    main ()