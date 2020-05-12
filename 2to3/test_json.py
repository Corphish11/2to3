import json
import math
vx = 1
vy = 2
test_dict = {'bigberg': [7600, {1: [['iPhone', [vx, vy]], ['Bike', 800], ['shirt', 300]]}]}
print(test_dict)
print(type(test_dict))
#dumps 将数据转换成字符串
json_str = json.dumps(test_dict)
# print(json_str)
# print(type(json_str))
new_dict = json.loads(json_str)
print(new_dict)
print(type(new_dict))
with open("parabola_info.json",'r') as load_f:
    load_dict = json.load(load_f)
    print(len(load_dict))

if (3 > 2) & (4 > 3) & (5 < 2):
    print("yes")
print(math.sqrt(4))