import json

f = json.load(open('/Users/alan/Desktop/youcookii_annotations_trainval.json'))['database']
train = json.load(open('/Users/alan/Desktop/test_list.txt'))
train_data = []
for i in train:
    name = i.split('/')[1]
    try:
        for j in f[name]['annotations']:
            train_data.append([i,f[name]['duration'],j['segment'],j['sentence']])
    except:
        print(name)

print(train_data)
json.dump(train_data,open('/Users/alan/Desktop/youcookii_test.json','w'))

