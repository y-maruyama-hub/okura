import json
import urllib.parse
import urllib.request

#url = "http://localhost:5000/predict"
url = "http://localhost:5000/bgrenew"

f = open("sample/20210314120359_1.jpg", "rb")
reqbody = f.read()
f.close()

print(type(reqbody))

req = urllib.request.Request(
    url,
    reqbody,
    method="POST",
    headers={"Content-Type": "application/octet-stream"},
)

#with urllib.request.urlopen(req) as res:
#    print(json.loads(res.read()))
response = urllib.request.urlopen(req)
json_str = response.read()
response.close()

print(json_str)

j=json.loads(json_str)

print(j["prob"])
