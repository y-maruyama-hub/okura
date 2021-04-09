import json
import urllib.parse
import urllib.request

url = "http://localhost:5000/predict"

f = open("sample/20210314120359_1.jpg", "rb")
reqbody = f.read()
f.close()

req = urllib.request.Request(
    url,
    reqbody,
    method="POST",
    headers={"Content-Type": "application/octet-stream"},
)

#with urllib.request.urlopen(req) as res:
#    print(json.loads(res.read()))
response = urllib.request.urlopen(req)
content = response.read()
print(content)
response.close()
