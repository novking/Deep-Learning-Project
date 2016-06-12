import re

l = "this is cool, 123hr or 5% or 0"

obj = re.search(r'([0-9].+)', l)

for i in obj:
	print (i)