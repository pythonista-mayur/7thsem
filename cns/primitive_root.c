number = int(input("enter the number is:"))
def find_primitive(number):
	for i in range(number):
		nut = list()
		for j in range(number-1):
			nut.append(i**j%number)
		if len(nut) == len(set(nut)):
			return i 


