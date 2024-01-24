f=open("models.txt","r")
models=[]
for m in f.readlines():
	models.append(m[:-1])
f.close()

f=open("bias_toxic_template.conf","r")
spec=f.read()
f.close()

for m in models:
	run_spec=spec.replace("text",m)
	file_name=m.split("/")[1]+"_spec.conf"
	open(file_name,"w").write(run_spec)
	print(f"Formating run spec config {file_name} is finished")
	
