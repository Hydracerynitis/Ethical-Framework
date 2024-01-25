'''
This python file is to read models in model.txt and create run_specs.conf for each model by replacing "text" in bias_toxic_template.conf
with their names. So that I could run each individual experiment with helm-run command

Since I am running with Huggingfaces models, I can not use helm-run command with --models-to-run flag to directly run the template with 
selected models.
'''

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
	
