from flask import Flask, render_template, request, Response, redirect, url_for, jsonify
import subprocess
import re
import os

process = None

from pyngrok import ngrok
app = Flask(__name__)

ngrok.set_auth_token("3Ah3ScDXd7qw9TuTy74YKwddTJd_ttmDnP8PY6KFVh6ebpqY")
public_url=ngrok.connect(5000).public_url
print(public_url)

fileName = "File"

@app.route("/videoProgress" )
def videoProgress():
    return render_template('loadingPage.html' , videoUrl = fileName)

@app.route("/")
def home_page():
    return render_template('fightDetection.html')


@app.route("/getProccessStatus")
def getProcessStatus():
    global process
    if(process is None):
        return jsonify({"status" : "none"})
    elif(process.poll() is None):
        return jsonify({"status" : "running"})
    return jsonify({"status" : "complete"})

@app.route("/output")
def output():
    return render_template('output.html' , fileName = fileName)

@app.route("/videoProcess", methods=['POST'])
def videoProcess():
    global process
    global fileName
    fileName = request.form['fileName']
    runMode =  request.form['runtype']
   
            
            # Start the subprocess to run the command
            
    if(runMode == "CPU"):
         command = ['python', 'main.py', '--input','static/resources/videos/Test_Videos/'+ fileName, '--output', 'static/resources/videos/Output_Videos/fightdetect.mp4', '--device', 'cpu']
    else:
        command = ['python', 'main.py', '--input','static/resources/videos/Test_Videos/'+ fileName, '--output', 'static/resources/videos/Output_Videos/fightdetect.mp4']
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    return render_template('loadingPage.html')

if __name__ == "__main__":
    app.run(port = 5000)

