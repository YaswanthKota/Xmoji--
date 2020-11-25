
import numpy as np
from flask import Flask, request, render_template, Response
import models as dbHandler
import mailgenerator
import re
from flask_mysqldb import MySQL
import MySQLdb.cursors
from camera import VideoCamera

app = Flask(__name__)

mysql = MySQL(app)
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')


@app.route('/userlogin')
def userloginpage():
    return render_template('userlogin.html')


@app.route('/register', methods =['GET', 'POST'])
def register():
    if request.method=='POST':
        username=request.form['username']
        password=request.form['password']
        email=request.form['email']
        phone=request.form['phone']
        valid=dbHandler.checkuser(username,password,email,phone)
        if valid==True:
            return render_template('userlogin.html',info='User Already Exists')
        else:
            x=[]
            dbHandler.insertUser(username,password,email,phone)
            x.append(email)
            mailgenerator.Sending_report(x)
            return render_template('register.html',info='Registration Successful!')
    else:
        return render_template('register.html')

@app.route('/addusers', methods =['GET', 'POST'])
def addinguser():
    if request.method=='POST':
        username=request.form['username']
        password=request.form['password']
        email=request.form['email']
        phone=request.form['phone']
        valid=dbHandler.checkuser(username,password,email,phone)
        if valid==True:
            return render_template('addusers.html',info='')
        else:
            x=[]
            dbHandler.insertUser(username,password,email,phone)
            x.append(email)
            mailgenerator.Sending_report(x)
            return render_template('addusers.html',info='Registration Successful!')
    else:
        return render_template('addusers.html')


@app.route('/deleteuser', methods =['GET','POST'])
def deleteuser():
    if request.method=='POST':
        username=request.form['username']
        email=request.form['email']
        x=dbHandler.deletinguser(username,email)
        if x==True:
            return render_template('deletinguser.html',info='Deletion Successful!')
        else:
            return render_template('deletinguser.html',info=x)

@app.route('/userlogin',methods=['POST','GET'])
def userlogin():
    if request.method=='POST':
        username=request.form['username']
        password=request.form['password']
        valid=dbHandler.check(username,password)
        if valid==True:
            return render_template('index.html',user=username)
        else:
            return render_template('userlogin.html',info=valid)
    else:
        render_template('userlogin.html')


@app.route('/adminview')
def adminview():
    x=dbHandler.retrieveUsers()
    return render_template('adminview.html',user=' ',userslist=x)

@app.route('/adminlogin')
def adminloginpage():
    return render_template('adminlogin.html')

@app.route('/deletinguser')
def deletinguser():
    return render_template('deletinguser.html')

@app.route('/adminlogin',methods=['POST','GET'])
def adminlogin():
    if request.method=='POST':
        username=request.form['username']
        password=request.form['password']
        valid=dbHandler.checkadmin(username,password)
        if valid==True:
            x=dbHandler.retrieveUsers()
            return render_template('adminview.html',user=username,userslist=x)
        else:
            return render_template('adminlogin.html',info=valid)
    else:
        render_template('adminlogin.html',info=valid)

@app.route('/addemployee')
def add():
    return render_template('addemployee.html')


def generate(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def cartoon(camera):
     while True:
        car = camera.get_cartoon()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + car + b'\r\n\r\n')


def emoji(camera):
     while True:
        emj = camera.get_emoji()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + emj + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_cartoon')
def video_cartoon():
    return Response(cartoon(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_emj')
def video_emj():
    return Response(emoji(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')




if __name__ == '__main__':
    app.run(debug=True)
