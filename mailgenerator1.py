import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email import encoders
from datetime import datetime

def Sending_report(mails):
    print('mail is sending.....wait.')
    now=datetime.now()
    t=now.strftime("%d/%m/%Y, %H:%M:%S")
    time=str(t)
    fromaddr = "sarathmajji999@gmail.com"
    msg = MIMEMultipart()
    msg['From'] = fromaddr
    for i in mails:
        msg['To'] = i
    msg['Subject'] = "Get Your Own Emoji through Xmoji!"
    #x="C:/yaswanthfiles/mini project/homeimages/display0.jpg"
    #y=open("C:/yaswanthfiles/mini project/homeimages/display0.jpg",'rb')
    #msg.attach(MIMEImage(y.read()))
    body = "Dear,Welcome to Xmoji! We're are glad you're here.Start using Xmoji today."+'To know how to use click on this https://yaswanthkota.github.io/Xmoji-Home/'
    msg.attach(MIMEText(body, 'plain'))
    filename = "C:/yaswanthfiles/mini project/homeimages/display0.jpg"
    attachment = open("C:/yaswanthfiles/mini project/homeimages/display0.jpg", "rb")
    p = MIMEBase('application', 'octet-stream')
    p.set_payload((attachment).read())
    encoders.encode_base64(p)
    p.add_header('Content-Disposition', "attachment; filename= %s" % filename)
    msg.attach(p) #to attach a pdf
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login(fromaddr,"msk271699")
    text = msg.as_string()
    cnt=0
    for i in mails:
        s.sendmail(fromaddr,i,text)
        print('welcome mail sent successfully to '+i)
        cnt+=1
    print('Total Number of mails sent',cnt)
    s.quit()
def Sending_otp(mails,otp):
    print('mail is sending.....wait.')
    now=datetime.now()
    t=now.strftime("%d/%m/%Y, %H:%M:%S")
    time=str(t)
    fromaddr = "sarathmajji999@gmail.com"
    msg = MIMEMultipart()
    msg['From'] = fromaddr
    for i in mails:
        msg['To'] = i
    msg['Subject'] = "OTP to Login"
    x="C:/yaswanthfiles/mini project/homeimages/display0.jpg"
    y=open("C:/yaswanthfiles/mini project/homeimages/display0.jpg",'rb')
    #msg.attach(MIMEImage(y.read()))
    body = "Dear,Welcome to Emojify!.Your OTP is "+str(otp)
    msg.attach(MIMEText(body, 'plain'))
    filename = "C:/yaswanthfiles/mini project/homeimages/display0.jpg"
    attachment = open("C:/yaswanthfiles/mini project/homeimages/display0.jpg", "rb")
    p = MIMEBase('application', 'octet-stream')
    p.set_payload((attachment).read())
    encoders.encode_base64(p)
    p.add_header('Content-Disposition', "attachment; filename= %s" % filename)
    msg.attach(p) #to attach a pdf
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login(fromaddr,"msk271699")
    text = msg.as_string()
    cnt=0
    for i in mails:
        s.sendmail(fromaddr,i,text)
        print('welcome mail sent successfully to '+i)
        cnt+=1
    print('Total Number of mails sent',cnt)
    s.quit()