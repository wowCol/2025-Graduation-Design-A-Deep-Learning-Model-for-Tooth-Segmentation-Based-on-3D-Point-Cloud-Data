# 常用指令添加
import os
import smtplib
import datetime
from email.mime.text import MIMEText


def alert_sd(begin, data_list):
    # 运行完关机
    print(f"Train success, shutdown computer.")
    # 使用QQ邮箱提醒完成
    sender = user = 'wqy2693699654@qq.com'
    passwd = 'rxegdqksnrlydefe'

    receiver = 'wqy2693699654@qq.com'

    msg = MIMEText(f'服务器上运行的训练程序已经完成!\n'
                   f'开启时间为: {begin}\n'
                   f'结束时间为: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n'
                   f'总计训练次数为: {data_list[0]}\n'
                   f'是否触发早停: {data_list[1]}\n'
                   f'保存的最好模型的序号是: {data_list[2]}\n'
                   f'服务器即将关机。~o( =∩ω∩= )m', 'plain', 'utf-8')

    msg['From'] = f'MyServer<wqy2693699654@qq.com>'
    msg['To'] = receiver
    msg['Subject'] = '服务器完成训练邮件通知'

    try:
        smtp = smtplib.SMTP_SSL('smtp.qq.com', 465)

        smtp.login(user, passwd)

        smtp.sendmail(sender, receiver, msg.as_string())
        print("完成邮件发送")
        smtp.quit()
    except Exception as e:
        print(e)
        print("发送邮件失败")
    # 关机
    os.system('shutdown -h now')
