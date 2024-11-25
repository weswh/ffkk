import paramiko
import time
import os
import cv2
from time import sleep
from multiprocessing import Process

images_path = "/home/chuck/Feet3D/gaussian_surfels/TakePhotos"

def get_image(id,a):
    # 创建SSH客户端
    num = 3*id+1
    # resolution = (640,480)
    # skip_frames = 5
    ssh_client = paramiko.SSHClient()

    # 添加主机到信任列表
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # 连接远程主机
    ssh_client.connect(hostname=f'192.168.1.1{id+1}', username='orangepi', password='orangepi')

    print(f'Hub {id+1} successfully conected.')
    # print(time.time()-t)
    # 执行远程命令
    # ssh_client.exec_command('./img.sh')
    if a==0:
        print(f'Taking pictures in orangepi {id+1}...')
        stdin, stdout, stderr = ssh_client.exec_command('sh img.sh')
        # time.sleep(5)
        # 输出命令执行结果
        print(stderr.read().decode())
    # print(time.time()-t)
    # time.sleep(2)
    # # 上传文件到远程服务器
    if a==1:
        sftp = ssh_client.open_sftp()

        for i in range(3):
            
            name1 = i+1
            name2 = str(num+i).zfill(3)
            # name2 = str(num+i+7).zfill(4)
            # sftp.get(name1+'.jpg', './images/'+name2+'.jpg')
            sftp.get(f'{name1}.jpg', f'{images_path}/images/'+name2+'.jpg')
    # sftp.get('002.jpg', './002.jpg')
    # 关闭SSH连接
    if a==2:
        ssh_client.exec_command(f'rm *.jpg')
        time.sleep(0.01)
    ssh_client.close()

def workflow(a):
    t1 = time.time()
    os.makedirs(f'{images_path}/images',exist_ok=True)
    ps = []
    ps2 = []
    for i in range(8):
        p1 = Process(target=get_image,args=(i+1,a))
        ps.append(p1)
        p1.start()    

    for p1 in ps:
        p1.join()
        # p2.join()
    
    print(time.time()-t1)
    
if __name__ == '__main__':
    workflow(0)
    # USB_capture()
    print("\nFinish taking photos.")
    print("Finish taking photos.")
    print("Finish taking photos.")
    print("Finish taking photos.")
    print("Finish taking photos.")
    print("Finish taking photos.\n")
    workflow(1)
    print("\nFinish transmission")
    print("Finish transmission")
    print("Finish transmission")
    print("Finish transmission")
    print("Finish transmission")
    workflow(2)

    # get_image(8)
    print('\nDone.\nDone.\nDone.')

