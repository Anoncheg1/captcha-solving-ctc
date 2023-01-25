import subprocess
# -- unset PYTHONPATH for Tensorflow - it uses own numpy version
import subprocess, os
my_env = os.environ.copy()
my_env["PYTHONPATH"] = ""

pipe = subprocess.Popen(["python3", "./service_process.py"], text=True, universal_newlines=True,
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=my_env)
print('asf')
# pipe.wait(2)
# print(pipe.stdout.readline())
# print('0', pipe.stdout.readline()+'9')
# print('0', pipe.stdout.readline()+'9')
while pipe.stdout.readline() != 'ready\n':
    pass
print('awtf')
pipe.stdin.write("/home/u2/h4/PycharmProjects/captcha_image/\n")
pipe.stdin.flush()
print(pipe.poll())
# print(pipe.communicate("/home/u2/h4/PycharmProjects/captcha_image/\n"))
print('awtf')
# pipe.stdout = subprocess.PIPE
# pipe.stdout.
print(pipe.stdout.readline())
print('awtssf')
pipe.stdin.writelines(["/home/u2/h4/PycharmProjects/captcha_image/test/2д6мм.jpg\n"])
pipe.stdin.flush()
print('awtf')
# v = pipe.communicate('asd\n')
# print(v)
# print('as', pipe.stdout.read())
r = pipe.stdout.readline()
print(r, 's'+r[:-1][7:]+'s')
print()
print('awtf2')
print(pipe.stdout.readline())

