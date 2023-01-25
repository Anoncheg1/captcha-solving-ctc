import os
import shutil

s = '/home/u2/h4/PycharmProjects/captcha_image/09_01_23'
trus = '/home/u2/h4/PycharmProjects/captcha_image/09_01_23_alpha'

from cnncaptcha.sequence import ALPHABET

ALPHABET = ALPHABET + ['r']

ld = list(os.listdir(s))
for i in range(len(ld)):
    f = ld[i]
    spl = f.split('-')
    c = spl[0]
    if all([(x.lower() in ALPHABET) for x in c]):
        file = os.path.join(s, f)
        file2 = os.path.join(trus, f)
        shutil.move(file, file2)
        print(spl)

    # print(ld[i])
    # ldi = random.randint(0, len(ld)-1)
    # print(ldi)
    # filename = ld[i]
    # solv = filename[0:5]
    # file = os.path.join(p1, filename)
