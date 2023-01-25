import os

# from tensorflow import keras
# keras.backend.ctc_batch_cost

p1 = '09_01_23_alpha/'
# p1 = 'train/'

ld = list(os.listdir(p1))
max = 1
min = 9
for i in range(len(ld)):
    # f = ld[i]
    f = ld[i].split('-')[0] + '.jpg'
    l = len(f[:-4])
    if l == 3:
        print(f, f[:-4])

    if l > max:
        max = l
    if l < min:
        min = l

print("max", max)
print("min", min)
