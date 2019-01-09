import os
import glob
import pickle
#d = []
dic = pickle.load(open( "/Users/konstantinfomin/Documents/kyrsovaja/dump_of_emotion.p", "rb" ))
for fe in os.listdir(os.getcwd()):
    for filename in glob.glob(str(fe+'/')+'*.lm2'):
        f = open(filename,'r')
        lines = f.readlines()
        a =  int(lines[2][:2])+2
        b = int(lines[2][:2])
        MS = str(filename)+"\n"
        MS+= dic.get(filename[12:-6])+"\n"
        for i in range(5,b+5):
            s1 = lines[i].rstrip()
            s2 = lines[i+a].rstrip()
            s  =  s1 +" "+ s2+"\n"
            MS +=s
            #if s1 not in d:
               # d.append(s1)
        fn =  filename[:-4]+".txt"
        file = open (fn,'w')
        file.write(MS)
    #break
#print(d)
#print(len(d))
print("Done")
