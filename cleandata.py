import shutil
import csv

with open('./NEW/Run2/driving_log.csv') as csvfile:
  logreader = csv.reader(csvfile, delimiter=',')
  count = 0
  for row in logreader:
    count += 1
    if count % 100 == 0:
      print(".",end="")
    if count % 1000 == 0:
      print(count)
    if count == 1:
      continue
    for i in range(1):
      src = './NEW/Run2/'+row[i].strip()
      dest = './data/'+row[i].strip()
      #print(src,dest)
      shutil.copy2(src, dest)
    #break
print("Moved {0} files".format(count))