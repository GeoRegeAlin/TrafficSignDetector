import matplotlib.pyplot as plt
import os

left = [1,2,3,4,5]
height=[]
for filename in os.listdir('GTSRB/Training/images/'):
    dirList=os.listdir('GTSRB/Training/images/'+filename)
    height.append(len(dirList))


# labels for bars
tick_label = [i for i in range(0,43)]
print(type(height))
print(type(tick_label))
# plotting a bar chart
plt.bar(left, height, tick_label=tick_label,
        width=0.8, color=['red', 'green'])

# naming the x-axis
plt.xlabel('x - axis')
# naming the y-axis
plt.ylabel('y - axis')
# plot title
plt.title('My bar chart!')


plt.show()