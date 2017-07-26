import pandas as pd
import numpy as np
import csv
from matplotlib import pyplot as plt
from pandas.compat import StringIO
from collections import Counter
from scipy.misc import imread


from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=8, usetex=True)

# We'll create a QuadTree class which will recursively subdivide the
# space into quadrants
class QuadTree:
    """Simple Quad-tree class"""
    array=[]
    def addGrid(self,listName):
         print('New Grid added to the list contain number of tweets =',len(listName)+1)
         for i in range(0, len(listName)):
            self.array.append(listName[i,:])
         #print len(self.array)

         return self.array
    # class initialization function
    def __init__(self, data, mins, maxs,maxNGrid,depth):#label,
        self.data = np.asarray(data[:,1:3])
        # data should be two-dimensional
        assert self.data.shape[1] == 2
        # print self.data
        if mins is None:
            mins = data.min(0)
        if maxs is None:
            maxs = data.max(0)

        self.mins = np.asarray(mins)
        self.maxs = np.asarray(maxs)
        self.sizes = self.maxs - self.mins
        #self.label =[]
        self.children = []
#self.array=[]

        mids = 0.5 * (self.mins + self.maxs)
        xmin, ymin = self.mins
        xmax, ymax = self.maxs
        xmid, ymid = mids
        #print self.data.shape[0],self.data.shape[1]
        if self.data.shape[0]>maxNGrid:
            # split the data into four quadrants
            data_q1 = data[(data[:, 1] < mids[0])& (data[:, 2] < mids[1])]
            data_q2 = data[(data[:, 1] < mids[0])& (data[:, 2] >= mids[1])]
            data_q3 = data[(data[:, 1] >= mids[0])& (data[:, 2] < mids[1])]
            data_q4 = data[(data[:, 1] >= mids[0])& (data[:, 2] >= mids[1])]
        else:return None

            # recursively build a quad tree on each quadrant which has data
        l1=np.empty(data_q1.shape[0]); l1.fill('0')
        l2=np.empty(data_q2.shape[0]); l2.fill('1')
        l3=np.empty(data_q3.shape[0]); l3.fill('2')
        l4=np.empty(data_q4.shape[0]); l4.fill('3')
        # print l1,l2,l3,l4

        if data_q1.shape[0] >maxNGrid:
            data_q1=np.c_[data_q1,l1]
            self.children.append(QuadTree(data_q1,[xmin,ymin],[xmid, ymid],maxNGrid,depth-1))
        elif data_q1.shape[0]>=0:
            data_q1=np.c_[data_q1,l1]
            self.children.append(QuadTree(data_q1,[xmin,ymin],[xmid, ymid],maxNGrid,depth-1))
            self.addGrid(data_q1)

        if data_q2.shape[0] > maxNGrid:
            data_q2=np.c_[data_q2,l2]
            self.children.append(QuadTree(data_q2,[xmin, ymid], [xmid, ymax],maxNGrid,depth-1))
        elif data_q2.shape[0]>=0:
            data_q2=np.c_[data_q2,l2]
            self.children.append(QuadTree(data_q2,[xmin, ymid], [xmid, ymax],maxNGrid,depth-1))
            self.addGrid(data_q2)

        if data_q3.shape[0] > maxNGrid:
            data_q3=np.c_[data_q3,l3]
            self.children.append(QuadTree(data_q3,[xmid, ymin], [xmax, ymid],maxNGrid,depth-1))
        elif data_q3.shape[0]>=0:
            data_q3=np.c_[data_q3,l3]
            self.children.append(QuadTree(data_q3,[xmid, ymin], [xmax, ymid],maxNGrid,depth-1))
            self.addGrid(data_q3)

        if data_q4.shape[0] > maxNGrid:
            data_q4=np.c_[data_q4,l4]
            self.children.append(QuadTree(data_q4,[xmid, ymid], [xmax, ymax],maxNGrid,depth-1))
        elif data_q4.shape[0]>=0:
            data_q4=np.c_[data_q4,l4]
            self.children.append(QuadTree(data_q4,[xmid, ymid], [xmax, ymax],maxNGrid,depth-1))
            self.addGrid(data_q4)

    def draw_rectangle(self, ax, depth):
        """Recursively plot a visualization of the quad tree region"""
        rect = plt.Rectangle(self.mins, *self.sizes, zorder=2,
                                 ec='#000000', fc='none')
        ax.add_patch(rect)   
        if depth is None or depth == 0:
            rect = plt.Rectangle(self.mins, *self.sizes, zorder=2,
                                 ec='#000000', fc='none')
            ax.add_patch(rect)
        if depth is None or depth >= 0:
            for child in self.children:
                child.draw_rectangle(ax, depth - 1)

# def draw_grid(ax, xlim, ylim, Nx, Ny, **kwargs):
#          """ draw a background grid for the quad tree"""
#          for x in np.linspace(xlim[0], xlim[1], Nx):
#              ax.plot([x, x], ylim, **kwargs)
#          for y in np.linspace(ylim[0], ylim[1], Ny):
#              ax.plot(xlim, [y, y], **kwargs)



# ------------------------------------------------------------
# ------------------------------------------------------------
# ------------------------------------------------------------
# ------------------------------------------------------------
# ------------------------------------------------------------

print('==Loading data................................\n')

lines=list(csv.reader(open('MergedGeo32US.csv')))
print ('==checking data...............................\n')
for i in range(1,len(lines)):
    a=len(lines[i])
    if a >4:
        print('adjusting the line number=', i)
        lin=list(lines[i])
        # print 'len(lines[i])=====',len(lines[i])
        lin[0:a-3] = [' '.join(lin[0:a-3])]
        #print 'len(lin)-----',len(lin)
        lines[i]=lin
        #print 'len(lines[i]) -----',len(lines[i])
header, values = lines[0], lines[1:] 
input_data= {h:v for h,v in zip (header, zip(*values))}
df = pd.DataFrame.from_dict(input_data)

Text=df.iloc[:,3:4]
data=df.iloc[:,0:2]

np.savetxt('temp.csv',data, delimiter=';',fmt='%5s')

data= np.loadtxt('temp.csv',dtype=float,delimiter=';')#,unpack=True )
X=data[:,0:2]

print ('\n===input data intwo columns X.shape():=',X.shape)

print ('\n===add index to data materix (ID Tweet)).....\n')

id_tweet=np.zeros((len(X))).astype(int)
for i in range(0,len(X)):
    id_tweet[i]=int(i)

X=np.c_[id_tweet,X]

#------------------------------------------------------------
# Use our Quad Tree class to recursively divide the space
mins=np.min(X[:,1:3].astype(float), axis=0)
maxs=np.max(X[:,1:3].astype(float), axis=0)
print (mins)
print (maxs)
print ("*Note:\nbe aware for chosing max number in each Grid \n(if too low is possible to get 'maximum recursion depth exceeded ':)\n")
print ("Please input the max number of tweets in each Grid=:")

maxNGrid=int(input())
# maxNGrid=200
depth=4
#label=np.empty(X.shape[0]); label.fill('0')
print('\n===building Quadtree structure................\n')

QT = QuadTree(X, mins, maxs,maxNGrid,depth)

print('\n===get data label in binary format ............\n')
label=QuadTree.array

#labeled_data=np.c_[label[1:3],labeled_data]

print('\n===compute labeld Tweet from binary to integer (concatenation).........\n')

k=0
xb=np.zeros((len(label),1)).astype(int)
lb=np.zeros((len(label),3)).astype(float)
data_labled=[]
for item in label:
    lb[k,:]=item[0:3]
    xb[k]=0
    for i in range(3,len(item)):
        y=int(item[i])
        xb[k]=(xb[k] << 2) + y
    k=k+1
data_labled=np.c_[lb,xb]



data_sort_temp=data_labled[data_labled[:,3].argsort()]

print ('\n===indexing label data................\n')
counterlist=Counter(data_labled[:,3])
Grid_Number=len(counterlist)
print('\n=Grid number= ',Grid_Number)


k=np.zeros((Grid_Number,1)).astype(int)
index_lable=np.zeros((len(data),1)).astype(int)

for i in range(0,Grid_Number):
     j=i+1
     k[i]=sum(list(counterlist.values())[0:j])

for i in range(0,Grid_Number):
    ss=int(k[i-1])
    ll=int(k[i])
    if i==0:ss=0
    l=len(index_lable[ss:ll])
    index_lable[ss:ll]=i

data_sort_temp=np.c_[data_sort_temp,index_lable]
data_labled=data_sort_temp[data_sort_temp[:,0].argsort()]

#print index_lable


data_sorted=data_labled[data_labled[:,0].argsort()]
data_labled0=np.c_[Text,data_sorted[:,1:]]
data_labled=np.c_[id_tweet,data_labled0]
# data_labled=np.c_[data_labled,index]


print ('\n====save data with labeld Tweet.............................')
np.savetxt('data_labled.csv',data_labled, delimiter='\t',
    header=" index_tweet\ttext\tgeocoordinate0\tgeocoordinate1\tlabel\tindex_label",fmt='%d,%s,%3.7f,%3.7f,%d,%d')

#bb=cat(bb,bin(item(i))
#np.savetxt('sss.csv',datat, delimiter=' ',fmt='%5s')
#------------------------------------------------------------
print('=====# Plot four different levels of the quad tree====')
# print('how many samples of data you want to plot for depth= ',depth)
# z=int(input())
z=len(X)
fig = plt.figure(figsize=(8, 8))
fig.subplots_adjust(wspace=0.1, hspace=0.15,
                     left=0.1, right=0.9,
                     bottom=0.05, top=0.9)
plt.hold(True)

for level in range(1, depth+1):
    ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
    ax.scatter(X[1:z, 1], X[1:z, 2])
    QT.draw_rectangle(ax, depth=level - 1)

    # Nlines = 1 + 2 ** (level - 1)
    # draw_grid(ax, (mins[0], maxs[0]), (mins[1], maxs[1]),
    #           Nlines, Nlines, linewidth=1,
    #           color='#CCCCCC', zorder=0)

    ax.set_xlim(mins[0],maxs[0])
    ax.set_ylim(mins[1],maxs[1])
    #ax.set_xlim(-125,-71 )
    # ax.set_ylim(23,50)
    ax.set_title('level %i' % level)
img = imread("maps.jpg")

plt.imshow(img, aspect='auto', extent=(mins[0],maxs[0],mins[1],maxs[1]), 
                      alpha=0.5, origin='upper',zorder=-2)

plt.axis('off')
 # suptitle() adds a title to the entire figure
fig.suptitle('Quad-tree')
plt.show()


