from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = [[0, 0, 0, 0, 4, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 6, 2, 0], [4, 1, 1, 0, 16, 1], [6, 1, 2, 33, 0, 10], [0, 0, 0, 0, 19, 0]]

# plt.matshow(cm)
# plt.title('Logistical Regression')
# plt.colorbar()
# plt.ylabel('BI-RAD')
# plt.xlabel('Predicted BI-RAD')
# plt.show()

df = {'BI-RADS' : [0, 1, 2, 3, 4, 5],
 'counts' :[ 0, 0, 0, 0, 4, 0,
   0, 0, 0, 0, 0, 0,
   0, 0, 0, 6, 2, 0,
   4, 1, 1, 0, 16, 1,
   6, 1, 2, 33, 0, 10,
   0, 0, 0, 0, 19, 0
   ]}

sns.set(font_scale=1.5)
sns.set_style(style='white')
# ax = sns.barplot("BI-RADS", "counts", data=df,
#                palette=sns.cubehelix_palette(1))


cmap = sns.cubehelix_palette(as_cmap=True)
ax = sns.heatmap(cm, cmap=cmap)
ax.set(xlabel='Predicted BI-RAD', ylabel='BI-RAD')
plt.xticks(rotation=25)
plt.tight_layout()

plt.show()
extract()