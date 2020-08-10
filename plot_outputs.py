import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

# Methodology
## Distribution annotation width and height
if os.path.exists("/media/melchior/Elements/MaastrichtUniversity/BISS/MasterThesis/annotation.csv"):
    annotations = pd.read_csv("/media/melchior/Elements/MaastrichtUniversity/BISS/MasterThesis/annotation.csv",
                              header = None,
                              names = ["Label", "X", "Y", "width", "height", "Image_Name", "Xdim", "Ydim"])

plt.subplot(1,2,1)
sns.distplot(annotations["width"], hist = False)
plt.title("Distribution Of Annotated Width")
plt.axvline(annotations["width"].mean(), color = "r")
plt.legend(["Mean"])

plt.subplot(1,2,2)
sns.distplot(annotations["height"], hist = False)
plt.title("Distribution Of Annotated Height")
plt.axvline(annotations["height"].mean(), color = "r")


plt.tight_layout()
plt.show()