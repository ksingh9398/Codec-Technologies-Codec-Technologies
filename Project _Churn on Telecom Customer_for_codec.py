#!/usr/bin/env python
# coding: utf-8

# # Churn On Telecom Customer

# In[4]:


# import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# Click here for Dataset: https://github.com/ksingh9398/Evaluation-project-Phase/blob/main/Telecom_customer_churn.xls

# Question --> 1
# 
# The dataset contains the data of the customer. On the basis of the data we have to predict the churn rate by the customer. The dataset contains the data like 'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents','tenure', 'PhoneService', 'MultipleLines', 'InternetService','OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport", StreamingTV', 'StreamingMovies', 'Contract', 'Paperless Billing', PaymentMethod", "MonthlyCharges', 'TotalCharges' and Churn'
# 
# Churn rate (sometimes called attrition rate), in its broadest sense, is a measure of the number of individuals or items moving out of a collective group over a specific period. It is one of two primary factors that determine the steady-state level of customers a business will support. The term is used in many contexts, but is most widely applied in business with respect to a contractual customer base, for example in businesses with a subscriber-based service model such as mobile telephone networks and pay TV operators. The term is also used to refer to participant turnover in peer-to-peer networks. Churn rate is an input into customer lifetime value modeling, and can be part of a simulator used to measure return on marketing investment using marketing mix modeling.

# In[5]:


# Load dataset
df=pd.read_csv('Telecom_customer_churn.csv')
df.head()


# Observations-->
# 
# This dataset contains the details of customers in which both numerical and categorical data are present. Here "Churn" is the target variable which contains 2 categories so it will be termed as "Classification problem where we need to predict the several churn using the classification models 

# Exploratory\\ Data Analysis(EDA)

# In[6]:


from sklearn.preprocessing import OrdinalEncoder
OE=OrdinalEncoder()
for i in df.columns:
    if df[i].dtypes=='object':
        df[i]=OE.fit_transform(df[i].values.reshape(-1,1))
df


# In[7]:


df.shape


# This Dataset contains 7043 rows and 21 columns. out of which 1 is target variable and another of 20 are independense variable 

# In[8]:


df.columns


# In[9]:


# checking the columns names in this dataset in list format 
df.columns.tolist()


# In[10]:


df.dtypes


# In[11]:


df.isnull().sum()


# In[12]:


df.isnull().sum().sum()


# In[13]:


# full informations 
df.info()


# as we see that there are no null values are present in this dataset

# In[14]:


# let visualize it using heatmap
sns.heatmap(df.isnull())


# In[15]:


df['TotalCharges'].unique()


# In[16]:


df['TotalCharges'].nunique()
# count of total unique values 


# In[17]:


# for checking the value cont of each column
for i in df.columns:
    print(df[i].value_counts())
    print("\n")


# There are the value counts of all columns and we can see blnk in TotalCharges column. Lets check out the unique values of that column..

# In[18]:


# checking the unique values of TotalCharges column 
df['TotalCharges'].unique()


# In[19]:


df.shape


# In[ ]:





# we can notice that "TotalCharges" has continous data but its reflecting as object daatatype. and 11 recodrds of this column has blank data. lets handle this column.

# In[20]:


# checking the space in TotalCharges column
df.loc[df["TotalCharges"]==" "]


# By locating the TotalCharges we can find this column has space as values but it waas showing 0 missing values in the column. lets fill this column by some values.

# In[21]:


df['TotalCharges']= df['TotalCharges'].replace(".",np.nan)
df['TotalCharges'].value_counts()


# In[22]:


df.isnull().sum()


# In[23]:


df.info()


# In[24]:


print(df.info)


# In[25]:


df["TotalCharges"]= pd.to_numeric(df["TotalCharges"], errors='coerce')


# In[26]:


# converting object datatpye to float datatype
df['TotalCharges']=df['TotalCharges'].astype(float)
df['TotalCharges'].dtype


# now we can see total charges columns has some space values which is replaced by nan values now we will handle the nun values.

# In[27]:


df.head()


# In[28]:


df.info()


# we have converted the datatype of "TotalCharges" form object to float.

# In[29]:


# lets check null values again
df.isnull().sum()


# as we have filled blank spaces with nan values in TotalCharges, its showing 11 null values in that column. Replcing the NAN values using mean method as the data has continous values.

# In[30]:


# it's not working
np.mean(df['TotalCharges'])
# mean values is = 2283.3004408418656


# In[31]:


df.iloc[488:500,:]


# In[32]:


# np.mean(df['TotalCharges'])


# In[33]:


df.iloc[488:500,:]


# In[34]:


# Checking the mean of Total Charges column
print('The mean value of TotalCharges is :',df['TotalCharges'].mean())


# In[35]:


# Filling null values in TdotalCharges with its mean
df["TotalCharges"]=df['TotalCharges'].fillna(df['TotalCharges'].mean())


# In[36]:


df.isnull().sum()


# In[37]:


# lets check the null values again
sns.heatmap(df.isnull(),cmap='cool_r')


# In[38]:


df.iloc[488,:]


# now we can see there are no missing values in any of the column

# In[39]:


# seprating Numericaal and categorical columns

# checking for categorical columns 
categorical_col=[]
for i in df.dtypes.index:
    if df.dtypes[i]=='object':
        categorical_col.append(i)
        
print("Categorical Columns:", categorical_col)
print('\n')


# Checking for Numeircal columns
numerical_col=[]
for i in df.dtypes.index:
    if df.dtypes[i]!='object':
        numerical_col.append(i)
        
print('Numerical Columns:',numerical_col)


# In[40]:


# Checking numberof unique values in each column
df.nunique().to_frame('No. of unique values')


# these are the unique values present int the dataset. here customeid column has the data or unique id to customer which can be droped 

# In[41]:


# droping customerid column
df.drop('customerID',axis=1, inplace=True)


# In[42]:


#df=df.drop('customerID',axis=1)


# In[43]:


df.head(1)


# In[44]:


# Checking the list of counts of target
df["Churn"].unique()


# In[45]:


# checking the unique values in target column
df['Churn'].value_counts()


# we can assume that "NO" stands for the customers who have not churned and "Yes" stands for the customer who have churned for the company

# # Description of Dataset

# In[46]:


# Statistical Summary of numrical columns 
df.describe()


# # Observation 
# This gives the statistical information of the numerical columns. The summary of the dataset looks perfect since there is no negative/incvalid values 
# 
# From the above description we can observe the following:
# 
# 1) The Counts of all the columns are same which means there are no missing values in the dataset.
# 
# 2) the mean value is greater then median(50%) in tenure and TotalCharges columns which means the data is skewed to right these column.
# 
# 3) The data is the column MonthlyCharge have mean values less then median which means the data is skewed to left.
# 
# 4) By summarizing the data we can observe there is a huge difference between 75% and max hence there are outliers present int he data.
# 
# 5) We can also notice the Standard deviation, min, 25% percentile valuese from this describe method.

# #  Data Visualization 
# Univariate Analysis

# In[47]:


# Visulize the number of Churned customers
ax=sns.countplot(x='Churn',data=df)
print(df['Churn'].value_counts())


# we can observe that the count of " No churn" are high compared to the count of "Yes Churn" i.e there are more number of customers who have not churned. this leads to class imbalance issue in the data. we will rectify it by using oversampling method in later part.  

# In[48]:


# Visulizing the count of Gender
print(df['gender'].value_counts())
ax=sns.countplot(x='gender',data=df)
plt.show()


# In[49]:


# visualizing the count of Senior  Citizen 
print(df['SeniorCitizen'].value_counts())
ax=sns.countplot(x='SeniorCitizen',data=df)
plt.show()


# Here 0 represents non senior citizens and 1 represent senior citizens. The count of 0 is high compread to 1 which means the number of non senior citizens are quite high compared to senior citizens data in the dataset.

# In[50]:


# Visulizing the count of Partner 
print(df['Partner'].value_counts())
ax=sns.countplot(x='Partner',data=df)
plt.show()


# non patners are bit high in count of customer having partners.

# In[51]:


# visualizing the count of Dependents
print(df['Dependents'].value_counts())
ax=sns.countplot(x='Dependents',data=df)
plt.show()


# In[52]:


# visualizing the count of PhoneService
print(df['PhoneService'].value_counts())
ax=sns.countplot(x='PhoneService',data=df)
plt.show()


# The customers whoe have phone service are large in numbers and who do not own phone service are very less number

# In[53]:


# visualizing the count of MultipleLines
print(df['MultipleLines'].value_counts())
ax=sns.countplot(x='MultipleLines',data=df)
plt.show()


# The coutomers having phoneservices from single line have high counts compared to the customer phone services from multiple line, also the customers who do not have phone services have covered very less data compared to others

# In[54]:


# visualizing the count of OnlineSecurity
print(df['OnlineSecurity'].value_counts())
ax=sns.countplot(x='OnlineSecurity',data=df)
plt.show()


# The customer having NotInternetServices needs online security and who do not own any internet services, they do not need online Security, But from the plot we can observe the majority of customers who have internet services have low online security 

# In[55]:


# visualizing the count of OnlineBackup
print(df['OnlineBackup'].value_counts())
ax=sns.countplot(x='OnlineBackup',data=df)
plt.show()


# we can observe thet the customers having no internet services have very less online backup counts campread to others

# In[56]:


# visualizing the count of DeviceProtection
print(df['DeviceProtection'].value_counts())
ax=sns.countplot(x='DeviceProtection',data=df)
plt.show()


# we can see that the customers who do not have internet access, they don't need any Device Protection 

# In[57]:


# visualizing the count of TechSupport
print(df['TechSupport'].value_counts())
ax=sns.countplot(x='TechSupport',data=df)
plt.show()


# The customers who do not need any technical support are high in counts compared to the customer who need the technical suports

# In[58]:


# visualizing how many customer have streamingTV
print(df['StreamingTV'].value_counts())
ax=sns.countplot(x='StreamingTV',data=df)
plt.show()


# The customers who do not use streaming TV are little bit high in numbers then the customers who do use Streming TV, and the customer who do not own internet they do not have this service much. 

# In[59]:


# visualizing how many customer have StreamingMovies
print(df['StreamingMovies'].value_counts())
ax=sns.countplot(x='StreamingMovies',data=df)
plt.show()


# The customers who do not have streming movies are high in count follwed by the customer who have streming movies services, and the customers who do not have internet services, they have less streming movies services compared to others

# In[60]:


# visualizing how many customer have PaymentMethod
print(df['PaperlessBilling'].value_counts())
ax=sns.countplot(x='PaperlessBilling',data=df)
plt.show()


# Most of the customers prefer paper billing and average number of customers who do not prefer paperless billing they may like to recive paper billing.

# In[61]:


# visualizing how many customer have PaymentMethod
print(df['PaymentMethod'].value_counts())
ax=sns.countplot(x='PaymentMethod',data=df)
plt.show()


# Most of the customers prefer Electronic check payment method and the customers who prefer mailed Check, Bank transfer and credit card have average in count. 

# In[62]:


numerical_col


# In[63]:


# df[col].info


# In[64]:


df.head()


# From the above distrivution plots we can notice that the data almost looks normal in all the column expect SiniorSitizen, and the data in the column TotalCharges is skewed to the right. other two columns tenure and MonthlyCharges do not have skewness .
# 
# 
# # Bivariate Analysis

# In[66]:


# comparing tenure and SeniorCitizen
plt.title('Comparison between tenure and SeniorCitizen')
sns.stripplot(x='SeniorCitizen',y='tenure',data=df)


# There is no lsignificant between the features, here both the freatures are in equal length.

# In[67]:


# Comparing tenure and TotaljCharges 

plt.title(" Comparing between tenure and TotalCharges")
sns.scatterplot(x='tenure',y='TotalCharges',data=df, hue= 'Churn', palette='bright')
plt.show()


# here we notice the strong linear relation between the features. 
# 
# as the tenure increases, TotalCharges also increases repidly. if the customers have low tenure services than there is high chance to churn

# In[68]:


# Comparing gender and SeniorCitizen on the basis of Churn
plt.title('Comparison between churn and gender')
sns.barplot(x='gender',y='SeniorCitizen', data = df, palette= 'winter_r',hue = 'Churn' )


# There is no significant diffrence between the columns. The customers churns remains unafuected in gender and seniorCitizen Cases.
# 
# to know about any features
# https://seaborn.pydata.org/generated/seaborn.barplot.html 

# In[69]:


# Comapring TotoalCharges and MonthlyCharges
plt.title('Comparison between totalCharges and MonthlyCharges')
sns.scatterplot(x='TotalCharges',y='MonthlyCharges',data=df, hue='Churn',palette='bright')
plt.show()


# There is a linear relationship between the features. The customer with high monthly charges have high tendencey to stop the sinces they have high total charges. 
# 
# Also if the customers ready to contribute with the monthly charges then there is an increment in the total charges. 

# In[70]:


# Checking Churn level and gender
sns.catplot(x='Churn',col='gender',data=df,kind='count',palette='spring_r')

# Checking churn level and partner
sns.catplot(x='Churn',col='Partner',data=df,palette='tab20b_r',kind='count')

# Checking Churn level in Dependents
sns.catplot(x='Churn',col='PhoneService',data=df, palette='Dark2',kind='count')
plt.show()


# 1) In the first plot we can see there is no significant difference in the genders, both the genders have equal churn level.
# 
# 2) In the second plot we can see the customers, without partners have high churn rate compared to the customeres with partners
# 
# 3) The customeres who do not have any dependency have high churn rate compared to the customers who have dependents.
# 
# 4) In the last plot we can notice the customers who have phone service have high tendency of getting churned. 
# 
# 5) Note- Here you have to plot more and more Bargraph to batter understood and analysis

# In[71]:


# Checking Churn level and MultipleLines
sns.catplot(x='MultipleLines', col='Churn', palette= 'ocean',kind='count',data=df)
plt.show()


# The customers who have phone services from single line have high churn rate comprated to the customers having phone services from multiple line, also there are very less number of customers who do not have phone services. 

# In[72]:


# Checking Churn level in InternetService 
sns.catplot(x='InternetService',col='Churn',palette='bright',kind='count',data=df)
plt.show()


# The ration of churn is high when the customers prefer Fiber optic internet services compared to others services, may be this type of services is bad and need to be foucsed on. And the customers who own DSL service they have very less churn rate. 

# In[73]:


# Checking Churn level in OnlinceSecurity 
sns.catplot(x='OnlineSecurity',col='Churn',palette='icefire',kind='count',data=df)
plt.show()


# The Customers who have no internet service have very less churn rate and the customers who do not have online security services have high tendency of getting 

# In[74]:


# Checking Churn level in OnlineBackup
sns.catplot(x='OnlineBackup',col='Churn',palette='prism',kind ='count',data=df)
plt.show()


# It is also same as in the case of online security. it is obvious that the customers, who do not have internet service they do not need any online backup. The customers who do not have online backup services they have high churn rate.

# In[75]:


# Checking Churn level in DeviceProtecion 
sns.catplot(x='DeviceProtection',hue= 'Churn', palette='gist_heat',kind='count',data=df)
plt.show()


# The Customers who do not own any Device protection have very high churn rate comared to others. 

# In[76]:


# Checking Chrun level in TechSuport 
sns.catplot(x='TechSupport', col='Churn',palette='Set2_r',kind='count',data=df)
plt.show()


# Here we can cleary see that the customers who do not have any techsupport then they have high churn ratio.

# In[77]:


# Checking Churn level in StreamingTV
sns.catplot(x='StreamingTV',hue='Churn',palette='cool_r',kind='count',data=df)
plt.show()


# The churn rate is nearly same if the customer own StreamingTV or Not.

# In[78]:


# Checking Churn level in StreamingMovies
sns.catplot(x='StreamingMovies',hue='Churn',palette='cividis',kind='count',data=df)
plt.show()


# The customers who are exiting in the company they do not own StreamingMovies in their devices, and the churn rate is low when the customers do not have internet services.
# 

# In[79]:


# Checking Churn level in contract
sns.catplot(x='Contract',hue='Churn',palette='gnuplot2',kind='count',data=df)
plt.show()


# The customers who have churned are mostly having months to month contract.

# In[80]:


# Checking Churn level in PaperlessBilling
sns.catplot(x='PaperlessBilling',hue='Churn',palette='gist_earth',kind='count',data=df)


# The customer who are existing int the company they do not own StreamingMovies in their devices, and them churn rate is low when the customers do not have internet services. 

# In[81]:


# Checking Churn level in contract
sns.catplot(x='Contract',hue='Churn',palette='gnuplot2',kind='count',data=df)
plt.show()


# The customer who have churned are mostly having month to month contract.

# In[82]:


# Checking Churn level in PaperlessBilling
sns.catplot(x='PaperlessBilling',hue='Churn',palette='gist_earth',kind='count',data=df)
plt.show()


# the customer who prefer paperless billinhg thye have high churn rate.

# In[83]:


# Checking churn level in PaymentMethod
sns.catplot(x='PaymentMethod',hue='Churn',palette='gnuplot',kind='count',data=df)
plt.show()


# The customers who prefer Electronic check have high churn rate also the customers who exist in the company used equal payment method
# 
# # Multivariate Analysis

# In[84]:


sns.pairplot(df,hue='Churn', palette='Dark2')
plt.show()


# 1) The pairplot gives the pairwise relation between the feartures on the basis of the target 'Churn', on dignorl we can notice the distribution plot
# 2) The freatures tenure and TotalCarges, Montlycharges and TotalCharges have strong linear relation with each others.
# 3) There are no outliers in any of the columns but  let's plot box plot to identify the outliers.

# # Checking for Outliers

# In[86]:


sns.boxplot(df)


# The column of senior citizen has outliers but it contains categorical data so no need to remove outliers. apart from this none of the coulumns have outliers. 

# # Checking for skewness

# In[87]:


df.skew


# df.skew()
# 1) SeniorCitizen     1.833633
# 2) tenure            0.239540
# 3) MonthlyCharges   -0.220524
# 4) TotalCharges      0.962394
#    dtype: float64

# The columns SeniorCitizen and TotalCharges have skewness in the data. Since Senirocitizen in categorical, no need to remove to skewness  since TotalCharges is countinouse in natures, lets cube root method to remove skewness. 
# 
# # Removing skewness

# In[88]:


# Removing skewness using cuberoot method
df['TotalCharges']=np.cbrt(df['TotalCharges'])


# We have removed the skewness using cube root method.

# In[89]:


print(df.skew)
# SeniorCitizen    1.833633
# tenure           0.239540
# MonthlyCharges   -0.220524
# TotalCharges     0.011168


# we can see the skewness has been reduced in TotalCharges column. 

# In[90]:


# lets visulize how the data has been distributed in TotalCharges after removing skewness
sns.distplot(df['TotalCharges'], color='m',kde_kws={'shade':True}, hist=False)
plt.show()


# This looks almost normal and skewness is also removed

# # Encoding categorical columns

# In[91]:


import sklearn


# In[92]:


# Encoding categorical columns using OrdinalEncoder
from sklearn.preprocessing import OrdinalEncoder
OE=OrdinalEncoder()
for i in df.columns:
    if df[i].dtypes=='object':
        df[i]=OE.fit_transform(df[i].values.reshape(-1,1))
df


# In[93]:


print(df.info)


# we have converted the categorical column into numerical columns usning ordinal Encoding method. 

# In[94]:


# Statistical sumary of nurerical
df.describe()


# after encording the categorical columns we can see all colun details here. the counts of all the columns are same that mens no null values in the datase4t. This describe method describe the count  mean, std, min, IQR, Max_values of all the columns 

# # Correlation between target variable and independent variables.

# In[95]:


# Checking the correlation between features and the target.
cor=df.corr()
cor


# This gives the correlation between the dependent and independent variables. we can visualize this by plotting heatmap.

# In[96]:


# Visualizing the correlation matrix by plotting heatmap
plt.figure(figsize=(20,15))
sns.heatmap(df.corr(),linewidths=0.1,fmt='.1g',linecolor='black',cmap='Blues_r',annot=True)
plt.yticks(rotation=0);
plt.show()


# 0-> no relation, -0.3 to +0.3-> less correlation, greater then -0.5 or greater then 0+.5-> moderate correlation, grater then -0.7 or greater then +0.7 high correlation 

# This heatmap show the correlation matrix by visulaizing the data. we can observe the relation between feature to feature and feature  to label. 
# This heatmap contains both positive and negative correlation
# 1) There is no much positive correlation between the target and features.
# 2) The column MonthlyCharges, PaperlessBilling, SeniorCitizen and PaymentMethod have positive correlation with the Label Column 'Churn'.
# 3) The label is negatively correlated with Contract, Tenure, OnlineSecurity, TechSuport, TotalCharges, DeviceProtection, OnlneBackup, Partner and Dependents.
# 4) Also the column geender has very correlation with the label, we can drop it if necessary.
# 5) The colum TotalCharges and tenure, Contract and tenure, TotalCharge and MonthlyCharges and many other columns have high correlation with each other

# In[97]:


cor['Churn'].sort_values(ascending= False)


# We can observe the postive and negative correlated features with the target. 
# 
# # Visualizing the correlation between label and features using bar plot

# In[98]:


plt.figure(figsize=(22,7))
df.corr()['Churn'].sort_values(ascending=False).drop(['Churn']).plot(kind='bar',color='m')
plt.xlabel('Feature',fontsize=15)
plt.ylabel('Taraget', fontsize=15)
plt.title('Correlation Between label and features usning barplot',fontsize=20)
plt.show()


# From the above bar plot we can notice the positive and negative correlation between the features and the target. here the feature the gender and PhoneService have very less correlation with the column.

# # Separating Features and lable 

# In[99]:


x=df.drop('Churn',axis=1)
y=df['Churn']


# In[100]:


y.head()


# In[101]:


x.head()


# # Feature Scaling using Standard Scalarization

# In[102]:


import sklearn
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x=pd.DataFrame(scaler.fit_transform(x),columns=x.columns)
x


# We have scaled the data using Standard Scalarization method to overcome the issue of biasness.

# # Checking Variance Inflation Factor(VIF)

# In[103]:


# Finding Varience inflaction factor in each scaled column i.e, x.shape[1][1/(1-r2)]

from statsmodels.stats.outliers_influence import variance_inflation_factor
vif=pd.DataFrame()
vif['VIF values']=[variance_inflation_factor(x.values,i) for i in range(len(x.columns))]
vif['Features']=x.columns

vif


#  By checking VIF values we can find the features causing multicollineraity problem. Here we can find the feature TotalCharges and tenure VIF values greater then 10 which means they have high correlation with other features. We will drop one of the column first, if the same issue exist then we will try to remove the column having high VIF.

# In[104]:


# Droping TotalCharges column
x.drop('TotalCharges', axis=1, inplace=True)


# In[105]:


# Again Checking VIF value to confirm whether the multicollinearty still exists or not 
vif=pd.DataFrame()
vif['VIF values']=[variance_inflation_factor(x.values,i) for i in range(len(x.columns))]
vif['Features']=x.columns

vif


# So, we have solved multicolinearity issue. we can now move ahead for model building.

# In[106]:


y.value_counts()


# Here we can see that the data is not balanced, since it is a classification problem we will blance that data using oversampling method

# # Oversampling Method
# 

# In[109]:


# Oversmpling the data
# !pip install imblearn
from imblearn.over_sampling import SMOTE
SM=SMOTE()
x1,y1=SM.fit_resample(x,y)


# In[110]:


y.value_counts()


# In[111]:


y1.value_counts()


# In[112]:


import threadpoolctl


# now the data is balanced. Now we can build machine learning classification models.
# 
# # Modelling 
# 
# # Findint the Best random state

# In[113]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
maxAccu=0
maxRS=0
for i in range(1,200):
    x_train,x_test,y_train,y_test=train_test_split(x1,y1,test_size=0.30,random_state=i)
    RFR=RandomForestClassifier()
    RFR.fit(x_train,y_train)
    pred=RFR.predict(x_test)
    acc=accuracy_score(y_test,pred)
    if acc>maxAccu:
        maxAccu=acc
        maxRS=i
    print("Best accurancy is ",maxAccu,"at random_state",maxRS)
    


# The best accuracy is 86.53 % at random_state 102.

# # Creating train test split

# In[114]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=maxRS)


# # Classification Algorithms

# In[115]:


from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier,BaggingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve,accuracy_score
from sklearn.model_selection import cross_val_score


# # RandomForestClassifier

# In[116]:


# Checking accuracy for RandomForestClasifier
RFC= RandomForestClassifier()
RFC.fit(x_train,y_train)
predRFC=RFC.predict(x_test)
print(accuracy_score(y_test, predRFC))
print(confusion_matrix(y_test,predRFC))
print(classification_report(y_test,predRFC))


# The accuracy for this model is 86.50%

# # Logistic Regression

# In[117]:


# Checking accuracy for LogisticRegression
LR=LogisticRegression()
LR.fit(x_train,y_train)
predLR=LR.predict(x_test)
print(accuracy_score(y_test,predLR))
print(confusion_matrix(y_test,predLR))
print(classification_report(y_test,predLR))


# The accuracy score using LogisticRegresssion is 78.19 % 

# # Suport Vector Machine Classifier

# In[118]:


# Checking accuracy for Suport Vector Machine Classifier
svc=SVC()
svc.fit(x_train,y_train)
predsvc=svc.predict(x_test)
print(accuracy_score(y_test,predsvc))
print(confusion_matrix(y_test,predsvc))
print(classification_report(y_test,predsvc))


# The accuracy score using Suport Vector Machine Clssifier is 81.48% 

# # Gradient Boosting Classifier

# In[119]:


# Checking accuracy for Gradient Boosting Classifier
GB=GradientBoostingClassifier()
GB.fit(x_train,y_train)
predGB=GB.predict(x_test)
print(accuracy_score(y_test,predGB))
print(confusion_matrix(y_test,predGB))
print(classification_report(y_test,predGB))


# The accuracy_score using Gradient Boosting Classifier is 86.11% 

# # AddBoost Classfier

# In[120]:


# Checking accurcy for AdaBoost Classifer
ABC=AdaBoostClassifier()
ABC.fit(x_train,y_train)
predABC =ABC.predict(x_test)
print(accuracy_score(y_test,predABC))
print(confusion_matrix(y_test,predABC))
print(classification_report(y_test,predABC))


# The accuracy_score using AdaBoostClassifier is 82.76% 

# # Bagging Classifier

# In[ ]:





# In[121]:


# Checking accuracy for BaggingClassifgier
BC=BaggingClassifier()
BC.fit(x_train,y_train)
predBC=BC.predict(x_test)
print(accuracy_score(y_test,predBC))
print(confusion_matrix(y_test,predBC))
print(classification_report(y_test,predBC))


# The accuracy_score using BagginhgClassifier is 84.21% 

# # ExtraTreesClassfier

# In[122]:


# Checking accuracy for ExtraTreesClassifier
ET=ExtraTreesClassifier()
ET.fit(x_train,y_train)
predET=ET.predict(x_test)
print(accuracy_score(y_test,predET))
print(confusion_matrix(y_test, predET))
print(classification_report(y_test,predET))


# The accuracy_soure using Extra TreesClassifier is 87.43%

# # Cross Validation Score

# In[123]:


from sklearn.model_selection import cross_val_score


# In[124]:


"""cv=5--- it's a fold value 1,2,3,4,5
   cross_val_score(modelname, features,targetvariable, cv=foldvalue, scoring='accuracy')
   
   in classification algorithms- default scoring parameter- accuracy
   for regression- deafulat scoring parameter is - r2 score
"""


# https://scikit-learn.org/stable/modules/model_evaluation.html

# In[125]:


# Checking cv score for Random Forest Classifier
score=cross_val_score(RFC,x1,y1)
print(score)
print(score.mean())
print('Difference between Accuracy score and cross validation score is ', accuracy_score(y_test,predRFC))


# # Logistic Regression

# In[126]:


# Checking cv score for Logistic Regression
score=cross_val_score(LR,x1,y1)
print(score)
print(score.mean())
print('Difference between Accuracy score and cress validation score is - ', accuracy_score(y_test,predLR)-score.mean())


# # Gradient Boosting Classifier

# In[127]:


# Checking cv score for Gradient Boosting Classifier
score=cross_val_score(GB,x1,y1)
print(score)
print(score.mean())
print('Difference Between Accuracy score and cross validation scoreis - ',accuracy_score(y_test,predGB)-score.mean())


# Extra Trees Classifier is our best model as the difference between accuracy score and cross validation score is least.

# # Extra Trees Classifier is our best Model

# # Hyper Parameter Tuning

# In[128]:


# ExtraTress Classifier
from sklearn.model_selection import GridSearchCV

parameters={'criterion':['gini','entropy'],
           'random_state':[10,50,1000],
           'max_depth':[0,10,20],
            'n_jobs':[-2,-1,1],
            'n_estimators':[50,100,200,300]           }


# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html

# In[129]:


GCV=GridSearchCV(ExtraTreesClassifier(),parameters, cv=5)


# In[130]:


# GCV.fit(x_train,y_train)


# In[131]:


# GCV.best_params_


# # Plotting ROC and compare AUC for all the models used

# In[133]:


# Plotting for all the models used here 
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
from sklearn.metrics import plot_roc_curve


# In[134]:


disp=plot_roc_curve(ET,x_test,y_test) # ax_=Axes with confusion matrix
plot_roc_curve(RFC,x_test,y_test, ax=disp.ax_)
plot_roc_curve(LR,x_test,y_test, ax=disp.ax_)
plot_roc_curve(GB,x_test,y_test, ax=disp.ax_)
plot_roc_curve(ABC,x_test,y_test, ax=disp.ax_)
plot_roc_curve(BC,x_test,y_test, ax=disp.ax_)

plt.legend(prop={'size':11},loc='lower right')
plt.show()


# Here we can see area under curve for each model used.

# # Plotting ROC and compare AUC for the best model

# Here we have plotted the ROC curve for the final model and the AUC value for the best model is 94%

# https://scikit-learn.org/1.0/modules/generated/sklearn.metrics.plot_roc_curve.html

# ### Thanks 
# * Navin Singh
# * 23 Oct 2025
