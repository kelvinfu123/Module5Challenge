# %%
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import glob
import csv
from scipy import stats

# %%
file= open(r"C:\Users\Kelvin\source\repos\Module5Challenge\Study_results.csv")
with open(r"C:\Users\Kelvin\source\repos\Module5Challenge\Study_results.csv") as file:
 csvreader_studyr = csv.reader(file)
 studyreport = pd.DataFrame(csvreader_studyr)


file2= open(r"C:\Users\Kelvin\source\repos\Module5Challenge\Mouse_metadata.csv")
with open(r"C:\Users\Kelvin\source\repos\Module5Challenge\Mouse_metadata.csv") as file:
 csvreader_mouse = csv.reader(file2)
 mouse = pd.DataFrame(csvreader_mouse)
 


# %%
left = pd.DataFrame(mouse)

left.columns = left.iloc[0]
left = left[1:]



# %%
studyrepot = pd.DataFrame(studyreport)

studyrepot.columns = studyrepot.iloc[0]
studyrepot = studyrepot[1:]



# %%
csv_files = glob.glob('*.{}'.format('csv'))
print(csv_files)

# %%
df_csv_concat = pd.DataFrame()

 
df_csv_concat = pd.concat([pd.read_csv(file) for file in csv_files ], ignore_index=True)
print(df_csv_concat)

# %%
right = pd.DataFrame(studyrepot)
right = studyrepot[studyrepot["Timepoint"] == str(45)]


# %%
df_csv_concat = pd.DataFrame()

# %%
df_csv_concat = pd.concat([pd.read_csv(file) for file in csv_files ], ignore_index=True)
print(df_csv_concat)

# %%


clean_df_csv_concat = pd.DataFrame(df_csv_concat)

clean_df_csv_concat = pd.merge(left, right, left_on ="Mouse ID", right_on = "Mouse ID" , how='inner')
print(clean_df_csv_concat)

print("The number of Mice is :" ,len(clean_df_csv_concat))

# %%
clean_df_csv_concat['Tumor Volume (mm3)'] = pd.to_numeric(clean_df_csv_concat['Tumor Volume (mm3)'])

# %%
mean = clean_df_csv_concat.groupby("Drug Regimen")["Tumor Volume (mm3)"].mean()
median = clean_df_csv_concat.groupby("Drug Regimen")["Tumor Volume (mm3)"].median()
variance = clean_df_csv_concat.groupby("Drug Regimen")["Tumor Volume (mm3)"].var()
sd = clean_df_csv_concat.groupby("Drug Regimen")["Tumor Volume (mm3)"].std()

n_samples = len(right)
sem = sd / np.sqrt(n_samples)

# %%
statistics = pd.DataFrame({"Mean": mean,
                           "Median": median,
                           "Variance": variance,
                           "SD": sd,
                           "SEM": sem})

print(statistics)

# %%
value = (clean_df_csv_concat["Mouse ID"]).count()
print(value)

# %%

drug = {
         "Drug Regimen" :
         ['Capomulin ' , 'Ceftamin' , 'Infubinol' ,"Ketapril","Naftisol","Placebo","Propriva","Ramicane","Stelasyn","Zoniferol"]}


count = (clean_df_csv_concat.groupby("Drug Regimen")["Mouse ID"]).count()

Count = pd.DataFrame({"Count": count, 
                      #"Drug Regimen" : drug 
                      })

# %%
Count.plot( y = "Count" , kind = "bar" , color = "blue" , title = "Count of mices remaining through respective treaments")

# %%
drug_list = ['Capomulin ' , 'Ceftamin' , 'Infubinol' ,"Ketapril","Naftisol","Placebo","Propriva","Ramicane","Stelasyn","Zoniferol"]
yy = list(count)


plt.bar(drug_list , yy , width = 0.5 ,alpha = 0.5 , align = 'center')


plt.xlabel('Drug Regimen')
plt.xticks(rotation = 90)
plt.ylabel('Count')
plt.title('Count of mices remaining through respective treaments')
plt.grid(False)
plt.show()

# %%
s = (clean_df_csv_concat.groupby("Sex")["Drug Regimen"]).count()
S = pd.DataFrame(s)


# %%
S.plot(kind='pie', y='Drug Regimen', autopct='%1.1f%%', legend=False)
plt.axis('equal')  # Equal aspect ratio ensures that the pie chart is drawn as a circle.
plt.title('Gender Distribution of Mice')
plt.show()

# %%
gender_list = ["Female" , "Male"]


plt.pie( s, labels = gender_list, autopct='%1.1f%%')
plt.title('Gender Distribution of Mice')
plt.show()

# %%
right = pd.DataFrame(studyrepot)
right = studyrepot[studyrepot["Timepoint"] <= str(45)]

resetr = pd.DataFrame(right)
resetr = right.reset_index(drop=True)
resetr.index = resetr.index + 1 
print(right)
#print(resetr)
x=len(right)
print(x)

# %%
max_time = (right.groupby("Mouse ID")["Timepoint"]).max()
drug_type =(left.groupby("Mouse ID")["Drug Regimen"]).unique()
tumor_volume = (right.groupby("Mouse ID")["Tumor Volume (mm3)"]).max()

max_time_df = pd.DataFrame(max_time)
drug_type_df = pd.DataFrame(drug_type)
tumor_volume_df = pd.DataFrame(tumor_volume)



# %%
max_time_drug_type = pd.DataFrame()
max_time_drug_type = pd.merge(drug_type_df, max_time_df, left_on ="Mouse ID", right_on = "Mouse ID" , how='inner')


# %%
max_time_tumor_volume = pd.DataFrame()
max_time_tumor_volume = pd.merge(max_time_df, tumor_volume_df, left_on ="Mouse ID", right_on = "Mouse ID" , how='inner')

result = pd.DataFrame()
result = pd.merge(max_time_drug_type, max_time_tumor_volume, left_on ="Mouse ID", right_on = "Mouse ID" , how='inner')


print(result)

# %%
box_table = pd.DataFrame(result)
box_table = box_table.drop(columns='Timepoint_x')
box_table = box_table.drop(columns='Timepoint_y')

print(box_table)

# %%
box_table_capomulin = pd.DataFrame(box_table)
box_table_capomulin = box_table[box_table["Drug Regimen"] == "Capomulin"]
print(box_table_capomulin)

box_table_ramicane = pd.DataFrame(box_table)
box_table_ramicane = box_table[box_table["Drug Regimen"] == "Ramicane"]
print(box_table_ramicane)

box_table_infubinol = pd.DataFrame(box_table)
box_table_infubinol = box_table[box_table["Drug Regimen"] == "Infubinol"]
print(box_table_infubinol)

box_table_ceftamin = pd.DataFrame(box_table)
box_table_ceftamin = box_table[box_table["Drug Regimen"] == "Ceftamin"]
print(box_table_ceftamin)

clean_df_csv_concat['Tumor Volume (mm3)'] = pd.to_numeric(clean_df_csv_concat['Tumor Volume (mm3)'])


# %%
col_list_capomulin = box_table_capomulin['Tumor Volume (mm3)'].to_numpy(dtype=float)
col_list_capomulin = np.sort(col_list_capomulin)


col_list_ramicane = box_table_ramicane['Tumor Volume (mm3)'].to_numpy(dtype=float)
col_list_ramicane = np.sort(col_list_ramicane)


col_list_infubinol = box_table_infubinol['Tumor Volume (mm3)'].to_numpy(dtype=float)
col_list_infubinol = np.sort(col_list_infubinol)


col_list_ceftamin = box_table_ceftamin['Tumor Volume (mm3)'].to_numpy(dtype=float)
col_list_ceftamin = np.sort(col_list_ceftamin)


# %%
def cal_iqr_and_outliers(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    print("Lower Bound:",lower_bound)
    upper_bound = q3 + 1.5 * iqr
    print("Upper Bound:",upper_bound)

    for x in data :
        if x <lower_bound or x > upper_bound:
            print(x)
            
    
    

    
    print("IQR:", iqr)
    print("Outliers:",x)
    print()

# Display IQR and outliers for each treatment group
print("Treatment Group Capomulin:")
cal_iqr_and_outliers(col_list_capomulin)

print("Treatment Group Ramicane:")
cal_iqr_and_outliers(col_list_ramicane)

print("Treatment Group Infubinol:")
cal_iqr_and_outliers(col_list_infubinol)

print("Treatment Group Ceftamin:")
cal_iqr_and_outliers(col_list_ceftamin)


# %%

capomulin_s = pd.Series(col_list_capomulin, name='Capomulin')
ramicane_s = pd.Series(col_list_ramicane, name='Ramicane')
infubinol_s  = pd.Series(col_list_infubinol,name='Infubinol')
ceftamin_s = pd.Series(col_list_infubinol,name='Ceftamin')



df = pd.concat({'Capomulin': capomulin_s,
              'Ramicane': ramicane_s,
              'Infubinol': infubinol_s,
              'Ceftamin' : ceftamin_s,
              },axis=1)



b_plot = df.boxplot(column = ['Capomulin', 'Ramicane', 'Infubinol','Ceftamin']) 



df.boxplot(showfliers = True )

plt.title("Final Tumor Volume of Mice ")
plt.xlabel("Columns")
plt.ylabel("Values")
plt.show()



# %%
sorted_df = df_csv_concat.sort_values(by='Mouse ID')


line_plot = pd.DataFrame(sorted_df)
line_plot = sorted_df[sorted_df["Mouse ID"] == "l509"]





# %%
sorted_line = line_plot.sort_values(by='Timepoint')

sorted_line = sorted_line.iloc[0:10]


# %%
plt.plot(sorted_line["Timepoint"], sorted_line["Tumor Volume (mm3)"])
plt.xlabel('Timepoint')  
plt.ylabel('Tumor Volume (mm3)')  
plt.title('Tumor Volume of one mouse against Time')  
plt.xlim(0, max(sorted_line['Timepoint']+5))
plt.ylim(40, max(sorted_line['Tumor Volume (mm3)']+5))
plt.grid(True)  
plt.show()

# %%
mouse_weight = (left.groupby("Mouse ID")["Weight (g)"]).unique()
scatterp = pd.merge(mouse_weight , df_csv_concat , left_on ="Mouse ID", right_on = "Mouse ID" , how='outer')



# %%
plt.scatter(scatterp['Weight (g)_x'], scatterp['Tumor Volume (mm3)'])
plt.xlabel('Weight(g)')
plt.ylabel('Tumor Volume (mm3)')
plt.title('Tumor Volume against Weight')
plt.show()

scatterp['Tumor Volume (mm3)'] = pd.to_numeric(scatterp['Tumor Volume (mm3)'])

# %%
selected_columns = ['Weight (g)_x', 'Tumor Volume (mm3)']
cox = scatterp[selected_columns].corr()
correalation = cox["Tumor Volume (mm3)"].tolist()
print("The correlation coefficient is :" ,correalation[0])
corrrr = correalation[0]
print(corrrr)

# %%
x1 = pd.DataFrame(scatterp)
x1 = scatterp['Weight (g)_x'].to_numpy(dtype=int)



# %%
y = scatterp['Tumor Volume (mm3)'].to_numpy(dtype=int)

slope, intercept, r_value, p_value, std_err = stats.linregress(x1 , y)

# %%
m = np.polyfit(x1, y,1)


regression_line = np.poly1d(corrrr)

#print(regression_line)

plt.scatter(x1, y)

plt.plot(x1, intercept + slope * x1, 'r', label='Regression Line')

plt.xlabel('Weight(g)')
plt.ylabel('Tumor Volume (mm3)')
plt.title('Tumor Volume against Weight')
plt.show()



