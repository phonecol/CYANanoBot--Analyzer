import pandas as pd 

csv_path = "79ROI22021_12_01-14_00_02x.csv"
data = pd.read_csv(csv_path)
print(data.head())
# conc = df['# Cyanide Concentration']
# print(df['# Cyanide Concentration'])

# for i in range(len(conc)):
    
#     for j in range(3):
#         if conc[i] == conc[j]:
#             conce = df['# Cyanide Concentration'].mean()
#             print(conce)

# print(df.loc[df['# Cyanide Concentration'] == 0.0])
# print(df.loc[df['# Cyanide Concentration'] == 0.0].mean())
# print(df.loc[df['# Cyanide Concentration'] == 14.1].mean())
# print(df.loc[df['# Cyanide Concentration'] == 21.7].mean())
# print(df.loc[df['# Cyanide Concentration'] == 32.5].mean())
# print(df.loc[df['# Cyanide Concentration'] == 47.8].mean())
# print(df.loc[df['# Cyanide Concentration'] == 55.2].mean())
# print(df.loc[df['# Cyanide Concentration'] == 62].mean())
# print(df.loc[df['# Cyanide Concentration'] == 70.3].mean())
# print(df.loc[df['# Cyanide Concentration'] == 75].mean())
# print(df.loc[df['# Cyanide Concentration'] == 89.3].mean())
# print(df.loc[df['# Cyanide Concentration'] == 112].mean())

df1 = data.loc[data['# Cyanide Concentration'] == 0.0].mean(axis=0)
df2 = data.loc[data['# Cyanide Concentration'] == 14.1].mean(axis=0)
df1_transposed = df1.T
df2_transposed = df2.T
# # print(df1.T)
# df3 = pd.concat(df1_transposed,df2_transposed)
# print(df3)
# df1_transposed.to_csv('file1.csv',index=False)

df4 = data[data['# Cyanide Concentration'] == 0.0].mean()
print(df4)

df5 = data.groupby('# Cyanide Concentration').R.mean()
print(df5)
df6 = data.groupby(['# Cyanide Concentration']).mean()
print(df6)
df6.to_csv('file7.csv')
# print("okay")
# frames = [df1,df2]
data2 = pd.read_csv('file7.csv')
print(data2.head)
# # result = pd.concat(frames)
# result = df1.append(df2)
# print(result)
# result_t = result.T
# result_t.to_csv('file2.csv',index=False)
# # result.to_csv('file1.csv',index=False)

# print("okay")
# frames.to_csv(index=False)

# print(mean_list)