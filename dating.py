import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
#from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC

df = pd.read_csv("profiles.csv")

print(df.columns)
print(len(df))
#print(df.age.head())
#print(df.age.value_counts())
#print(df.job.value_counts())
#print('Min: ', min(df.age))
#print('Max: ', max(df.age))
#print(df.body_type.value_counts())

# plt.hist(df.age, bins=25)
# plt.xlabel("Age")
# plt.ylabel("Frequency")
# plt.xlim(16, 80)
# plt.show()

#print(df.sign.value_counts())
#print(df.speaks.value_counts())
#print(df.education.value_counts())
#print(df.income.value_counts())

# Augmenting data
#drink_mapping = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}
#df["drinks_code"] = df.drinks.map(drink_mapping)
#print(df.drinks_code.value_counts())

#print(df.smokes.value_counts())
#smoke_mapping = {"no": 0, "sometimes": 1, "when drinking": 2, "trying to quit": 3, "yes": 4}
#df["smokes_code"] = df.smokes.map(smoke_mapping)
#print(df.smokes_code.value_counts())

# print(df.drugs.value_counts())
# drugs_mapping = {"never": 0, "sometimes": 1, "often": 2}
# df["drugs_code"] = df.drugs.map(drugs_mapping)
# print(df.drugs_code.value_counts())

# print(df.religion.value_counts())
# religion_mapping = {"atheism and very serious about it": 1,
#                     "atheism and somewhat serious about it": 2,
#                     "atheism": 3,
#                     "atheism but not too serious about it": 4,
#                     "atheism and laughing about it": 5,
#                     "agnosticism and very serious about it": 6,
#                     "agnosticism and somewhat serious about it": 7,
#                     "agnosticism": 8,
#                     "agnosticism but not too serious about it": 9,
#                     "agnosticism and laughing about it": 10,
#                     "buddhism and very serious about it": 11,
#                     "buddhism and somewhat serious about it": 12,
#                     "buddhism": 13,
#                     "buddhism but not too serious about it": 14,
#                     "buddhism and laughing about it": 15,
#                     "catholicism and very serious about it": 16,
#                     "catholicism and somewhat serious about it": 17,
#                     "catholicism": 18,
#                     "catholicism but not too serious about it": 19,
#                     "catholicism and laughing about it": 20,
#                     "christianity and very serious about it": 21,
#                     "christianity and somewhat serious about it": 22,
#                     "christianity": 23,
#                     "christianity but not too serious about it": 24,
#                     "christianity and laughing about it": 25,
#                     "hinduism and very serious about it": 26,
#                     "hinduism and somewhat serious about it": 27,
#                     "hinduism": 28,
#                     "hinduism but not too serious about it": 29,
#                     "hinduism and laughing about it": 30,
#                     "judaism and very serious about it": 31,
#                     "judaism and somewhat serious about it": 32,
#                     "judaism": 33,
#                     "judaism but not too serious about it": 34,
#                     "judaism and laughing about it": 35,
#                     "islam and very serious about it": 36,
#                     "islam and somewhat serious about it": 37,
#                     "islam": 38,
#                     "islam but not too serious about it": 39,
#                     "islam and laughing about it": 40,
#                     "other and very serious about it": 41,
#                     "other and somewhat serious about it": 42,
#                     "other": 43,
#                     "other but not too serious about it": 44,
#                     "other and laughing about it": 45}
# df["religion_code"] = df.religion.map(religion_mapping)
# print(df.religion_code.value_counts()/len(df.religion_code))
# print(len(df.religion_code))

# religiosity_mapping = {"atheism and very serious about it": 0,
#                        "atheism and somewhat serious about it": 0,
#                        "atheism": 0,
#                        "atheism but not too serious about it": 0,
#                        "atheism and laughing about it": 0,
#                        "agnosticism and very serious about it": 0,
#                        "agnosticism and somewhat serious about it": 0,
#                        "agnosticism": 0,
#                        "agnosticism but not too serious about it": 0,
#                        "agnosticism and laughing about it": 0,
#                        "buddhism and very serious about it": 3,
#                        "buddhism and somewhat serious about it": 3,
#                        "buddhism": 2,
#                        "buddhism but not too serious about it": 1,
#                        "buddhism and laughing about it": 1,
#                        "catholicism and very serious about it": 3,
#                        "catholicism and somewhat serious about it": 3,
#                        "catholicism": 2,
#                        "catholicism but not too serious about it": 1,
#                        "catholicism and laughing about it": 1,
#                        "christianity and very serious about it": 3,
#                        "christianity and somewhat serious about it": 3,
#                        "christianity": 2,
#                        "christianity but not too serious about it": 1,
#                        "christianity and laughing about it": 1,
#                        "hinduism and very serious about it": 3,
#                        "hinduism and somewhat serious about it": 3,
#                        "hinduism": 2,
#                        "hinduism but not too serious about it": 1,
#                        "hinduism and laughing about it": 1,
#                        "judaism and very serious about it": 3,
#                        "judaism and somewhat serious about it": 3,
#                        "judaism": 2,
#                        "judaism but not too serious about it": 1,
#                        "judaism and laughing about it": 1,
#                        "islam and very serious about it": 3,
#                        "islam and somewhat serious about it": 3,
#                        "islam": 2,
#                        "islam but not too serious about it": 1,
#                        "islam and laughing about it": 1,
#                        "other and very serious about it": 3,
#                        "other and somewhat serious about it": 3,
#                        "other": 2,
#                        "other but not too serious about it": 1,
#                        "other and laughing about it": 1}
# df["religiosity_code"] = df.religion.map(religiosity_mapping)
# print(df.religiosity_code.value_counts())

print("\n## Data mapping")
# print(df.diet.value_counts())
diet_mapping = {"strictly vegan": 0,
                "vegan": 0,
                "mostly vegan": 1,
                "strictly vegetarian": 2,
                "vegetarian": 2,
                "mostly vegetarian": 3,
                "strictly anything": 4,
                "anything": 4,
                "mostly anything": 5,
                "strictly other": 6,
                "other": 6,
                "mostly other": 7,
                "strictly kosher": 8,
                "kosher": 8,
                "mostly kosher": 9,
                "strictly halal": 10,
                "halal": 10,
                "mostly halal": 11}
df["diet_code"] = df.diet.map(diet_mapping)
# print(df.diet_code.value_counts())
# print(len(df.diet_code))

# print(df.body_type.value_counts())
body_type_mapping = {"used up": 0,
                     "skinny": 1,
                     "thin": 2,
                     "average": 3,
                     "fit": 4,
                     "athletic": 5,
                     "curvy": 6,
                     "full figured": 7,
                     "a little extra": 8,
                     "jacked": 9,
                     "overweight": 10,
                     "rather not say": 11}
df["body_type_code"] = df.body_type.map(body_type_mapping)
# print(df.body_type_code.value_counts())
# print(len(df.body_type_code))

print(df.sex.value_counts()/len(df.sex))
sex_mapping = {"m": 0, "f": 1}
df["sex_code"] = df.sex.map(sex_mapping)
# print(df.sex_code.value_counts())
# print(len(df.sex_code))

print("\n### Normalization:")
scaler = MinMaxScaler(feature_range=(0, 19))
df["age_scaled"] = scaler.fit_transform(df[["age"]])
print("Data min: ", scaler.data_min_, " -> ", scaler.transform([scaler.data_min_]))
print("Data max: ", scaler.data_max_, " -> ", scaler.transform([scaler.data_max_]))
print("18 -> ", scaler.transform([[18]]))
print("70 -> ", scaler.transform([[70]]))

print("\n## Data classification")
wanted_features = ["diet_code", "body_type_code", "sex_code", "age"]
rows_to_cluster = df.dropna(subset = wanted_features)

# print("\n### KMeans classification")
# columns_to_fit = rows_to_cluster[wanted_features]
# classifier = KMeans(n_clusters=3)
# classifier.fit(columns_to_fit)
# print("Diet - Body type - Age")
# print(classifier.cluster_centers_)
# 
# print(classifier.labels_)
# cluster_zero_indices = []
# cluster_one_indices = []
# cluster_two_indices = []
# for i in range(len(classifier.labels_)):
#     if classifier.labels_[i] == 0:
#         cluster_zero_indices.append(i)
#     elif classifier.labels_[i] == 1:
#         cluster_one_indices.append(i)
#     elif classifier.labels_[i] == 2:
#         cluster_two_indices.append(i)
# 
# cluster_zero_df = rows_to_cluster.iloc[cluster_zero_indices]
# cluster_one_df = rows_to_cluster.iloc[cluster_one_indices]
# cluster_two_df = rows_to_cluster.iloc[cluster_two_indices]
# 
# print(cluster_zero_df["sex"].value_counts()/len(cluster_zero_df))
# print(cluster_one_df["sex"].value_counts()/len(cluster_one_df))
# print(cluster_two_df["sex"].value_counts()/len(cluster_two_df))


data_set = rows_to_cluster[["diet_code", "body_type_code", "age_scaled"]]
label_set = rows_to_cluster["sex_code"]
# encoder = LabelEncoder()
# age_encoded = encoder.fit_transform(label_set)

train_set, test_set, train_labels, test_labels = train_test_split(data_set, label_set, train_size=0.8, test_size=0.2, random_state=42)
# 
# 
# print("\n### K-Nearest Neighbour Classification")
# 
# k_range = range(1, 201)
# scores = []
# best_score = (0, 0)
# for k in k_range:
#     classifier = KNeighborsClassifier(n_neighbors = k)
#     classifier.fit(train_set, train_labels)
#     this_score = classifier.score(test_set, test_labels)
#     scores.append(this_score)
#     if this_score > best_score[1]:
#         best_score = (k, this_score)
#     #print("K-Nearest Neighbour, n=", k, ", score: ", scores)
# print("KNN: k=", best_score[0], " with score=", best_score[1])
# 
# plt.plot(k_range, scores)
# plt.xlabel("k")
# plt.ylabel("score")
# plt.show()

print("\n### K-Nearest Neighbour Regressor")

regressor = KNeighborsRegressor(n_neighbors = 75, weights = "distance")
regressor.fit(train_set, train_labels)
this_score = regressor.score(test_set, test_labels)
print(this_score)

# ----------

# print("\n### Support Vector Classifier")
# 
# c_range = range(1, 21)
# scores = []
# best_score = (0, 0)
# for k in c_range:
#     classifier = SVC(C = k)
#     classifier.fit(train_set, train_labels)
#     this_score = classifier.score(test_set, test_labels)
#     scores.append(this_score)
#     if this_score > best_score[1]:
#         best_score = (k, this_score)
# print("SVC: C=", best_score[0], " with score=", best_score[1])
# 
# plt.plot(c_range, scores)
# plt.xlabel("c")
# plt.ylabel("score")
# plt.show()

# classifier = SVC(C = 3, gamma = 'auto')
# classifier.fit(train_set, train_labels)
# this_score = classifier.score(test_set, test_labels)
# print("SVC: C=3, gamma='auto', score: ", this_score)


# Combining essays
#essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]
#all_essays = df[essay_cols].replace(np.nan, '', regex=True)
#all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)
#df["essay_len"] = all_essays.apply(lambda x: len(x))
#print(df.essay_len.value_counts())

# print('Orientation:')
# print(df.orientation.value_counts())
# print('Sex:')
# print(df.sex.value_counts())
#print('Religion:')
#print(df.religion.value_counts())

# colors = []
# for sex in rows_to_cluster['sex_code']:
#     if sex == 0:
#         colors.append('blue')
#     else:
#         colors.append('red')
# 
# plt.scatter(df['diet_code'], df['body_type_code'], c=colors, alpha=0.01)
# plt.ylabel("Diet")
# plt.xlabel("Body type")
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(df['diet_code'], df['body_type_code'], df['age_scaled'], c=colors)
# ax.set_xlabel('Diet')
# ax.set_ylabel('Body Type')
# ax.set_zlabel('Age (scaled)')
# plt.show()

# plt.scatter(rows_to_cluster['diet_code'], rows_to_cluster['body_type_code'], c=colors, alpha=0.01)
# plt.xlabel("Diet")
# plt.ylabel("Body Type")
# plt.show()