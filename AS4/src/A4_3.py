import cv2
import numpy as np
import random
import sys

#default parameters to use
d = 6
n = 10
w = 0.5
p = 0.98
t = 0.25

with open("..\data\RANSAC.config", 'r',encoding="utf-8") as conf_file:
	for line in conf_file:
		line = line.strip('\n')
		data = line.split("=")
		if data[0] == "d ":
			d = int(data[1])
		elif data[0] == "n ":
			n = int(data[1])
		elif data[0] == "w ":
			w = float(data[1])
		elif data[0] == "p ":
			p = float(data[1])
		elif data[0] == "t ":
			t = float(data[1])
			
print("Using parameters: \n")
print("d: ",d,"\n")
print("n: ",n,"\n")
print("w: ",w,"\n")
print("p: ",p,"\n")
print("t: ",t,"\n")
i_p = []
w_p = []

if len(sys.argv) > 1:
	file_name = str(sys.argv[1])
else:
	file_name = input("Data file path : ")

#file_name = "points_csv.csv"
#file_name = "Noisy1.csv"
#file_name = "Noisy2.csv"

with open(file_name,"r", encoding = "utf-8") as d_f:
	for line in d_f.readlines():
		line = line.strip('\n')
		data = line.split(",")
		if len(data) < 5:
			continue
		w_p.append((float(data[0]), float(data[1]), float(data[2])))
		i_p.append((float(data[3]), float(data[4])))

def compute_k(pW, pP, pN):
	return np.log(1 - pP) / np.log(1 - pW ** pN)

def compute_model_and_find_inliers(points):
	global t
	#print(points)
	A = []
	
	#fit the model based on randomly selected points
	for j in points:
		A.append([ w_p[j][0], w_p[j][1], w_p[j][2], 1, 0, 0, 0, 0, -1 * i_p[j][0] * w_p[j][0], -1 * i_p[j][0] * w_p[j][1], -1 * i_p[j][0] * w_p[j][2], -1 * i_p[j][0] * 1 ])
		A.append([ 0, 0, 0, 0, w_p[j][0], w_p[j][1], w_p[j][2], 1, -1 * i_p[j][1] * w_p[j][0], -1 * i_p[j][1] * w_p[j][1], -1 * i_p[j][1] * w_p[j][2], -1 * i_p[j][1] * 1 ])
	
	U, D, V = np.linalg.svd(A, full_matrices=True)
	M = np.split(V[11], 3)
	
	distances = []
	inliers = []
	for l in range(0, 268):
		p = np.matmul(M, [ w_p[l][0], w_p[l][1], w_p[l][2], 1])
		#calculate distance
		X_err = (i_p[l][0] - p[0]/p[2]) ** 2
		Y_err = (i_p[i][1] - p[1]/p[2]) ** 2
		dist = np.sqrt(X_err + Y_err)
		distances.append(dist)
		if dist < t:
			inliers.append(l)
	distances.sort()
	t = 1.5 * np.median(distances)
	#print(inliers)
	return inliers

best_model_data = []
max_inliers = 0
best_t = t
k = int(compute_k(w, p, n))
for i in range(0, k):
	#selection points at random
	select = []
	for i in range(0, n):
		select.append(random.randint(0,267))
	#print(select)
	random_fit_inliers = compute_model_and_find_inliers(select)
	#print(random_fit_inliers)
	"""
	#fit the model based on randomly selected points
	for j in select:
		A.append([ w_p[j][0], w_p[j][1], w_p[j][2], 1, 0, 0, 0, 0, -1 * i_p[j][0] * w_p[j][0], -1 * i_p[j][0] * w_p[j][1], -1 * i_p[j][0] * w_p[j][2], -1 * i_p[j][0] * 1 ])
		A.append([ 0, 0, 0, 0, w_p[j][0], w_p[j][1], w_p[j][2], 1, -1 * i_p[j][1] * w_p[j][0], -1 * i_p[j][1] * w_p[j][1], -1 * i_p[j][1] * w_p[j][2], -1 * i_p[j][1] * 1 ])
	U, D, V = np.linalg.svd(A, full_matrices=True)
	M = np.split(V[11], 3)
	
	inliers = []
	for l in range(0, 268):
		p = np.matmul(M, [ w_p[l][0], w_p[l][1], w_p[l][2], 1])
		#calculate distance
		X_err += (i_p[l][0] - p[0]/p[2]) ** 2
		y_err += (i_p[i][1] - p[1]/p[2]) ** 2
		dist = np.sqrt(X_err + Y_err)
		if dist < t:
			inliers.append(l)
	"""
	if len(random_fit_inliers) > d:
		recompute_model_inliers = compute_model_and_find_inliers(random_fit_inliers)
		
		w = len(recompute_model_inliers)/268
		k = int(compute_k(w, p, n))
		
		if len(recompute_model_inliers) > max_inliers:
			max_inliers = len(recompute_model_inliers)
			best_model_data = recompute_model_inliers[:]
			best_t = t
	#else:
		#w = len(random_fit_inliers)/268
		#k = int(compute_k(w, p, n))
	
print("Best Model:\n")	
print("Inliers: ", max_inliers)
print("Model data: ", best_model_data)
print("Best t: ",best_t)
print("Final estimate of k: ", k)
print("Final estimate of w: ", w)
		
	
		
	
	
	