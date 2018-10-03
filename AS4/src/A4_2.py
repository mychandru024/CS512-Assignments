import cv2
import numpy as np
import sys

if len(sys.argv) > 1:
	file_name = str(sys.argv[1])
else:
	file_name = input("Data file path : ")

#read data from file
i_p = []
w_p = []

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

#estimate projection matirix M
A = []
for i in range(0,268):
	A.append([ w_p[i][0], w_p[i][1], w_p[i][2], 1, 0, 0, 0, 0, -1 * i_p[i][0] * w_p[i][0], -1 * i_p[i][0] * w_p[i][1], -1 * i_p[i][0] * w_p[i][2], -1 * i_p[i][0] * 1 ])
	A.append([ 0, 0, 0, 0, w_p[i][0], w_p[i][1], w_p[i][2], 1, -1 * i_p[i][1] * w_p[i][0], -1 * i_p[i][1] * w_p[i][1], -1 * i_p[i][1] * w_p[i][2], -1 * i_p[i][1] * 1 ])
	
U, D, V = np.linalg.svd(A, full_matrices=True)
#print(U.shape)
#print(D.shape)
#print(V.shape)

#print(V[11])

M = np.split(V[11], 3)
print("M*: ", M)
print()

p_i_p = []
for i in range(0, 268):
	p = np.matmul(M, [ w_p[i][0], w_p[i][1], w_p[i][2], 1])
	p_i_p.append((p[0]/p[2], p[1]/p[2]))

#print(p_i_p)

#calculate error
err = 0
for i in range(0, 268):
	err += (i_p[i][0] - p_i_p[i][0]) ** 2
	err += (i_p[i][1] - p_i_p[i][1]) ** 2
	
print("Error: ", err)
print()

a1 = [ M[0][0], M[0][1], M[0][2] ]
a2 = [ M[1][0], M[1][1], M[1][2] ]
a3 = [ M[2][0], M[2][1], M[2][2] ]
b = [  M[0][3], M[1][3], M[2][3] ]
#print("a1: ", a1)
#print("a2: ", a2)
#print("a3: ", a3)
#print("b: ", b)

#mag_row = 1/np.sqrt(np.dot(a3, a3))
mag_row = 1/np.linalg.norm(a3)
print("mag_row: ", mag_row)
print()

u0 = mag_row * mag_row * np.dot(a1,a3)
v0 = mag_row * mag_row * np.dot(a2,a3)
print("(u0,v0): ",u0,v0)
print()

alpha_v = np.sqrt(mag_row * mag_row * np.dot(a2,a2 ) - v0 * v0)
#print("alphaV: ", alpha_v)
#print()

#s = (mag_row * mag_row * mag_row * mag_row) / np.dot( alpha_v * np.cross(a1, a3), np.cross(a2, a3) )
s = np.dot( (mag_row * mag_row * mag_row * mag_row) / alpha_v * np.cross(a1, a3), np.cross(a2, a3) )
print("s: ", s)
print()

alpha_u = np.sqrt(mag_row * mag_row * np.dot(a1,a1) - s * s - u0 * u0)
print("(alphaU, alphaV): ", alpha_u, alpha_v)
print()

k_star = [ [alpha_u, s, u0], [0, alpha_v, v0], [0, 0, 1] ]
print("K*: ", k_star)
print()

e = 1
if b[2] < 0:
	e = -1
print("e: ", e)
print()

t_star = e * mag_row * np.matmul(np.linalg.inv(k_star) , b)
print("T*: ", t_star)
print()

r3 = e * mag_row * np.transpose(a3)
#print("r3: ", r3)

r1 = mag_row * mag_row / alpha_v * np.cross(a2, a3)
#print("r1: ", r1)

r2 = np.cross(r3, r1)
#print("r2: ", r2)

r_star = np.matrix([np.transpose(r1), np.transpose(r2), np.transpose(r3)])
print("R*: ", r_star)

