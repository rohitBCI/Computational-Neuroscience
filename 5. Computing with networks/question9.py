import numpy as np

W = np.array([
    [0.6, 0.1, 0.1, 0.1, 0.1],
    [0.1, 0.6, 0.1, 0.1, 0.1],
    [0.1, 0.1, 0.6, 0.1, 0.1],
    [0.1, 0.1, 0.1, 0.6, 0.1],
    [0.1, 0.1, 0.1, 0.1, 0.6]])
print(f"W:\n{W}") # 5x5

u = np.array([[0.6], [0.5], [0.6], [0.2], [0.1]])
print(f"u:\n{u}") # 5x1

M = np.array([
    [-0.125, 0, 0.125, 0.125, 0],
    [0, -0.125, 0, 0.125, 0.125],
    [0.125, 0, -0.125, 0, 0.125],
    [0.125, 0.125, 0, -0.125, 0],
    [0, 0.125, 0.125, 0, -0.125]
])
print(f"M:\n{M}") # 5x5

h = np.dot(W,u)
print(f"h:{h}") # 5x1

ev, e = np.linalg.eig(M) 
print(f"ev:{ev}") # 5,
print(f"e:{e}")   # 5x5
ev = ev.reshape(5,1)

h_e = np.dot(np.transpose(h),e)

h_e_ev = h_e/np.transpose(1-ev)
v_ss= np.dot(h_e_ev,e)
print(v_ss.shape)
print(np.sum(v_ss, axis=0))
# [0.57468284 0.46104003 0.5556308  0.35789389 0.33646672]