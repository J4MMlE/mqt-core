from sympy import Quaternion, pi, cos, sin, N


# R(theta, phi): rotation by theta around axis (cos(phi), sin(phi), 0) in XY plane
def r_gate(theta, phi):
    return Quaternion(
        cos(theta / 2), sin(theta / 2) * cos(phi), sin(theta / 2) * sin(phi), 0
    )


# U2(phi, lambda) = U(pi/2, phi, lambda) -> ZYZ decomposition with theta=pi/2
def u2_gate(phi, lam):
    return Quaternion.from_euler([phi, pi / 2, lam], "ZYZ")


# rotation of 1rad around x axis
rx = Quaternion.from_euler([1, 0, 0], "xyz")
# rotation of 1rad around y axis
ry = Quaternion.from_euler([0, 1, 0], "xyz")
# rotation of 1rad around z axis
rz = Quaternion.from_euler([0, 0, 1], "xyz")

# rotation of -1rad around x axis
mx = Quaternion.from_euler([-1, 0, 0], "xyz")
# rotation of -1rad around y axis
my = Quaternion.from_euler([0, -1, 0], "xyz")
# rotation of -1rad around z axis
mz = Quaternion.from_euler([0, 0, -1], "xyz")

# rotation of 1rad around x axis
px = Quaternion.from_euler([pi, 0, 0], "xyz")
# rotation of 1rad around y axis
py = Quaternion.from_euler([0, pi, 0], "xyz")
# rotation of 1rad around z axis
pz = Quaternion.from_euler([0, 0, pi], "xyz")


smallx = Quaternion.from_euler([0.001, 0, 0], "xyz")
smally = Quaternion.from_euler([0, 0.001, 0], "xyz")

# smallx -> smally
q = smally.mul(smallx)
print(f"small rotations: {tuple(axis.evalf() for axis in q.to_euler("ZYZ"))}")

# 3. ry(-1)*rz(-1)*rz(1)*ry(1)
# RY(1)->RZ(1)->RZ(-1)->RY(-1)
# h1 = my.mul(mz)
# h2 = h1.mul(rz)
# h3 = h2.mul(ry)

# h1 = rz.mul(ry)
# h2 = mz.mul(h1)
# h3 = my.mul(h2)

# u1 = rz*ry
# u2 = rz*u1
# u3 = ry*u2

h1 = rz.mul(ry)
h2 = mz.mul(h1)
h3 = my.mul(h2)

hq = Quaternion(h3.a.evalf(), h3.b.evalf(), h3.c.evalf(), h3.d.evalf())
print(
    f"RY(1)->RZ(1)->RY(-1)->RZ(-1): {tuple(axis.evalf() for axis in hq.to_euler("ZYZ"))}"
)

# px -> pz -> py
p1 = py.mul(px)
p2 = pz.mul(p1)
print(f"p {tuple(axis.evalf() for axis in p2.to_euler("ZYZ"))}")
print(f"PI: {pi.evalf()}")
print(f"PI: {(2*pi).evalf()}")


# rx -> ry
xy = ry.mul(rx)
print(f"xy {tuple(axis.evalf() for axis in xy.to_euler("ZYZ"))}")

# rx -> rz
xz = rz.mul(rx)
print(f"xz {tuple(axis.evalf() for axis in xz.to_euler("ZYZ"))}")

# ry -> rx
yx = rx.mul(ry)
print(f"yx {tuple(axis.evalf() for axis in yx.to_euler("ZYZ"))}")

# ry -> rz
yz = rz.mul(ry)
print(f"yz {tuple(axis.evalf() for axis in yz.to_euler("ZYZ"))}")

# rz -> rx
zx = rx.mul(rz)
print(f"zx {tuple(axis.evalf() for axis in zx.to_euler("ZYZ"))}")

# rz -> ry
zx = ry.mul(rz)
print(f"zy {tuple(axis.evalf() for axis in zx.to_euler("ZYZ"))}")


r11 = r_gate(1, 1)
r12 = r_gate(1, 2)  # different phi, non-degenerate
r34 = r_gate(3, 4)  # different theta, same phi

# R(1,1) -> R(1,2)  [apply r11 first, then r12]
prod = r34.mul(r12)
prod_n = Quaternion(N(prod.a), N(prod.b), N(prod.c), N(prod.d))
print(f"R(1,2)->R(3,4): {tuple(N(a) for a in prod_n.to_euler('ZYZ'))}")

# R(1,1) -> R(1,1)  [same axis]
prod2 = r11.mul(r11)
prod2_n = Quaternion(N(prod2.a), N(prod2.b), N(prod2.c), N(prod2.d))
print(f"R(1,1)->R(1,1): {tuple(N(a) for a in prod2_n.to_euler('ZYZ'))}")

# U2(1,2) -> U2(3,4)
u2_12 = u2_gate(1, 2)
u2_34 = u2_gate(3, 4)
prod3 = u2_34.mul(u2_12)
prod3_n = Quaternion(N(prod3.a), N(prod3.b), N(prod3.c), N(prod3.d))
print(f"U2(1,2)->U2(3,4): {tuple(N(a) for a in prod3_n.to_euler('ZYZ'))}")

rx_direct = r_gate(1, 0)
print(f"rx from euler:  {rx}")
print(f"rx direct:      {rx_direct}")
# should match: Quaternion(cos(1/2), sin(1/2), 0, 0)

# u1 = Quaternion.from_euler([1, 2, 3], "ZYZ")
# u2 = Quaternion.from_euler([4, 5, 6], "ZYZ")
u1 = Quaternion.from_euler([2, 1, 3], "ZYZ")
u2 = Quaternion.from_euler([5, 4, 6], "ZYZ")
uu = u2.mul(u1)  # remember reverse order for mul
# U(-0.150768558132143, 2.03289042623884, -4.61935453147843)
print(f"uu {tuple(axis.evalf() for axis in uu.to_euler("ZYZ"))}")
