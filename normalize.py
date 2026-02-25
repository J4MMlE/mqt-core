import math

vals = [
    (-5.43395386531173, "line 761 lambda (UU test)"),
    (2 * math.pi, "line 777 phi (rotationIdentity test)"),
    (-3.42348659369073, "line 858 lambda (RR test)"),
    (-4.85398163397448, "line 891 phi (U2U2 test)"),
]
for v, label in vals:
    n = math.atan2(math.sin(v), math.cos(v))
    print(f"{label}: {v} -> {n:.15g}")

# line 761 lambda (UU test): -5.43395386531173 -> 0.849231441867857
# line 777 phi (rotationIdentity test): 6.283185307179586 -> -2.44929359829471e-16
# line 858 lambda (RR test): -3.42348659369073 -> 2.85969871348886
# line 891 phi (U2U2 test): -4.85398163397448 -> 1.42920367320511
