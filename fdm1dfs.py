"""
1D Wave Propagation via Frequency Domain Green's Function Method
(Using Sparse Matrix System Solver)

이 코드는 1차원 파동 방정식을 주파수 영역에서 Green's function을 이용해 해석적으로 계산하고,
시간 영역으로 역변환하여 seismogram을 얻는 주파수 영역 해석 기반 모델이다.
특히, 공간 해석 시 희소행렬(sparse matrix)을 구성하여 효율적인 선형 시스템 해를 구한다.

사용된 방정식 (Helmholtz-like form):

    [-ω²/c² + ∂²/∂x²] U(x, ω) = f(x)

변수 설명:
- U(x, ω): 주파수 영역에서의 파동장
- c: 파동 속도 (등속, 상수)
- f(x): 소스 항 (중앙에 위치한 delta source)
- α: 지수 감쇠 계수 (시간 안정화 및 에너지 감쇠 모사)
- Green's function G(x, ω): 희소행렬을 통한 주파수별 응답
- 시간 도메인 소스 s(t)는 Gaussian 2차 도함수로 구성한 후 FFT 수행

주요 구현 사항:
- fdgaus(): Gaussian 2차 도함수 기반 소스 생성
- 희소행렬 구성: scipy.sparse.lil_matrix → csc_matrix 변환 후 spsolve() 사용
- 주파수 영역 응답 계산 후 conjugate symmetry를 적용하여 시간 영역 복원
- np.fft.ifft()를 통해 시간 영역으로 변환, 이후 감쇠 계수 적용

해석 파라미터:
- nx = 400, dx = 0.01 (격자 수 및 간격)
- c = 2.0 (m/s, 등속 매질)
- fmax = 20 Hz, tmax = 2.0 s
- dt = 0.1 / fmax, nt = int(tmax / dt)
- df = 1 / tmax, nf = int(fmax / df)
- alpha = ln(100) / tmax (지수 감쇠 계수)

시각화:
- x축: 거리 (m)
- y축: 시간 (s)
- 출력은 시간에 따른 공간 위치별 파동장의 진폭을 grayscale로 표현

작성자: 정성원
작성일: 2025-07-01
"""

############## 모듈 임포트 ##############
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve


############## Gaussian 소스 파형 정의 ##############
def fdgaus(cutoff, dt, nt):
    w = np.zeros(nt)
    pi = np.pi
    a = pi * (5.0 * cutoff / 8.0) ** 2
    amp = np.sqrt(a / pi)

    for i in range(nt):
        t = i * dt
        arg = -a * t ** 2
        if arg < -32.0:
            arg = -32.0
        w[i] = amp * np.exp(arg)

    for i in range(nt):
        if w[i] < 0.001 * w[0]:
            icut = i
            t0 = icut * dt
            break

    for i in range(nt):
        t = i * dt - t0
        arg = -a * t ** 2
        if arg < -32.0:
            arg = -32.0
        w[i] = -2.0 * np.sqrt(a) * a * t * np.exp(arg) / np.sqrt(pi)

    w /= np.max(np.abs(w))
    return w


############## 파라미터 설정 ##############
nx = 400;
dx = 0.01
c = 2.0
fmax = 20
tmax = 2.0
dt = 0.1 / fmax
nt = int(tmax / dt)
df = 1 / tmax
nf = int(fmax / df)
alpha = np.log(100) / tmax

print(f"nf : {nf}개")

############## 변수 초기화 ##############
green = np.zeros((nx, nf), dtype=complex)
f = np.zeros(nx, dtype=complex)
mat = lil_matrix((nx, nx), dtype=complex)
u = np.zeros((nx, nt), dtype=complex)

############## 소스 생성 및 FFT 변환 ##############
csource = fdgaus(fmax, dt, nt)
csource = np.fft.fft(csource)

############## 그린 함수 계산 (주파수별) ##############
for ifreq in range(1, nf):
    print(ifreq)
    w = 2.0 * np.pi * ifreq * df - 1j * alpha
    f[:] = 0.0
    f[nx // 2] = 1.0  # 소스 중앙

    mat[0, 0] = -(w ** 2 / c ** 2) + 2.0 / dx ** 2
    mat[1, 0] = -1.0 / dx ** 2
    mat[-1, -1] = -(w ** 2 / c ** 2) + 2.0 / dx ** 2
    mat[-2, -1] = -1.0 / dx ** 2

    for i in range(1, nx - 1):
        mat[i, i] = -(w ** 2 / c ** 2) + 2.0 / dx ** 2
        mat[i + 1, i] = -1.0 / dx ** 2
        mat[i - 1, i] = -1.0 / dx ** 2

    mat = csc_matrix(mat)  # 희소행렬 포맷 변경
    green[:, ifreq] = spsolve(mat, f)

############## 주파수 도메인 파형 조합 ##############
for ix in range(nx):
    for ifreq in range(nf):
        u[ix, ifreq] = green[ix, ifreq] * csource[ifreq]
    for ifreq in range(1, nf):
        u[ix, nt - ifreq] = np.conj(u[ix, ifreq])  # 대칭 구성

############## 시간 영역으로 역변환 및 감쇠 보정 ##############
u = np.fft.ifft(u, axis=1).real / nt

for it in range(nt):
    u[:, it] *= np.exp(alpha * it * dt)

############## 시각화 ##############
plt.imshow(np.real(u), cmap='binary', aspect='auto', extent=[0, nx * dx, 0, tmax])
plt.ylabel('Distance (m)')
plt.xlabel('Time (s)')
plt.title("1D Wave Propagation (Frequency Domain Solution)")
plt.tight_layout()
plt.show()