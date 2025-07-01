"""
2D Wave Propagation via Frequency Domain Green's Function Method
(Using Sparse Matrix Solver for Helmholtz Equation)

이 코드는 2차원 파동 방정식을 주파수 영역에서 Green's function을 통해 수치적으로 해석한 후,
시간 영역으로 역변환하여 seismogram을 생성하는 주파수 영역 기반 모델이다.
각 주파수 성분마다 2D Helmholtz 방정식을 희소 행렬 시스템(sparse matrix system)으로 구성하고,
각 위치에 대한 응답을 계산한다.

사용된 방정식 (2D Helmholtz form):

    [-ω²/c² + ∇²] U(x, z, ω) = f(x, z)

- U(x, z, ω): 주파수 영역에서의 파동장
- c: 등속 매질에서의 파동 속도
- f(x, z): 소스 항 (x 중심, 얕은 깊이 위치)
- α: 지수 감쇠 항 (역변환 시 안정성 및 고주파 감쇠를 위함)
- Green's function G(x, ω): z = 2 깊이에서 수신된 응답으로 구성
- 시간 도메인 소스 s(t): Gaussian 2차 도함수 형태, FFT로 변환

구현 개요:
- fdgaus(): Gaussian 2차 도함수 형태의 시간 영역 소스 생성
- 각 주파수별 Helmholtz 방정식을 희소 행렬로 구성 (lil_matrix → csc_matrix)
- scipy의 spsolve()를 이용해 선형 시스템 해 구함
- 수신 깊이(z=2)에서의 각 x 위치 응답을 Green's function으로 저장
- conjugate symmetry 구성 후 ifft를 통해 시간 영역 복원
- 감쇠 계수(alpha)를 곱해 후방 반사 억제

파라미터:
- nx = 200, nz = 100 (격자 수), dx = dz = 0.005 (m)
- velocity = 1.5 m/s (등속)
- G = 16 (최소 격자 수 기준으로 fmax 산정)
- fmax = velocity / (G * dx)
- dt = 0.1 / fmax, tmax = 1.0 s, nt = int(tmax / dt)
- df = 1 / tmax, nf = int(fmax / df)
- alpha = ln(100) / tmax (지수 감쇠 계수)

시각화:
- 시공간 파형 u(x, t)를 시간(t) vs. 거리(x) 형태로 시각화
- 회색조(gray) 색상맵, 상하 반전 없이 자연스러운 시간 축 방향 유지
- 99% 절댓값 기준으로 amplitude clipping

기술적 특징:
- 2D 공간 격자에서 Helmholtz 방정식을 희소행렬(sparse matrix)로 효율적 구성
- 각 주파수마다 독립적 계산 가능 (병렬화 확장 가능)
- Green's function 기반 고해상도 시간-공간 응답 복원 가능

작성자: cjwjeong
작성일: 2025-07-01
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

# Gaussian 소스 파형 정의
def fdgaus(cutoff, dt, nt):
    w = np.zeros(nt)
    pi = np.pi
    a = pi * (5.0 * cutoff / 8.0) ** 2
    amp = np.sqrt(a / pi)
    for i in range(nt):
        t = i * dt
        arg = -a * t**2
        if arg < -32.0:
            arg = -32.0
        w[i] = amp * np.exp(arg)

    # 중심 시간 추정
    for i in range(nt):
        if w[i] < 0.001 * w[0]:
            t0 = i * dt
            break

    for i in range(nt):
        t = i * dt - t0
        arg = -a * t**2
        if arg < -32.0:
            arg = -32.0
        w[i] = -2.0 * np.sqrt(a) * a * t * np.exp(arg) / np.sqrt(pi)

    w /= np.max(np.abs(w))  # 진폭 정규화
    return w

# 파라미터 설정
nx, nz = 200, 100
dx = dz = 0.005
tmax = 1.0
G = 16
velocity = 1.5  # m/s
fmax = velocity / (G * dx)
dt = 0.1 / fmax
nt = int(tmax / dt)
df = 1 / tmax
nf = int(fmax / df)
alpha = np.log(100) / tmax

# 소스 설정
source = fdgaus(fmax, dt, nt)
csource = np.fft.fft(source)

# 수신 데이터 저장 배열
green = np.zeros((nx, nf), dtype=complex)

# 선형 시스템 설정
for ifreq in range(1, nf):
    print(f"Solving frequency index: {ifreq}/{nf-1}")
    w = 2.0 * np.pi * ifreq * df - 1j * alpha
    mat = lil_matrix((nx * nz, nx * nz), dtype=complex)
    f = np.zeros(nx * nz, dtype=complex)

    # 소스 위치 (중앙, 깊이 2)
    src_ix, src_iz = nx // 2, 2
    f[src_ix * nz + src_iz] = 1.0

    # 희소 행렬 구성
    for ix in range(nx):
        for iz in range(nz):
            m = ix * nz + iz
            mat[m, m] = -(w**2 / velocity**2) + 2 / dx**2 + 2 / dz**2

            if iz > 0:
                mat[m, m - 1] = -1 / dz**2
            if iz < nz - 1:
                mat[m, m + 1] = -1 / dz**2
            if ix > 0:
                mat[m, m - nz] = -1 / dx**2
            if ix < nx - 1:
                mat[m, m + nz] = -1 / dx**2

    mat = csc_matrix(mat)
    sol = spsolve(mat, f)

    # 수신 깊이 2에서의 응답 저장
    for ix in range(nx):
        green[ix, ifreq] = sol[ix * nz + 2]

# 시간 영역으로 변환
u = np.zeros((nx, nt), dtype=complex)
for ix in range(nx):
    for ifreq in range(nf):
        u[ix, ifreq] = green[ix, ifreq] * csource[ifreq]
    # 대칭 복소수 성분 구성
    for ifreq in range(1, nf):
        u[ix, nt - ifreq] = np.conj(u[ix, ifreq])

# IFFT 및 감쇠 적용
u = np.fft.ifft(u, axis=1).real / nt
for it in range(nt):
    u[:, it] *= np.exp(alpha * it * dt)

# 시각화
per = 99
bound = max(np.percentile(u, per), -np.percentile(u, 100 - per))
u = np.rot90(u, k=3)  # 자료 회전 (시간축을 수직 방향으로)
plt.imshow(u, cmap='gray', vmin=-bound, vmax=bound, aspect='auto',
           extent=[0, nx * dx, tmax, 0])
plt.xlabel("x (m)")
plt.ylabel("Time (s)")
plt.title("2D Wave Equation Modeling (FDM in Frequency Domain)")
plt.tight_layout()
plt.show()
