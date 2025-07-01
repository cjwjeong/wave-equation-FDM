"""
2D Seismic Wave Propagation using Finite Difference Method (FDM)

본 코드는 2차원 탄성파(또는 음파)의 전파를 유한차분법(FDM)을 통해 수치적으로 해석합니다.
중심 위치에 소스를 부여하고, 시간에 따른 파동의 전파를 계산하여 2D 시공간 상에서
seismogram을 구성합니다.

사용된 방정식 (2D Wave Equation):

    ∂²u/∂t² = v² ( ∂²u/∂x² + ∂²u/∂z² ) + f(x, z)·s(t)

여기서:
 - u(x, z, t): 시간과 공간에 따른 변위
 - v: 파동 속도 (상수 vmax 사용)
 - f(x, z): 소스 위치 (delta function 형태)
 - s(t): 시간 도메인 소스 파형 (Gaussian 2차 도함수, fdgaus)

구성:
 - fdgaus(): 주파수 제한 Gaussian 2차 도함수 형태의 소스 생성
 - FDM scheme: 2차 시간 중심, 2차 공간 중심 유한차분 스텐실 사용
 - 경계 조건은 단순 Dirichlet (u=0) 처리
 - seismogram: 모든 시간에 대해 (x, z) 격자상의 파형 기록
 - matplotlib으로 시공간 단면 시각화

파라미터:
 - xmax = zmax = 0.5 (km), dx = dz = 0.003 (km), nx = nz = 167
 - tmax = 0.6 (s), dt = 0.00075 (s), nt = 800
 - vmax = 2.0 (km/s)
 - fmax = 25 (Hz)

시각화:
 - x축: 거리(x), y축: 시간(t)
 - z=1 (표면 근처)에서의 seismogram 출력
 - 회색조 반전('gray_r') colormap과 99% 절댓값 기준 클리핑

작성자: cjwjeong
작성일: 2025-07-01
"""

############## 모듈 임포트 ##############
import numpy as np
import matplotlib.pyplot as plt


############## Gaussian 소스 파형 함수 정의 ##############
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

    # 중심 시간 추정
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

    w /= np.max(np.abs(w))  # 진폭 정규화
    return w


############## 파라미터 설정 ##############
xmax = 0.5;
dx = 0.003;
nx = int(xmax / dx)
zmax = 0.5;
dz = 0.003;
nz = int(zmax / dz)
tmax = 0.6;
dt = 0.00075;
nt = 800
fmax = 25
vmax = 2.0

print("Simulation started...")

############## 격자 및 변수 초기화 ##############
u1 = np.zeros((nx, nz))
u2 = np.zeros((nx, nz))
u3 = np.zeros((nx, nz))
f = np.zeros((nx, nz))
seismogram = np.zeros((nt, nx, nz))

############## 소스 설정 ##############
w = fdgaus(fmax, dt, nt)
f[nx // 2, 1] = 1.0  # 소스 위치 설정 (x 중심, z=1)

############## 메인 타임 루프 ##############
for it in range(nt):
    if it % 100 == 0:
        print(f"{it}/{nt} steps")

    for ix in range(1, nx - 1):
        for iz in range(1, nz - 1):
            laplacian = ((u2[ix + 1, iz] - 2 * u2[ix, iz] + u2[ix - 1, iz]) +
                         (u2[ix, iz + 1] - 2 * u2[ix, iz] + u2[ix, iz - 1]))

            u3[ix, iz] = (vmax ** 2 * dt ** 2 / dx ** 2) * laplacian \
                         + 2 * u2[ix, iz] - u1[ix, iz] \
                         + (vmax ** 2 * dt ** 2) * w[it] * f[ix, iz]

    u1[:, :], u2[:, :] = u2[:, :], u3[:, :]
    seismogram[it, :, :] = u3[:, :]

############## Seismogram 시각화 (z=1 단면) ##############
result = seismogram[:, :, 1]  # z=1 위치에서의 시공간 파형
perc = 99
boundary = max(np.percentile(result, perc), -np.percentile(result, 100 - perc))

plt.figure(figsize=(10, 6))
plt.title("2D Wave Equation FDM Modeling (T-X view at z=1)")
plt.xlabel('Distance (km)')
plt.ylabel('Time (s)')
plt.imshow(result.squeeze(), cmap='gray_r', aspect='auto',
           extent=[0, dx * nx, dt * nt, 0], vmin=-boundary, vmax=boundary)
plt.tight_layout()
plt.show()
