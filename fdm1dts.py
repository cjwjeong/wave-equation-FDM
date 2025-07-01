"""
1D Seismic Wave Propagation using Finite Difference Method (FDM)

본 코드는 1차원 탄성파(또는 음파)의 전파를 유한차분법(FDM)을 통해 수치적으로 해석합니다.
공간상에서 주어진 위치에 소스를 부여하고, 시간에 따른 파동의 전파를 계산하며, 결과적으로
seismogram을 출력합니다.

사용된 방정식 (1D Wave Equation):

    ∂²u/∂t² = v² ∂²u/∂x² + f(x)·s(t)

여기서:
 - u(x, t): 시간과 위치에 따른 변위
 - v: 파동 속도 (상수 vmax 사용)
 - f(x): 소스 위치 (delta function 형태)
 - s(t): 시간 도메인 소스 파형 (Gaussian 2차 도함수, fdgaus)

구성:
 - fdgaus(): 주파수 제한 Gaussian 2차 도함수 형태의 소스 생성
 - FDM scheme: 2차 시간 중심, 2차 공간 중심 유한차분 스텐실 사용
 - 경계 조건은 단순 Dirichlet (u=0) 처리
 - seismogram: 각 위치에서 시간에 따른 파형 기록
 - matplotlib으로 시각화

파라미터:
 - xmax = 1.0 (km), dx = 0.005 (km), nx = 200
 - tmax = 1.0 (s), dt = 0.001 (s), nt = 1000
 - vmax = 1.2 (km/s)
 - fmax = 25 (Hz)

시각화:
 - y축: 공간 (거리), x축: 시간
 - 회색조 반전('gray_r') colormap으로 진폭 표현

작성자: 정성원
작성일: 2025-07-01
"""

############## 모듈 정보 ###############
import numpy as np
import matplotlib.pyplot as plt

############## 소스 함수 정의 ###############
def fdgaus(cutoff, dt, nt):
    w = np.zeros(nt)
    phi = 4 * np.arctan(1.0)
    """ a=phi*cutoff**2 """
    a = phi * (5.0 * cutoff / 8.0) ** 2
    amp = np.sqrt(a / phi)
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
        t = i * dt
        t = t - t0
        arg = -a * t ** 2
        if arg < -32.0:
            arg = -32.0
        w[i] = -2.0 * np.sqrt(a) * a * t * np.exp(arg) / np.sqrt(phi)

    smax = np.max(np.abs(w))

    for i in range(nt):
        w[i] = w[i] / smax
    return w

############## 파라미터 ###############
xmax = 1.0 ; dx = 0.005 ; nx = 200
vmax = 1.2
tmax = 1.0 ; dt = 0.001 ; nt = 1000
fmax = 25

############## 초기화 ###############
u1 = np.zeros(nx)
u2 = np.zeros(nx)
u3 = np.zeros(nx)
f = np.zeros(nx)
seismogram = np.zeros((nx, nt))

############## 소스 설정 및 위치 설정 ###############
w = fdgaus(fmax, dt, nt)

f[nx//2] = 1.0

############## 계산 및 실행 ###############
for it in range(nt):
    for ix in range(1, nx - 1): # 유한차분법은 양쪽 끝 값 계산 불가
        u3[ix] = (2 * u2[ix] - u1[ix] +
                  (vmax * dt / dx)**2 * (u2[ix + 1] - 2 * u2[ix] + u2[ix - 1])) + f[ix] * w[it]

    u1[:], u2[:] = u2, u3 #[:]는 내용만 복사 주소는 복사 x
    seismogram[: , it] = u3

# Seismogram 시각화
plt.imshow(seismogram, cmap='gray_r',  aspect='auto',
           extent=[0, dx*nx, dt*nt, 0])
plt.title('1D Seismogram (FDM with fdgaus source)')
plt.xlabel('Time (s)')
plt.ylabel('x_dist (km)')
plt.tight_layout()
plt.show()