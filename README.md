# 1D & 2D Wave Propagation Modeling  
시간 영역 / 주파수 영역 기반의 파동 방정식 시뮬레이션  
(FDM 및 Green's Function 접근법 포함)

## 개요

이 프로젝트는 1차원 및 2차원 등속 매질에서의 파동 전파(Wave Propagation)를  
시간 영역(Time Domain)과 주파수 영역(Frequency Domain) 모두에서 수치적으로 시뮬레이션합니다.

- 시간 영역: 유한차분법(FDM) 기반
- 주파수 영역: Helmholtz 방정식 + Green’s Function + 희소 행렬 기반

---

## 시뮬레이션 구성

| 시뮬레이션 종류 | 해석 방식        | 공간 차원 | 주요 기법                      |
|----------------|------------------|-----------|-------------------------------|
| 1D 시간 영역    | 시간 영역 (FDM)   | 1D        | 2차 중심 차분                 |
| 2D 시간 영역    | 시간 영역 (FDM)   | 2D        | 2D Laplacian 사용             |
| 1D 주파수 영역 | 주파수 영역       | 1D        | Green's Function 해석         |
| 2D 주파수 영역 | 주파수 영역 + 희소 행렬 | 2D        | Helmholtz + spsolve() 사용     |

---

## 사용된 방정식

### 시간 영역 (1D/2D)

\[
\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u + f(x, t)
\]

- \( u(x,t) \): 시간과 공간에 따른 변위
- \( c \): 파동 속도 (등속)
- \( f(x,t) \): 시간 도메인 소스 (Gaussian 2차 도함수)

### 주파수 영역 (1D/2D Helmholtz 형태)

\[
\left( -\frac{\omega^2}{c^2} + \nabla^2 \right) U(x, \omega) = f(x)
\]

- 각 주파수마다 Green's function 계산 후, IFFT로 시간 영역 복원

---

## 구현 기술

- `fdgaus()`: Gaussian 2차 도함수 기반 소스 파형 생성
- `np.fft.fft`, `np.fft.ifft`: 시간 ↔ 주파수 변환
- `scipy.sparse.lil_matrix`, `csc_matrix`: 희소 행렬 구성
- `scipy.sparse.linalg.spsolve`: 희소 선형 시스템 해법
- `np.percentile`: 고진폭 클리핑 (시각화용)
- `matplotlib.pyplot.imshow`: Seismogram 시각화

---

## 시각화

- X축: 거리 (m 또는 km)
- Y축: 시간 (s)
- 컬러맵: `gray`, `gray_r`, `binary` 등 선택
- 진폭 클리핑: 99% 절댓값 기준으로 표현 범위 제한

---

## 예시 코드 구성

| 파일명                   | 설명                                      |
|--------------------------|-------------------------------------------|
| `fdm_1d_time.py`         | 1D 시간 영역 유한차분법 모델               |
| `fdm_2d_time.py`         | 2D 시간 영역 유한차분법 모델               |
| `green_1d_freq.py`       | 1D 주파수 영역 Green’s Function 모델       |
| `green_2d_freq_sparse.py`| 2D Helmholtz + 희소 행렬 기반 모델         |

---

## 실행 환경

- Python 3.8 이상
- NumPy
- SciPy
- Matplotlib

설치:
