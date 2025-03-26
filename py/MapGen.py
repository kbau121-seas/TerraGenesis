import numpy as np
import cv2

def genCircle(dim=512, inner=80, outer=100, offset=(0, 0)):
	hdim = dim // 2
	circ = np.zeros((dim, dim))

	xv, yv = np.meshgrid(np.linspace(0, dim, dim), np.linspace(0, dim, dim))
	xv -= hdim - offset[0]
	yv -= hdim - offset[1]
	sq = xv * xv + yv * yv

	fill = np.logical_and(pow(inner, 2) < sq, sq < pow(outer, 2))
	circ[fill] = 1

	return circ

def genCircleLinearFalloff(dim=512, outer=100, offset=(0, 0)):
	hdim = dim // 2
	circ = np.zeros((dim, dim))

	xv, yv = np.meshgrid(np.linspace(0, dim - 1, dim), np.linspace(0, dim - 1, dim))
	xv -= hdim - offset[0]
	yv -= hdim - offset[1]
	sq = xv * xv + yv * yv

	circ = np.clip((outer - np.sqrt(sq)) / outer, 0, 1)

	return circ

def genMap(dim=512, blur=5):
	blur = blur * 2 + 1

	m = genCircle(dim)
	m += m * np.random.normal(0, 0.25, (dim, dim)) * 4
	m += np.random.normal(0, 0.25, (dim, dim))
	m = cv2.GaussianBlur(m, (blur, blur), 0)

	m_min = np.min(m)
	m_max = np.max(m)

	m -= m_min

	if (m_max - m_min) != 0:
		m /= (m_max - m_min)
	
	return m

def genSimple(dim=512, blur=50):
	blur = blur * 2 + 1

	mix = 0.2
	m = np.ones((dim, dim))
	circs = genCircleLinearFalloff(dim, 50, (-dim // 6, -dim // 6))
	circs = np.maximum(circs, genCircleLinearFalloff(dim, 50, (dim // 6, -dim // 6)))
	circs = np.maximum(circs, genCircleLinearFalloff(dim, 50, (dim // 6, dim // 6)))
	circs = np.pow(circs, 3) * 0.5
	noise = cv2.GaussianBlur(np.random.normal(-0.015, 0.25, (dim, dim)) * 5, (blur, blur), 0)

	m = m * mix + circs * (1 - mix) + noise * 0.2

	m = m - np.min(m)
	m = np.clip(m, 0, 1)
	
	'''
	m_min = np.min(m)
	m_max = np.max(m)

	m -= m_min

	if (m_max - m_min) != 0:
		m /= (m_max - m_min)
	'''
	
	return m