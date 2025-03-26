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

	m = genCircle(dim, 0, 1, (-dim // 6, -dim // 6))
	m += genCircle(dim, 0, 1, (dim // 6, -dim // 6))
	m += genCircle(dim, 0, 1, (dim // 6, dim // 6))
	m += np.random.normal(0, 0.25, (dim, dim)) * 0.1
	m = cv2.GaussianBlur(m, (blur, blur), 0)

	m_min = np.min(m)
	m_max = np.max(m)

	m -= m_min

	if (m_max - m_min) != 0:
		m /= (m_max - m_min)
	
	return m