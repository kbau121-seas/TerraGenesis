import maya.OpenMaya as om
import maya.OpenMayaMPx as ompx
import maya.cmds as cmds
import maya.utils
import math
import numpy as np
from PIL import Image
import threading

from PySide2 import QtWidgets, QtCore, QtGui
from PySide2.QtCore import Qt
from shiboken2 import wrapInstance
import maya.OpenMayaUI as omui

from scipy import signal
from scipy.ndimage import gaussian_filter

import TerraGenesis

# Utility functions for setting attribute flags
def setInputAttr(attr):
    attr.keyable = True
    attr.storable = True
    attr.readable = True
    attr.writable = True

def setOutputAttr(attr):
    attr.keyable = False
    attr.storable = False
    attr.readable = True
    attr.writable = False

def get_maya_main_window():
    main_window_ptr = omui.MQtUtil.mainWindow()
    return wrapInstance(int(main_window_ptr), QtWidgets.QMainWindow)

class ViewWidget(QtWidgets.QWidget):
    def __init__(self, getter, parent=None):
        super(ViewWidget, self).__init__(parent)

        self.getter = getter

        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        activeMap = self.getter()
        self.setMinimumHeight(activeMap.shape[0])
        self.setMinimumWidth(activeMap.shape[1])
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QtGui.QColor("#1e1e1e"))

        activeMap = self.getter()
        if activeMap is not None:
            activeMap = (activeMap * 255).astype(np.uint8)

            image_h, image_w = activeMap.shape
            widget_w, widget_h = self.width(), self.height()

            # Scaling
            scale = min(widget_w / image_w, widget_h / image_h)

            # Centering offset
            offset_x = (widget_w - (image_w * scale)) / 2
            offset_y = (widget_h - (image_h * scale)) / 2

            # Create QImage
            qimage = QtGui.QImage(
                activeMap.data, image_w, image_h, image_w,
                QtGui.QImage.Format_Grayscale8
            )

            pixmap = QtGui.QPixmap.fromImage(qimage).scaled(
                image_w * scale, image_h * scale,
                QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            )
            painter.drawPixmap(offset_x, offset_y, pixmap)

        painter.end()

class PaintWidget(QtWidgets.QWidget):
    def __init__(self, getters, setters, parent=None):
        super(PaintWidget, self).__init__(parent)

        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self.getters = getters
        self.setters = setters

        self.hardness = 50
        self.strength = 50

        self.setActive(0)

    def setActive(self, index):
        self.active = index

        activeMap = self.getters[index]()
        self.setMinimumHeight(activeMap.shape[0])
        self.setMinimumWidth(activeMap.shape[1])

        self.update()

    def mouseMoveEvent(self, event):
        activeMap = self.getters[self.active]()

        image_h, image_w = activeMap.shape

        # Get scale factor between image and widget
        widget_w, widget_h = self.width(), self.height()
        scale = min(widget_w / image_w, widget_h / image_h)

        # Get top-left origin for centering the image
        offset_x = (widget_w - (image_w * scale)) / 2
        offset_y = (widget_h - (image_h * scale)) / 2

        # Convert widget-space click to image-space
        x = int((event.x() - offset_x) / scale)
        y = int((event.y() - offset_y) / scale)

        sign = 0
        if event.buttons() & Qt.LeftButton:
            sign = 1
        elif event.buttons() & Qt.RightButton:
            sign = -1
        else:
            return

        if 0 <= x < image_w and 0 <= y < image_h:
            deltaMap = np.zeros_like(activeMap)

            sigma = ((self.hardness / 100) ** 2) * 20
            strength = ((self.strength / 100) ** 2) * 20 * sigma

            deltaMap[y, x] += strength * sign
            deltaMap = gaussian_filter(deltaMap, sigma=sigma)

            activeMap = activeMap + deltaMap
            activeMap = activeMap.clip(0, 1)

            self.setters[self.active](activeMap)
            self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QtGui.QColor("#1e1e1e"))

        activeMap = self.getters[self.active]()
        if activeMap is not None:
            activeMap = (activeMap * 255).astype(np.uint8)

            image_h, image_w = activeMap.shape
            widget_w, widget_h = self.width(), self.height()

            # Scaling
            scale = min(widget_w / image_w, widget_h / image_h)

            # Centering offset
            offset_x = (widget_w - (image_w * scale)) / 2
            offset_y = (widget_h - (image_h * scale)) / 2

            # Create QImage
            qimage = QtGui.QImage(
                activeMap.data, image_w, image_h, image_w,
                QtGui.QImage.Format_Grayscale8
            )

            pixmap = QtGui.QPixmap.fromImage(qimage).scaled(
                image_w * scale, image_h * scale,
                QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            )
            painter.drawPixmap(offset_x, offset_y, pixmap)

        painter.end()

class EditorUI(QtWidgets.QDialog):
    def __init__(self, parameters, getters, setters, parent=get_maya_main_window()):
        super(EditorUI, self).__init__(parent)

        self.parameters = parameters
        self.getters = getters
        self.setters = setters

        self.setWindowTitle("Parameter Editor")
        self.setMinimumWidth(350)
        self.setWindowFlags(self.windowFlags() ^ QtCore.Qt.WindowContextHelpButtonHint)

        self.create_widgets()
        self.create_layout()

    def updateStrength(self, value):
        self.painter.strength = value

    def updateHardness(self, value):
        self.painter.hardness = value

    def parameterChanged(self, index):
        self.painter.setActive(index)

    def onHeightChanged(self):
        self.view.update()

        if (self.painter.active == len(self.getters) - 1):
            self.painter.update()

    def create_widgets(self):
        self.parameterSelector = QtWidgets.QComboBox()
        self.parameterSelector.addItems(self.parameters)
        self.parameterSelector.currentIndexChanged.connect(self.parameterChanged)

        self.painter = PaintWidget(self.getters, self.setters)

        self.strengthLabel = QtWidgets.QLabel("Strength")
        self.strengthSlider = QtWidgets.QSlider(Qt.Horizontal)
        self.strengthSlider.setRange(1, 100)
        self.strengthSlider.valueChanged.connect(self.updateStrength)
        self.strengthSlider.setValue(50)

        self.hardnessLabel = QtWidgets.QLabel("Spread")
        self.hardnessSlider = QtWidgets.QSlider(Qt.Horizontal)
        self.hardnessSlider.setRange(1, 100)
        self.hardnessSlider.valueChanged.connect(self.updateHardness)
        self.hardnessSlider.setValue(50)

        self.view = ViewWidget(self.getters[-1])

    def create_layout(self):
        layout = QtWidgets.QVBoxLayout(self)

        layout.addWidget(self.parameterSelector)

        layout.addWidget(self.painter)

        optionsLayout = QtWidgets.QGridLayout()
        optionsLayout.addWidget(self.strengthLabel, 0, 0)
        optionsLayout.addWidget(self.strengthSlider, 0, 1)
        optionsLayout.addWidget(self.hardnessLabel, 1, 0)
        optionsLayout.addWidget(self.hardnessSlider, 1, 1)

        layout.addLayout(optionsLayout)

        layout.addWidget(self.view)

# Helper class to periodically run functions
class RepeatTimer(threading.Timer):
    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)

class TerraGenesisNode(ompx.MPxNode):
    # Define node name and unique ID
    kNodeName = "TerraGenesisNode"
    kNodeId   = om.MTypeId(0x00082000)

    # Attributes
    aUpliftPath       = None    # String attribute: uplift image file path
    aTimeStep         = None    # Float attribute: simulation time step
    aIterations       = None    # Int attribute: number of simulation iterations
    aCurrentIteration = None    # Int attribute: number of processed simulation iterations
    aGridSizeX        = None    # Float attribute: physical grid size in X direction
    aGridSizeZ        = None    # Float attribute: physical grid size in Z direction
    aCellSize         = None    # Float attribute: size of each cell in the grid
    aMeshOutput       = None    # Mesh attribute: output terrain mesh
    aDoRun            = None    # Boolean attribute: whether or not the simulation is running
    aDoReset          = None    # Boolean attribute: whether or not the simulation should reset
    aDoOpenEditor     = None
    aMode             = None
    def __init__(self):
        super(TerraGenesisNode, self).__init__()

        # upliftArray = self.loadUpliftImage("C:\\Users\\Kyle Bauer\\Courses\\CIS6600\\TerraGenesis\\py\\sample_uplift.png", (128, 128))
        upliftArray = np.zeros((128, 128))
        self.mModel = TerraGenesis.Simulator(upliftArray)
        self.mElevationImage = Image.fromarray(self.mModel.heightMap * 255)
        self.mRepeatTimer = RepeatTimer(0, self.testUpdate)
        self.mErosion = np.zeros((128, 128), dtype=np.float32)
        self.isRunning = False

        self.minSlopeDegree = 0
        self.maxSlopeDegree = 4
        self.minDrainageDegree = 0
        self.maxDrainageDegree = 2

    def compute(self, plug, dataBlock):
        # Only compute the output mesh
        if plug != TerraGenesisNode.aMeshOutput and plug.parent() != TerraGenesisNode.aMeshOutput:
            return

        nodeName = "TerraGenesisNode"

        # Retrieve input attribute values
        upliftPath       = dataBlock.inputValue(TerraGenesisNode.aUpliftPath).asString()
        timeStep         = dataBlock.inputValue(TerraGenesisNode.aTimeStep).asFloat()
        iterations       = dataBlock.inputValue(TerraGenesisNode.aIterations).asInt()
        currentIteration = dataBlock.inputValue(TerraGenesisNode.aCurrentIteration).asInt()
        gridX            = dataBlock.inputValue(TerraGenesisNode.aGridSizeX).asFloat()
        gridZ            = dataBlock.inputValue(TerraGenesisNode.aGridSizeZ).asFloat()
        cellSize         = dataBlock.inputValue(TerraGenesisNode.aCellSize).asFloat()
        doRun            = dataBlock.inputValue(TerraGenesisNode.aDoRun).asInt()
        doReset          = dataBlock.inputValue(TerraGenesisNode.aDoReset).asInt()
        doOpenEditor     = dataBlock.inputValue(TerraGenesisNode.aDoOpenEditor).asInt()
        modeVal          = dataBlock.inputValue(TerraGenesisNode.aMode).asShort()

        self.mode = modeVal
        self.isRunning = doRun

        # Calculate grid dimensions (ensure a minimum grid of 4x4 cells)
        rows = max(int(math.ceil(gridX / cellSize)), 4)
        cols = max(int(math.ceil(gridZ / cellSize)), 4)
        gridDims = (rows, cols)

        # Load the uplift image and resize it to match the grid dimensions
        #upliftArray = self.loadUpliftImage(upliftPath, gridDims)

        if doOpenEditor:
            self.showEditor()
            cmds.setAttr(nodeName + ".doEditor", 0)

        # Initialize the elevation array with zeros
        elevation = np.zeros(gridDims, dtype=np.float32)

        if doReset:
            # upliftArray = self.loadUpliftImage("C:\\Users\\Kyle Bauer\\Courses\\CIS6600\\TerraGenesis\\py\\sample_uplift.png", (128, 128))
            upliftArray = np.zeros((128, 128))
            self.mModel = TerraGenesis.Simulator(upliftArray)
            self.mElevationImage = Image.fromarray(self.mModel.heightMap * 255)
            self.mErosion=np.zeros_like(self.mModel.heightMap)
            cmds.setAttr(nodeName + ".doReset", 0)

            if (self.ui is not None):
                self.ui.update()

        # Run a basic simulation: for each iteration, add uplift scaled by the time step
        if self.mElevationImage != None:
            resizedElevationImage = self.mElevationImage.resize(gridDims, Image.BILINEAR)
            elevation = np.array(resizedElevationImage, dtype=np.float32) / 255.0

        if (doRun and (not self.mRepeatTimer.is_alive() or self.mRepeatTimer.finished.is_set())):
            self.mRepeatTimer.start()
        elif (not doRun and (self.mRepeatTimer.is_alive() and not self.mRepeatTimer.finished.is_set())):
            self.mRepeatTimer.cancel()
            self.mRepeatTimer = RepeatTimer(0, self.testUpdate)

        # Set the computed mesh as the output
        outHandle:om.MDataHandle = dataBlock.outputValue(TerraGenesisNode.aMeshOutput)
        meshData:om.MObject = om.MFnMeshData().create()
        self.buildMesh(elevation, cellSize,meshData)
        outHandle.setMObject(meshData)
        dataBlock.setClean(plug)
        

    def loadUpliftImage(self, path, dims):
        """
        Loads an image from 'path', converts it to grayscale,
        normalizes the pixel values to [0,1], and resizes it to 'dims'.
        """
        if not path:
            #om.MGlobal.displayWarning("Uplift path is empty.")
            return np.zeros(dims, dtype=np.float32)
        try:
            with Image.open(path) as img:
                #om.MGlobal.displayInfo(f"Uplift image loaded from {path}")
                # Convert the image to grayscale ('L' mode)
                img = img.convert("L")
                imgArray = np.array(img, dtype=np.float32) / 255.0
                # Resize the image to the desired dimensions (PIL expects (width, height))
                resizedImg = Image.fromarray((imgArray * 255).astype(np.uint8)).resize((dims[1], dims[0]), Image.BILINEAR)
                resizedArray = np.array(resizedImg, dtype=np.float32) / 255.0
                return resizedArray
        except Exception as e:
            #om.MGlobal.displayWarning("Error loading uplift image: " + str(e))
            return np.zeros(dims, dtype=np.float32)

    def buildMesh(self, elevation, cellSize,meshdata:om.MObject):
        """
        Constructs a polygon mesh based on the elevation data.
        The grid is centered at the origin on the XZ-plane, and each vertex's Y coordinate
        corresponds to the elevation value.
        """
        rows, cols = elevation.shape
        vertices = []
        polyCounts = []
        polyConnects = []
        # Center the grid on the origin
        originX = -0.5 * cellSize * (rows - 1)
        originZ = -0.5 * cellSize * (cols - 1)
        indexMap = {}

        # Create vertices from the elevation grid
        for i in range(rows):
            x = originX + i * cellSize
            for j in range(cols):
                z = originZ + j * cellSize
                y = elevation[i, j]
                vertices.append(om.MPoint(float(x),float(y), float(z)))
                indexMap[(i, j)] = len(vertices) - 1

        # Create quads for each cell in the grid (except the last row and column)
        for i in range(rows - 1):
            for j in range(cols - 1):
                polyCounts.append(4)
                polyConnects.extend([
                    indexMap[(i, j)],
                    indexMap[(i, j+1)],
                    indexMap[(i+1, j+1)],
                    indexMap[(i+1, j)]
                ])

        # Build the mesh using Maya's API
        pointsArray = om.MPointArray()
        for pt in vertices:
            pointsArray.append(pt)

        countsArray = om.MIntArray()
        for count in polyCounts:
            countsArray.append(count)

        connectsArray = om.MIntArray()
        for idx in polyConnects:
            connectsArray.append(int(idx))
        meshFn = om.MFnMesh()

        numVerts = pointsArray.length()
        numPolys = countsArray.length()

        meshObj = meshFn.create(numVerts, numPolys, pointsArray, countsArray, connectsArray, meshdata)
        wetlandColor = np.array([76.0, 187.0, 23.0]) / 255.0
        peak = np.array([1.0,   1.0,   1.0])
        blackGrey = np.array([0.1, 0.1, 0.1])    
        whiteGrey = np.array([0.4, 0.4, 0.4])    

        vertexColors = om.MColorArray()

        erosion_normalized = (self.mErosion - np.min(self.mErosion)) / (np.max(self.mErosion) - np.min(self.mErosion) + 1e-6)
        max_h = float(np.max(elevation)) if np.max(elevation) > 0 else 1.0
        for i in range(rows):
            for j in range(cols):
                e = float(erosion_normalized[i, j])
                h = float(elevation[i, j]) / max_h 
                if self.mode==0:
                    baseColor = (1.0 - h) * wetlandColor + h * peak
                    base_r, base_g, base_b = baseColor
                    r = (1.0 - e) * base_r + e * 0.0
                    g = (1.0 - e) * base_g + e * 0.0
                    b = (1.0 - e) * base_b + e * 1.0
                else:
                    baseColor = (h) * whiteGrey + (1.0-h) * blackGrey
                    base_r, base_g, base_b = baseColor
                
                    e_vis = e ** 0.5  
                    magmaColor = np.array([0.9, 0.4, 0.1])
                    r = (1.0 - e_vis) * base_r + e_vis * magmaColor[0]
                    g = (1.0 - e_vis) * base_g + e_vis * magmaColor[1]
                    b = (1.0 - e_vis) * base_b + e_vis * magmaColor[2]
           
                color = om.MColor(r,g,b, 1.0)  
                vertexColors.append(color)

        indices = om.MIntArray()
        for idx in range(numVerts):
            indices.append(idx)


        meshFn.setVertexColors(vertexColors, indices)

       

        # meshObj:om.MObject= meshFn.create(pointsArray,countsArray, connectsArray, meshdata)

        return meshObj

    def getUplift_EDITOR(self):
        return self.mModel.upliftMap

    def setUplift_EDITOR(self, value):
        self.mModel.upliftMap = value

    def getErosion_EDITOR(self):
        return self.mModel.erosionScale

    def setErosion_EDITOR(self, value):
        self.mModel.erosionScale = value

    def getSlopeContribution_EDITOR(self):
        degree = self.mModel.steepestSlopeDegree
        percent = (degree - self.minSlopeDegree) / (self.maxSlopeDegree - self.minSlopeDegree)

        return (1 - percent)

    def setSlopeContribution_EDITOR(self, value):
        degree = (1 - value) * (self.maxSlopeDegree - self.minSlopeDegree) + self.minSlopeDegree
        self.mModel.steepestSlopeDegree = degree

    def getDrainageContribution_EDITOR(self):
        degree = self.mModel.drainageDegree
        percent = (degree - self.minDrainageDegree) / (self.maxDrainageDegree - self.minDrainageDegree)

        return percent

    def setDrainageContribution_EDITOR(self, value):
        degree = value * (self.maxDrainageDegree - self.minDrainageDegree) + self.minDrainageDegree
        self.mModel.drainageDegree = degree

    def getHeight_EDITOR(self):
        return self.mModel.heightMap

    def setHeight_EDITOR(self, value):
        self.mModel = value

        self.ui.onHeightChanged()

        if (not self.isRunning):
            maya.utils.executeInMainThreadWithResult(self.testUpdate_main)

    def showEditor(self):
        def _show_ui():
            try:
                for widget in QtWidgets.QApplication.allWidgets():
                    if isinstance(widget, EditorUI):
                        widget.close()
            except:
                pass

            self.ui = EditorUI(
                ["Uplift", "Erosion", "Slope Contribution", "Drainage Contribution", "Height"],
                [self.getUplift_EDITOR, self.getErosion_EDITOR, self.getSlopeContribution_EDITOR, self.getDrainageContribution_EDITOR, self.getHeight_EDITOR],
                [self.setUplift_EDITOR, self.setErosion_EDITOR, self.setSlopeContribution_EDITOR, self.setDrainageContribution_EDITOR, self.setHeight_EDITOR])
            self.ui.show()

        maya.utils.executeDeferred(_show_ui)

    @staticmethod
    def creator():
        return TerraGenesisNode()

    @staticmethod
    def initialize():
        # Create numeric and typed attribute functions
        numAttrFn = om.MFnNumericAttribute()
        typeAttrFn = om.MFnTypedAttribute()
        strDataFn = om.MFnStringData()
        eAttr = om.MFnEnumAttribute() 
        # Uplift Path attribute (string, used as filename)
        defaultStr = strDataFn.create("")
        TerraGenesisNode.aUpliftPath = typeAttrFn.create("upliftFile", "upf", om.MFnData.kString, defaultStr)
        setInputAttr(typeAttrFn)
        typeAttrFn.usedAsFilename = True
        ompx.MPxNode.addAttribute(TerraGenesisNode.aUpliftPath)

        # Time Step attribute (float)
        TerraGenesisNode.aTimeStep = numAttrFn.create("timeStep", "ts", om.MFnNumericData.kFloat, 0.1)
        setInputAttr(numAttrFn)
        numAttrFn.setMin(0.0)
        numAttrFn.setHidden(True)
        ompx.MPxNode.addAttribute(TerraGenesisNode.aTimeStep)

        # Iterations attribute (int)
        TerraGenesisNode.aIterations = numAttrFn.create("iterations", "iter", om.MFnNumericData.kInt, 10)
        setInputAttr(numAttrFn)
        numAttrFn.setMin(0)
        numAttrFn.setHidden(True)
        ompx.MPxNode.addAttribute(TerraGenesisNode.aIterations)

        # Current Iteration attribute (int)
        TerraGenesisNode.aCurrentIteration = numAttrFn.create("currentIteration", "prog", om.MFnNumericData.kInt, 0)
        setInputAttr(numAttrFn)
        numAttrFn.setMin(0)
        numAttrFn.setHidden(True)
        ompx.MPxNode.addAttribute(TerraGenesisNode.aCurrentIteration)

        # Grid Size X attribute (float)
        TerraGenesisNode.aGridSizeX = numAttrFn.create("gridSizeX", "gsx", om.MFnNumericData.kFloat, 10.0)
        setInputAttr(numAttrFn)
        numAttrFn.setMin(0.0)
        ompx.MPxNode.addAttribute(TerraGenesisNode.aGridSizeX)

        # Grid Size Z attribute (float)
        TerraGenesisNode.aGridSizeZ = numAttrFn.create("gridSizeZ", "gsz", om.MFnNumericData.kFloat, 10.0)
        setInputAttr(numAttrFn)
        numAttrFn.setMin(0.0)
        ompx.MPxNode.addAttribute(TerraGenesisNode.aGridSizeZ)

        # Cell Size attribute (float)
        TerraGenesisNode.aCellSize = numAttrFn.create("cellSize", "cs", om.MFnNumericData.kFloat, 0.5)
        setInputAttr(numAttrFn)
        numAttrFn.setMin(0.001)
        ompx.MPxNode.addAttribute(TerraGenesisNode.aCellSize)

        # Is Running attribute (Bool)
        TerraGenesisNode.aDoRun = numAttrFn.create("doRun", "run", om.MFnNumericData.kInt, 0)
        setInputAttr(numAttrFn)
        numAttrFn.setHidden(True)
        ompx.MPxNode.addAttribute(TerraGenesisNode.aDoRun)

        # Should Reset attribute (Bool)
        TerraGenesisNode.aDoReset = numAttrFn.create("doReset", "reset", om.MFnNumericData.kInt, 0)
        setInputAttr(numAttrFn)
        numAttrFn.setHidden(True)
        ompx.MPxNode.addAttribute(TerraGenesisNode.aDoReset)

        # Do Open Editor attribute (Bool)
        TerraGenesisNode.aDoOpenEditor = numAttrFn.create("doEditor", "edit", om.MFnNumericData.kInt, 0)
        setInputAttr(numAttrFn)
        numAttrFn.setHidden(True)
        ompx.MPxNode.addAttribute(TerraGenesisNode.aDoOpenEditor)

        # Output Mesh attribute (mesh data)
        TerraGenesisNode.aMeshOutput = typeAttrFn.create("outputMesh", "out", om.MFnData.kMesh)
        setOutputAttr(typeAttrFn)
        typeAttrFn.setHidden(True)
        ompx.MPxNode.addAttribute(TerraGenesisNode.aMeshOutput)


        TerraGenesisNode.aMode = eAttr.create("mode", "md", 0)
        eAttr.addField("Normal", 0)
        eAttr.addField("Volcano", 1)
        setInputAttr(eAttr)                        
        ompx.MPxNode.addAttribute(TerraGenesisNode.aMode)

        # Define attribute affects so that changes in any input update the mesh
        ompx.MPxNode.attributeAffects(TerraGenesisNode.aUpliftPath, TerraGenesisNode.aMeshOutput)
        ompx.MPxNode.attributeAffects(TerraGenesisNode.aTimeStep, TerraGenesisNode.aMeshOutput)
        ompx.MPxNode.attributeAffects(TerraGenesisNode.aIterations, TerraGenesisNode.aMeshOutput)
        ompx.MPxNode.attributeAffects(TerraGenesisNode.aCurrentIteration, TerraGenesisNode.aMeshOutput)
        ompx.MPxNode.attributeAffects(TerraGenesisNode.aGridSizeX, TerraGenesisNode.aMeshOutput)
        ompx.MPxNode.attributeAffects(TerraGenesisNode.aGridSizeZ, TerraGenesisNode.aMeshOutput)
        ompx.MPxNode.attributeAffects(TerraGenesisNode.aCellSize, TerraGenesisNode.aMeshOutput)
        ompx.MPxNode.attributeAffects(TerraGenesisNode.aDoRun, TerraGenesisNode.aMeshOutput)
        ompx.MPxNode.attributeAffects(TerraGenesisNode.aDoReset, TerraGenesisNode.aMeshOutput)
        ompx.MPxNode.attributeAffects(TerraGenesisNode.aDoOpenEditor, TerraGenesisNode.aMeshOutput)
        ompx.MPxNode.attributeAffects(TerraGenesisNode.aMode, TerraGenesisNode.aMeshOutput)


    def testUpdate_main(self):
        self.mElevationImage = Image.fromarray(self.mModel.heightMap * 255)
        #self.mErosion=self.mModel.erosion
        nodeName = "TerraGenesisNode"
        currentIteration = cmds.getAttr(nodeName + ".currentIteration")
        cmds.setAttr(nodeName + ".currentIteration", currentIteration + 1)

        if self.ui is not None:
            self.ui.onHeightChanged()

    def testUpdate(self):
        self.mModel.run(1)

        maya.utils.executeInMainThreadWithResult(self.testUpdate_main)

def create_terrain_node():
    """
    Helper function to create an instance of the TerraGenesisNode in the scene.
    If a node with the same name exists, it is deleted first.
    """
    nodeName = "TerraGenesisNode"
    if cmds.objExists(nodeName):
        cmds.delete(nodeName)
    newNode = cmds.createNode(TerraGenesisNode.kNodeName, name=nodeName)
    om.MGlobal.displayInfo("Created node: " + newNode)
    transformName = cmds.createNode("transform", name="terrainTransform#")
    meshShape = cmds.createNode("mesh", parent=transformName, name="terrainMesh#")
    # connect output mesh to inmesh
    cmds.connectAttr(newNode + ".outputMesh", meshShape + ".inMesh", force=True)
    shaderName = cmds.shadingNode("lambert", asShader=True, name="grayShader#")
    cmds.setAttr(shaderName + ".color", 0.5, 0.5, 0.5, type="double3")
   
    shadingGroup = cmds.sets(renderable=True, noSurfaceShader=True, empty=True, name=shaderName + "SG")
    cmds.connectAttr(shaderName + ".outColor", shadingGroup + ".surfaceShader", force=True)

    cmds.sets(meshShape, edit=True, forceElement=shadingGroup)
    return newNode

# Plugin initialization functions
def initializePlugin(mObject):
    om.MGlobal.displayInfo("Created node: ")
    plugin = ompx.MFnPlugin(mObject, "MyCompany", "1.0", "Any")
    try:
        plugin.registerNode(TerraGenesisNode.kNodeName,
                              TerraGenesisNode.kNodeId,
                              TerraGenesisNode.creator,
                              TerraGenesisNode.initialize,
                              ompx.MPxNode.kDependNode)
    except Exception as e:
        om.MGlobal.displayError("Error registering node: " + str(e))

    create_terrain_node()

def uninitializePlugin(mObject):
    plugin = ompx.MFnPlugin(mObject)
    try:
        plugin.deregisterNode(TerraGenesisNode.kNodeId)
    except Exception as e:
        om.MGlobal.displayError("Error deregistering node: " + str(e))