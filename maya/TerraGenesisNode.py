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

class PaintWidget(QtWidgets.QWidget):
    def __init__(self, upliftMap, callback, parent=None):
        super(PaintWidget, self).__init__(parent)
        self.setMinimumHeight(upliftMap.shape[0])
        self.setMinimumWidth(upliftMap.shape[1])
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self.points = []
        self.upliftMap = (upliftMap * 255).astype(np.uint8)
        self.callback = callback
        
        brushWidth = 15
        gkern1d = signal.windows.gaussian(brushWidth, std=3).reshape(brushWidth, 1)
        self.brush = np.outer(gkern1d, gkern1d)
        self.brush = self.brush / np.max(self.brush)

    def mouseMoveEvent(self, event):
        image_h, image_w = self.upliftMap.shape

        # Get scale factor between image and widget
        widget_w, widget_h = self.width(), self.height()
        scale = min(widget_w / image_w, widget_h / image_h)

        # Get top-left origin for centering the image
        offset_x = (widget_w - (image_w * scale)) / 2
        offset_y = (widget_h - (image_h * scale)) / 2

        # Convert widget-space click to image-space
        x = int((event.x() - offset_x) / scale)
        y = int((event.y() - offset_y) / scale)

        if 0 <= x < image_w and 0 <= y < image_h:
            # self.points.append((x, y))
            self.upliftMap[y-7:y+8, x-7:x+8] += (self.brush * 5).astype(np.uint8)
            self.callback(self.upliftMap.astype(np.float32) / 255)
            self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QtGui.QColor("#1e1e1e"))

        if self.upliftMap is not None:
            image_h, image_w = self.upliftMap.shape
            widget_w, widget_h = self.width(), self.height()

            # Scaling
            scale = min(widget_w / image_w, widget_h / image_h)

            # Centering offset
            offset_x = (widget_w - (image_w * scale)) / 2
            offset_y = (widget_h - (image_h * scale)) / 2

            # Create QImage
            qimage = QtGui.QImage(
                self.upliftMap.data, image_w, image_h, image_w,
                QtGui.QImage.Format_Grayscale8
            )

            pixmap = QtGui.QPixmap.fromImage(qimage).scaled(
                image_w * scale, image_h * scale,
                QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            )
            painter.drawPixmap(offset_x, offset_y, pixmap)

            # Draw Points
            point_radius = 4
            pen = QtGui.QPen(QtCore.Qt.green)
            brush = QtGui.QBrush(QtCore.Qt.green)
            painter.setPen(pen)
            painter.setBrush(brush)

            for x, y in self.points:
                px = x * scale + offset_x
                py = y * scale + offset_y
                painter.drawEllipse(QtCore.QPointF(px, py), point_radius, point_radius)

        painter.end()

class EditorUI(QtWidgets.QDialog):
    def __init__(self, upliftMap, callback, parent=get_maya_main_window()):
        super(EditorUI, self).__init__(parent)

        self.upliftMap = upliftMap
        self.callback = callback

        self.setWindowTitle("Parameter Editor")
        self.setMinimumWidth(350)
        self.setWindowFlags(self.windowFlags() ^ QtCore.Qt.WindowContextHelpButtonHint)

        self.create_widgets()
        self.create_layout()

    def create_widgets(self):
        self.painter = PaintWidget(self.upliftMap, self.callback)

    def create_layout(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.painter)

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

    def __init__(self):
        super(TerraGenesisNode, self).__init__()

        # upliftArray = self.loadUpliftImage("C:\\Users\\Kyle Bauer\\Courses\\CIS6600\\TerraGenesis\\py\\sample_uplift.png", (128, 128))
        upliftArray = np.zeros((128, 128))
        self.mModel = TerraGenesis.Simulator(upliftArray)
        self.mElevationImage = Image.fromarray(self.mModel.heightMap * 255)
        self.mRepeatTimer = RepeatTimer(0, self.testUpdate)

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
            cmds.setAttr(nodeName + ".doReset", 0)

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

       

        # meshObj:om.MObject= meshFn.create(pointsArray,countsArray, connectsArray, meshdata)

        return meshObj

    def editorCallback(self, upliftMap):
        self.mModel.upliftMap = upliftMap
        return

    # def showEditor(self):
    #     try:
    #         for widget in QtWidgets.QApplication.allWidgets():
    #             if isinstance(widget, EditorUI):
    #                 widget.close()
    #     except:
    #         pass

    #     ui = EditorUI(self.mModel.upliftMap, self.editorCallback)
    #     ui.show()

    def showEditor(self):
        def _show_ui():
            try:
                for widget in QtWidgets.QApplication.allWidgets():
                    if isinstance(widget, EditorUI):
                        widget.close()
            except:
                pass

            self.ui = EditorUI(self.mModel.upliftMap, self.editorCallback)
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

    def testUpdate_main(self):
        self.mElevationImage = Image.fromarray(self.mModel.heightMap * 255)

        nodeName = "TerraGenesisNode"
        currentIteration = cmds.getAttr(nodeName + ".currentIteration")
        cmds.setAttr(nodeName + ".currentIteration", currentIteration + 1)

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